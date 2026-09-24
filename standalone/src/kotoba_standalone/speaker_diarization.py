from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kotoba_standalone.pyannote_vad import (
    BUNDLED_PYANNOTE_MODEL_CHECKPOINT,
    _allow_trusted_pyannote_checkpoint_globals,
    _disable_matplotlib_imports,
    _trusted_pyannote_checkpoint_load_compat,
)
from kotoba_standalone.subtitle import SubtitleChunk, _join_text, group_chunks_by_timing


DEFAULT_SPEAKER_EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"


class SpeakerDiarizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SpeakerDiarizationResult:
    turns: list[SpeakerTurn]
    model: str
    embedding: str
    speaker_count: int


def diarize_speakers_pyannote(
    wav_path: Path,
    *,
    device: str = "cuda:0",
    num_speakers: int | None = None,
) -> SpeakerDiarizationResult:
    """Run the pyannote 3.1 recipe with the bundled segmentation checkpoint.

    The official 3.1 pipeline config is gated. Its segmentation and embedding
    components can be loaded individually with the same published parameters.
    """
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("num_speakers must be positive")
    if not BUNDLED_PYANNOTE_MODEL_CHECKPOINT.is_file():
        raise SpeakerDiarizationError("Bundled pyannote segmentation checkpoint is missing")
    try:
        import torch

        _disable_matplotlib_imports()
        from pyannote.audio import Model
        from pyannote.audio.pipelines import SpeakerDiarization
    except ImportError as exc:
        raise SpeakerDiarizationError(f"pyannote speaker diarization dependency is missing: {exc}") from exc

    pipeline: Any | None = None
    model: Any | None = None
    try:
        _allow_trusted_pyannote_checkpoint_globals(torch)
        # Both checkpoints are distributed by the pyannote project.
        with _trusted_pyannote_checkpoint_load_compat(torch, "bundled"):
            model = Model.from_pretrained(str(BUNDLED_PYANNOTE_MODEL_CHECKPOINT))
            pipeline = SpeakerDiarization(
                segmentation=model,
                embedding=DEFAULT_SPEAKER_EMBEDDING,
                embedding_batch_size=16,
                segmentation_batch_size=16,
                embedding_exclude_overlap=True,
            )
        if pipeline is None:
            raise SpeakerDiarizationError("Could not load pyannote speaker diarization pipeline")
        pipeline.instantiate(
            {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 12,
                    "threshold": 0.7045654963945799,
                },
                "segmentation": {"min_duration_off": 0.0},
            }
        )
        pipeline.to(torch.device(device))
        diarization = pipeline(str(wav_path), num_speakers=num_speakers)
        turns = sorted(
            (
                SpeakerTurn(float(segment.start), float(segment.end), str(speaker))
                for segment, _, speaker in diarization.itertracks(yield_label=True)
                if segment.end > segment.start
            ),
            key=lambda turn: (turn.start, turn.end, turn.speaker),
        )
        return SpeakerDiarizationResult(
            turns=turns,
            model="pyannote/segmentation-3.0 + SpeakerDiarization 3.1 recipe",
            embedding=DEFAULT_SPEAKER_EMBEDDING,
            speaker_count=len({turn.speaker for turn in turns}),
        )
    except SpeakerDiarizationError:
        raise
    except Exception as exc:
        raise SpeakerDiarizationError(f"pyannote speaker diarization failed: {exc}") from exc
    finally:
        del pipeline
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def speaker_for_chunk(chunk: SubtitleChunk, turns: list[SpeakerTurn]) -> str | None:
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        if turn.start >= chunk.end:
            break
        overlap = min(chunk.end, turn.end) - max(chunk.start, turn.start)
        if overlap > 0:
            overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0.0) + overlap
    if overlap_by_speaker:
        return max(overlap_by_speaker, key=overlap_by_speaker.get)
    # Forced alignment and diarization often disagree by a few hundred ms.
    nearest = min(
        turns,
        key=lambda turn: max(turn.start - chunk.end, chunk.start - turn.end, 0.0),
        default=None,
    )
    if nearest is not None and max(nearest.start - chunk.end, chunk.start - nearest.end, 0.0) <= 0.3:
        return nearest.speaker
    return None


def group_chunks_by_speaker(
    chunks: list[SubtitleChunk], turns: list[SpeakerTurn]
) -> list[SubtitleChunk]:
    """Apply normal subtitle grouping without merging across speaker changes."""
    if not chunks:
        return []
    grouped: list[SubtitleChunk] = []
    run: list[SubtitleChunk] = []
    previous_speaker: str | None = None
    for chunk in chunks:
        speaker = speaker_for_chunk(chunk, turns)
        if (
            run
            and speaker != previous_speaker
            and chunk.start - run[-1].end <= 0.5
            and not re.search(r"[。！？.!?]$", run[-1].text)
            and re.fullmatch(r"(?:だ|た|の|か|よ|ね|です|ます|でした|ません)[。！？.!?]*", chunk.text)
        ):
            speaker = previous_speaker
        if run and speaker != previous_speaker:
            grouped.extend(_group_speaker_run(run))
            run = []
        run.append(chunk)
        previous_speaker = speaker
    grouped.extend(_group_speaker_run(run))
    readable: list[SubtitleChunk] = []
    for index, cue in enumerate(grouped):
        next_start = grouped[index + 1].start if index + 1 < len(grouped) else cue.end + 0.8
        end = min(max(cue.end, cue.start + 0.65), next_start, cue.start + 6.0)
        readable.append(SubtitleChunk(cue.start, end, cue.text))
    for index in range(len(readable) - 1):
        cue = readable[index]
        following = readable[index + 1]
        if cue.end - cue.start >= 0.5:
            continue
        desired_end = cue.start + 0.5
        if following.end - desired_end >= 0.5:
            readable[index] = SubtitleChunk(cue.start, desired_end, cue.text)
            readable[index + 1] = SubtitleChunk(max(following.start, desired_end), following.end, following.text)
    return readable


def _group_speaker_run(chunks: list[SubtitleChunk]) -> list[SubtitleChunk]:
    grouped = group_chunks_by_timing(chunks, max_gap_s=0.9)
    merged: list[SubtitleChunk] = []
    for cue in grouped:
        if merged:
            previous = merged[-1]
            combined_text = _join_text(previous.text, cue.text)
            if (
                (previous.end - previous.start < 0.8 or cue.end - cue.start < 0.35)
                and cue.start - previous.end <= 0.9
                and cue.end - previous.start <= 4.0
                and len(combined_text) <= 42
            ):
                merged[-1] = SubtitleChunk(previous.start, cue.end, combined_text)
                continue
        merged.append(cue)
    return merged


def turns_to_json(turns: list[SpeakerTurn]) -> list[dict[str, float | str]]:
    return [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": turn.speaker}
        for turn in turns
    ]
