from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kotoba_standalone.media import ffmpeg_exe
from kotoba_standalone.subtitle import SubtitleChunk, clean_text


class WhisperXAlignmentError(RuntimeError):
    pass


class WhisperXDependencyError(WhisperXAlignmentError):
    pass


@dataclass(frozen=True)
class WhisperXAlignmentResult:
    chunks: list[SubtitleChunk]
    metadata: dict[str, Any]


def align_chunks_with_whisperx(
    chunks: list[SubtitleChunk],
    wav_path: Path,
    *,
    language_code: str = "ja",
    device: str = "cuda:0",
    model_name: str | None = None,
    preserve_original_end: bool = True,
    min_subtitle_duration_s: float = 0.8,
) -> WhisperXAlignmentResult:
    if not chunks:
        return WhisperXAlignmentResult([], {"engine": "whisperx", "aligned_count": 0})
    try:
        import whisperx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WhisperXDependencyError(
            "WhisperX is not installed. Install the optional alignment dependency, then retry with --whisperx-align."
        ) from exc

    try:
        audio = _load_audio(wav_path, sample_rate=int(getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000)))
        if model_name:
            align_model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=device,
                model_name=model_name,
            )
        else:
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(
            [_chunk_to_segment(chunk) for chunk in chunks],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        raise WhisperXAlignmentError(f"WhisperX alignment failed: {exc}") from exc

    aligned_segments = result.get("segments", []) if isinstance(result, dict) else []
    aligned_chunks = [
        _aligned_segment_to_chunk(
            original,
            aligned,
            preserve_original_end=preserve_original_end,
            min_subtitle_duration_s=min_subtitle_duration_s,
        )
        for original, aligned in zip(chunks, aligned_segments, strict=False)
    ]
    if len(aligned_chunks) < len(chunks):
        aligned_chunks.extend(chunks[len(aligned_chunks) :])
    return WhisperXAlignmentResult(
        aligned_chunks,
        {
            "engine": "whisperx",
            "language_code": language_code,
            "device": device,
            "model": _metadata_model_name(metadata, model_name),
            "preserve_original_end": preserve_original_end,
            "min_subtitle_duration_s": min_subtitle_duration_s,
            "input_count": len(chunks),
            "aligned_count": len(aligned_segments),
            "changed_count": _changed_count(chunks, aligned_chunks),
            "segments": [
                {
                    "index": index,
                    "original_start": round(original.start, 3),
                    "original_end": round(original.end, 3),
                    "aligned_start": round(aligned.start, 3),
                    "aligned_end": round(aligned.end, 3),
                    "text": aligned.text,
                }
                for index, (original, aligned) in enumerate(zip(chunks, aligned_chunks, strict=True), 1)
            ],
        },
    )


def _chunk_to_segment(chunk: SubtitleChunk) -> dict[str, Any]:
    return {"start": chunk.start, "end": chunk.end, "text": chunk.text}


def _aligned_segment_to_chunk(
    original: SubtitleChunk,
    segment: Any,
    *,
    preserve_original_end: bool,
    min_subtitle_duration_s: float,
) -> SubtitleChunk:
    if not isinstance(segment, dict):
        return original
    start = _segment_time(segment, "start")
    end = _segment_time(segment, "end")
    text = clean_text(str(segment.get("text") or original.text))
    if start is None or end is None or end <= start:
        word_start, word_end = _word_boundaries(segment.get("words"))
        start = word_start if word_start is not None else original.start
        end = word_end if word_end is not None else original.end
    start = max(0.0, float(start))
    if preserve_original_end and original.end > start:
        end = max(float(end), original.end)
    end = max(start + min_subtitle_duration_s, float(end))
    return SubtitleChunk(start, end, text or original.text)


def _segment_time(segment: dict[str, Any], key: str) -> float | None:
    value = segment.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _word_boundaries(words: Any) -> tuple[float | None, float | None]:
    if not isinstance(words, list):
        return None, None
    starts: list[float] = []
    ends: list[float] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        if word.get("start") is not None:
            starts.append(float(word["start"]))
        if word.get("end") is not None:
            ends.append(float(word["end"]))
    return (min(starts) if starts else None, max(ends) if ends else None)


def _metadata_model_name(metadata: Any, fallback: str | None) -> str | None:
    if isinstance(metadata, dict):
        for key in ("model", "model_name", "name"):
            value = metadata.get(key)
            if value:
                return str(value)
    return fallback


def _changed_count(original: list[SubtitleChunk], aligned: list[SubtitleChunk]) -> int:
    count = 0
    for left, right in zip(original, aligned, strict=False):
        if abs(left.start - right.start) >= 0.02 or abs(left.end - right.end) >= 0.02:
            count += 1
    return count


def write_alignment_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_ffmpeg_on_path() -> None:
    ffmpeg_path = Path(ffmpeg_exe()).resolve()
    ffmpeg_dir = str(ffmpeg_path.parent)
    path = os.environ.get("PATH", "")
    if ffmpeg_dir.lower() not in [part.lower() for part in path.split(os.pathsep) if part]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + path


def _load_audio(path: Path, sample_rate: int) -> Any:
    try:
        import numpy as np
    except ImportError:
        _ensure_ffmpeg_on_path()
        import whisperx  # type: ignore[import-not-found]

        return whisperx.load_audio(str(path))
    ffmpeg = ffmpeg_exe()
    command = [
        ffmpeg,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    output = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(output, np.int16).flatten().astype(np.float32) / 32768.0
