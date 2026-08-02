from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kotoba_standalone.media import SilenceSpan


warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\.pretrained' was deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat-32 \(TF32\) has been disabled.*",
)


DEFAULT_PYANNOTE_VAD_MODEL = "pyannote/segmentation-3.0"
PYANNOTE_MODEL_URL = "https://huggingface.co/pyannote/segmentation-3.0"
BUNDLED_PYANNOTE_MODEL_REVISION = "e66f3d3b9eb0873085418a7b813d3b369bf160bb"
BUNDLED_PYANNOTE_MODEL_DIR = Path(__file__).resolve().parent / "models" / "pyannote-segmentation-3.0"
BUNDLED_PYANNOTE_MODEL_CHECKPOINT = BUNDLED_PYANNOTE_MODEL_DIR / "pytorch_model.bin"
PYANNOTE_VAD_CHUNK_DURATION_S = 30 * 60.0
PYANNOTE_VAD_CHUNK_OVERLAP_S = 1.0
PyannoteProgressCallback = Callable[[int, int], None]


class PyannoteVadError(RuntimeError):
    pass


class PyannoteVadDependencyError(PyannoteVadError):
    pass


class PyannoteVadAccessError(PyannoteVadError):
    pass


@dataclass(frozen=True)
class PyannoteVadResult:
    speech_spans: list[SilenceSpan]
    model_name: str
    device: str
    pyannote_audio_version: str
    model_source: str = "huggingface"
    model_revision: str | None = None
    chunk_duration_s: float | None = None
    processed_audio_duration_s: float | None = None


def detect_speech_spans_pyannote(
    wav_path: Path,
    model_name: str = DEFAULT_PYANNOTE_VAD_MODEL,
    device: str = "cuda:0",
    min_speech_duration_s: float = 0.25,
    min_silence_duration_s: float = 0.5,
    progress: PyannoteProgressCallback | None = None,
) -> PyannoteVadResult:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)
            import torch
            import torchaudio
            import pyannote.audio
            from huggingface_hub import get_token
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
    except ImportError as exc:
        raise PyannoteVadDependencyError(
            "pyannote VAD dependencies are not installed. Run "
            "'uv sync --group transcribe --group cuda --group pyannote', then retry. "
            f"Import error: {exc}"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise PyannoteVadDependencyError(
            f"pyannote VAD requested {device}, but CUDA is not available."
        )

    model: Any | None = None
    pipeline: Any | None = None
    try:
        checkpoint, model_source = _resolve_model_checkpoint(model_name)
        token = get_token() if model_source == "huggingface" else None
        if token is None and model_source == "huggingface":
            raise PyannoteVadAccessError(_access_error_message(model_name))
        try:
            model = Model.from_pretrained(str(checkpoint), use_auth_token=token)
        except Exception as exc:
            raise _model_load_error(model_name, exc) from exc
        if model is None:
            raise PyannoteVadAccessError(_access_error_message(model_name))

        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate(
            {
                "min_duration_on": max(0.0, min_speech_duration_s),
                "min_duration_off": max(0.0, min_silence_duration_s),
            }
        )
        pipeline.to(torch.device(device))
        speech_spans, chunk_duration_s, processed_audio_duration_s = _run_vad_pipeline(
            pipeline,
            wav_path,
            torchaudio,
            progress,
        )
        return PyannoteVadResult(
            speech_spans=speech_spans,
            model_name=model_name,
            device=device,
            pyannote_audio_version=pyannote.audio.__version__,
            model_source=model_source,
            model_revision=(
                BUNDLED_PYANNOTE_MODEL_REVISION if model_source == "bundled" else None
            ),
            chunk_duration_s=chunk_duration_s,
            processed_audio_duration_s=processed_audio_duration_s,
        )
    finally:
        del pipeline
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_model_checkpoint(model_name: str) -> tuple[Path | str, str]:
    if model_name == DEFAULT_PYANNOTE_VAD_MODEL and BUNDLED_PYANNOTE_MODEL_CHECKPOINT.is_file():
        return BUNDLED_PYANNOTE_MODEL_CHECKPOINT, "bundled"

    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        checkpoint = local_path / "pytorch_model.bin"
        if not checkpoint.is_file():
            raise PyannoteVadError(
                f"Local pyannote model directory does not contain pytorch_model.bin: {local_path}"
            )
        return checkpoint, "local"
    if local_path.is_file():
        return local_path, "local"
    return model_name, "huggingface"


def _run_vad_pipeline(
    pipeline: Any,
    wav_path: Path,
    torchaudio: Any,
    progress: PyannoteProgressCallback | None,
) -> tuple[list[SilenceSpan], float | None, float | None]:
    info = torchaudio.info(str(wav_path))
    sample_rate = int(info.sample_rate)
    total_frames = int(info.num_frames)
    if sample_rate <= 0 or total_frames <= 0:
        output = pipeline(str(wav_path), hook=_progress_hook(progress))
        return _speech_spans_from_output(output), None, None

    chunk_ranges = _chunk_frame_ranges(
        total_frames,
        sample_rate,
        PYANNOTE_VAD_CHUNK_DURATION_S,
        PYANNOTE_VAD_CHUNK_OVERLAP_S,
    )
    if len(chunk_ranges) <= 1:
        output = pipeline(str(wav_path), hook=_progress_hook(progress))
        return _speech_spans_from_output(output), None, total_frames / sample_rate

    speech_spans: list[SilenceSpan] = []
    for index, (frame_offset, num_frames) in enumerate(chunk_ranges, start=1):
        waveform, chunk_sample_rate = torchaudio.load(
            str(wav_path),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        output = pipeline({"waveform": waveform, "sample_rate": chunk_sample_rate})
        chunk_start = frame_offset / chunk_sample_rate
        speech_spans.extend(
            SilenceSpan(float(segment.start) + chunk_start, float(segment.end) + chunk_start)
            for segment in output.get_timeline().support()
            if segment.end > segment.start
        )
        if progress is not None:
            progress(index, len(chunk_ranges))

    return (
        _merge_overlapping_spans(speech_spans),
        PYANNOTE_VAD_CHUNK_DURATION_S,
        total_frames / sample_rate,
    )


def _speech_spans_from_output(output: Any) -> list[SilenceSpan]:
    return [
        SilenceSpan(float(segment.start), float(segment.end))
        for segment in output.get_timeline().support()
        if segment.end > segment.start
    ]


def _chunk_frame_ranges(
    total_frames: int,
    sample_rate: int,
    chunk_duration_s: float,
    overlap_s: float,
) -> list[tuple[int, int]]:
    chunk_frames = max(1, int(chunk_duration_s * sample_rate))
    overlap_frames = max(0, int(overlap_s * sample_rate))
    step_frames = max(1, chunk_frames - overlap_frames)
    ranges: list[tuple[int, int]] = []
    offset = 0
    while offset < total_frames:
        frames = min(chunk_frames, total_frames - offset)
        ranges.append((offset, frames))
        if offset + frames >= total_frames:
            break
        offset += step_frames
    return ranges


def _merge_overlapping_spans(spans: list[SilenceSpan]) -> list[SilenceSpan]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [sorted_spans[0]]
    for span in sorted_spans[1:]:
        previous = merged[-1]
        if span.start <= previous.end + 0.05:
            merged[-1] = SilenceSpan(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return merged


def _progress_hook(progress: PyannoteProgressCallback | None) -> Callable[..., None] | None:
    if progress is None:
        return None

    def hook(
        step_name: str,
        step_artefact: Any,
        file: Any = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        del step_name, step_artefact, file
        if completed is not None and total:
            normalized_total = max(1, int(total))
            progress(min(max(0, int(completed)), normalized_total), normalized_total)

    return hook


def _model_load_error(model_name: str, exc: Exception) -> PyannoteVadError:
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in ("gated", "401", "403", "access", "token")):
        return PyannoteVadAccessError(_access_error_message(model_name))
    return PyannoteVadError(f"Could not load pyannote VAD model '{model_name}': {message}")


def _access_error_message(model_name: str) -> str:
    return (
        f"The pyannote VAD model '{model_name}' requires Hugging Face access. "
        f"Open {PYANNOTE_MODEL_URL}, accept the model conditions, then run "
        "'huggingface-cli login' or set HF_TOKEN before retrying."
    )
