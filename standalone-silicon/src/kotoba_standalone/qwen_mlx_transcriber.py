from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from kotoba_standalone.media import wav_duration_seconds
from kotoba_standalone.mlx_transcriber import mlx_language_code
from kotoba_standalone.qwen_transcriber import attach_unaligned_text
from kotoba_standalone.transcriber import TranscriptionDependencyError, TranscriptionResult
from kotoba_standalone.types import ProcessOptions


class Qwen3MlxTranscriber:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self._session: Any | None = None
        self._aligner: Any | None = None
        self._mlx: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None

    def load(self) -> None:
        try:
            import mlx
            import mlx.core as mx
            from mlx_qwen3_asr import Session
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "MLX Qwen3-ASR dependencies are not installed. "
                "Run 'uv sync --group qwen-mlx', then retry."
            ) from exc

        dtype = _mlx_dtype(mx, self.options.model_dtype)
        self._session = Session(model=self.options.qwen_mlx_model_name, dtype=dtype)
        self._aligner = None
        if self.options.qwen_mlx_return_timestamps and self.options.qwen_aligner_model:
            try:
                from mlx_qwen3_asr.forced_aligner import ForcedAligner
            except ImportError:
                ForcedAligner = None
            if ForcedAligner is not None:
                self._aligner = ForcedAligner(model_path=self.options.qwen_aligner_model, dtype=dtype)
        self._mlx = mlx
        self.device_name = "mlx-gpu"
        self.torch_version = f"mlx {getattr(mlx, '__version__', 'unknown')}; mlx-qwen3-asr {_package_version('mlx-qwen3-asr')}"

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        if self._session is None:
            raise RuntimeError("Transcriber is not loaded")
        kwargs: dict[str, Any] = {
            "language": mlx_language_code(self.options.language),
            "return_chunks": True,
            "return_timestamps": self.options.qwen_mlx_return_timestamps,
            "verbose": False,
        }
        if self.options.qwen_mlx_return_timestamps:
            aligner = self._aligner or self.options.qwen_aligner_model
            if aligner:
                kwargs["forced_aligner"] = aligner
        result = self._session.transcribe(wav_path, **kwargs)
        raw, word_timestamps_used = qwen_mlx_result_to_raw_with_meta(
            result,
            fallback_duration_s=wav_duration_seconds(wav_path),
        )
        return TranscriptionResult(
            raw=raw,
            batch_size_used=1,
            device_name=self.device_name,
            torch_version=self.torch_version,
            torch_cuda_version=self.torch_cuda_version,
            word_timestamps_used=word_timestamps_used,
        )

    def unload(self) -> None:
        self._session = None
        self._aligner = None
        self._mlx = None
        load_models_mod = sys.modules.get("mlx_qwen3_asr.load_models")
        model_holder = getattr(load_models_mod, "_ModelHolder", None)
        clear_fn = getattr(model_holder, "clear", None)
        if callable(clear_fn):
            clear_fn()


def qwen_mlx_result_to_raw(result: Any, fallback_duration_s: float | None = None) -> dict[str, Any]:
    raw, _ = qwen_mlx_result_to_raw_with_meta(result, fallback_duration_s=fallback_duration_s)
    return raw


def qwen_mlx_result_to_raw_with_meta(
    result: Any,
    fallback_duration_s: float | None = None,
) -> tuple[dict[str, Any], bool]:
    text = _value(result, "text") or ""
    language = _value(result, "language")
    segments = qwen_mlx_chunks_to_raw_chunks(_value(result, "segments"))
    if segments:
        return (
            {
                "text": str(text),
                "language": language,
                "chunks": attach_unaligned_text(str(text), segments),
            },
            True,
        )
    chunks = qwen_mlx_chunks_to_raw_chunks(_value(result, "chunks"))
    if not chunks and text:
        chunks = [{"timestamp": [0.0, float(fallback_duration_s or 0.01)], "text": str(text).strip()}]
    return {"text": str(text), "language": language, "chunks": chunks}, False


def qwen_mlx_chunks_to_raw_chunks(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    chunks: list[dict[str, Any]] = []
    for item in items:
        parsed = _parse_chunk(item)
        if parsed is not None:
            chunks.append(parsed)
    return chunks


def _parse_chunk(item: Any) -> dict[str, Any] | None:
    text = str(_value(item, "text") or "").strip()
    if not text:
        return None
    start = _value(item, "start")
    end = _value(item, "end")
    if start is None or end is None:
        timestamp = _value(item, "timestamp")
        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
            start, end = timestamp[0], timestamp[1]
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        end_f = start_f + 0.01
    return {"timestamp": [start_f, end_f], "text": text}


def _mlx_dtype(mx: Any, dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized in {"bfloat16", "bf16"}:
        return mx.bfloat16
    if normalized in {"float32", "fp32"}:
        return mx.float32
    return mx.float16


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
