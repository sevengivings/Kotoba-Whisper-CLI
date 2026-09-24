from __future__ import annotations

from typing import Any

from kotoba_standalone.media import wav_duration_seconds
from kotoba_standalone.transcriber import TranscriptionDependencyError, TranscriptionResult
from kotoba_standalone.types import ProcessOptions


QWEN_LANGUAGE_NAMES = {
    "japanese": "Japanese",
    "ja": "Japanese",
    "english": "English",
    "en": "English",
    "korean": "Korean",
    "ko": "Korean",
    "chinese": "Chinese",
    "zh": "Chinese",
}


class Qwen3Transcriber:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self._model: Any | None = None
        self._torch: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None

    def load(self) -> None:
        configure_qwen_logging()
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "Qwen3-ASR dependencies are not installed. "
                "Run 'uv sync --group cuda --group pyannote --group qwen', then retry."
            ) from exc

        self._torch = torch
        self.torch_version = torch.__version__
        self.torch_cuda_version = torch.version.cuda
        if self.options.model_device.startswith("cuda:") and torch.cuda.is_available():
            device_index = int(self.options.model_device.split(":", 1)[1])
            self.device_name = torch.cuda.get_device_name(device_index)
        else:
            self.device_name = self.options.model_device

        kwargs: dict[str, Any] = {
            "dtype": _torch_dtype(torch, self.options.model_dtype),
            "device_map": self.options.model_device,
            "max_inference_batch_size": self.options.batch_size,
            "max_new_tokens": 512,
        }
        if self.options.qwen_return_timestamps and self.options.qwen_aligner_model:
            kwargs["forced_aligner"] = self.options.qwen_aligner_model
            kwargs["forced_aligner_kwargs"] = {
                "dtype": _torch_dtype(torch, self.options.model_dtype),
                "device_map": self.options.model_device,
            }
        self._model = Qwen3ASRModel.from_pretrained(self.options.qwen_model_name, **kwargs)

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Transcriber is not loaded")
        result = self._model.transcribe(
            audio=load_wav_for_qwen(wav_path),
            language=qwen_language_name(self.options.language),
            return_time_stamps=self.options.qwen_return_timestamps,
        )
        first = result[0] if isinstance(result, list) else result
        time_stamps = _value(first, "time_stamps") or _value(first, "timestamps")
        raw = qwen_result_to_raw(first, fallback_duration_s=wav_duration_seconds(wav_path))
        return TranscriptionResult(
            raw=raw,
            batch_size_used=self.options.batch_size,
            device_name=self.device_name,
            torch_version=self.torch_version,
            torch_cuda_version=self.torch_cuda_version,
            word_timestamps_used=bool(qwen_timestamps_to_chunks(time_stamps)),
        )


def configure_qwen_logging() -> None:
    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        return
    transformers_logging.set_verbosity_error()


def qwen_language_name(language: str) -> str | None:
    normalized = language.strip().lower()
    return QWEN_LANGUAGE_NAMES.get(normalized, language or None)


def load_wav_for_qwen(wav_path: str) -> tuple[Any, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise TranscriptionDependencyError(
            "Qwen3-ASR audio dependencies are not installed. "
            "Run 'uv sync --group cuda --group pyannote --group qwen', then retry."
        ) from exc

    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def qwen_result_to_raw(result: Any, fallback_duration_s: float | None = None) -> dict[str, Any]:
    text = _value(result, "text") or ""
    language = _value(result, "language")
    time_stamps = _value(result, "time_stamps") or _value(result, "timestamps") or []
    chunks = qwen_timestamps_to_chunks(time_stamps)
    if chunks:
        chunks = attach_unaligned_text(str(text), chunks)
    if not chunks and text:
        chunks = [
            {
                "timestamp": [0.0, float(fallback_duration_s or 0.01)],
                "text": str(text),
            }
        ]
    return {"text": str(text), "language": language, "chunks": chunks}


def attach_unaligned_text(text: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep punctuation omitted by the Qwen forced aligner tokenizer."""
    positions: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        word = str(chunk["text"])
        found = text.find(word, cursor)
        if found < 0:
            return chunks
        positions.append((found, found + len(word)))
        cursor = found + len(word)
    result = [dict(chunk) for chunk in chunks]
    result[0]["text"] = text[: positions[0][0]].strip() + result[0]["text"]
    for index, (_, end) in enumerate(positions):
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        result[index]["text"] += text[end:next_start].strip()
    return result


def qwen_timestamps_to_chunks(time_stamps: Any) -> list[dict[str, Any]]:
    # qwen-asr returns ForcedAlignResult(items=[...]), not a bare list.
    time_stamps = _value(time_stamps, "items") or time_stamps
    if not isinstance(time_stamps, list):
        return []
    chunks: list[dict[str, Any]] = []
    for item in time_stamps:
        parsed = _parse_timestamp_item(item)
        if parsed is not None:
            chunks.append(parsed)
    return chunks


def _parse_timestamp_item(item: Any) -> dict[str, Any] | None:
    text = _value(item, "text") or _value(item, "word") or _value(item, "unit") or ""
    start = _value(item, "start_time")
    end = _value(item, "end_time")
    if start is None:
        start = _value(item, "start")
    if end is None:
        end = _value(item, "end")
    if (start is None or end is None) and isinstance(item, (list, tuple)) and len(item) >= 3:
        text, start, end = item[0], item[1], item[2]
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
    return {"timestamp": [start_f, end_f], "text": str(text)}


def _torch_dtype(torch: Any, dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
