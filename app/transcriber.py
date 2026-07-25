from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    raw: dict[str, Any]
    batch_size_used: int
    device_name: str
    torch_version: str
    torch_cuda_version: str | None


class KotobaTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None
        self._punctuator: Any | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. CPU fallback is disabled.")

        self._torch = torch
        device_index = _parse_cuda_device(self.config.model.device)
        self.device_name = torch.cuda.get_device_name(device_index)
        self.torch_version = torch.__version__
        self.torch_cuda_version = torch.version.cuda

        dtype = torch.float16 if self.config.model.dtype == "float16" else torch.float32
        model_kwargs: dict[str, Any] = {
            "attn_implementation": self.config.model.attention_implementation
        }

        LOGGER.info("GPU detected: %s", self.device_name)
        LOGGER.info("Loading model without remote diarization pipeline: %s", self.config.model.name)
        processor = AutoProcessor.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            torch_dtype=dtype,
            **model_kwargs,
        )
        model.to(self.config.model.device)
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=self.config.model.device,
            batch_size=self.config.inference.batch_size,
        )
        LOGGER.info("Model loaded: %s", self.config.model.name)

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Transcriber is not loaded")

        for batch_size in batch_size_sequence(
            self.config.inference.batch_size,
            self.config.inference.fallback_batch_sizes,
        ):
            try:
                LOGGER.info("Transcription started: batch_size=%s", batch_size)
                with self._torch.inference_mode():
                    raw = self._pipe(
                        wav_path,
                        chunk_length_s=self.config.inference.chunk_length_s,
                        batch_size=batch_size,
                        return_timestamps=self.config.inference.return_timestamps,
                        generate_kwargs={
                            "language": self.config.inference.language,
                            "task": "transcribe",
                        },
                    )
                raw = _json_safe(raw)
                if self.config.inference.punctuation:
                    raw = self._apply_punctuation(raw)
                return TranscriptionResult(
                    raw=raw,
                    batch_size_used=batch_size,
                    device_name=self.device_name,
                    torch_version=self.torch_version,
                    torch_cuda_version=self.torch_cuda_version,
                )
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    LOGGER.warning("CUDA OOM at batch_size=%s; retrying smaller batch if possible", batch_size)
                    self._torch.cuda.empty_cache()
                    continue
                raise
        raise RuntimeError("GPU memory exhausted for all configured batch sizes")

    def _apply_punctuation(self, raw: dict[str, Any]) -> dict[str, Any]:
        chunks = raw.get("chunks")
        if not isinstance(chunks, list):
            return raw
        if self._punctuator is None:
            from punctuators.models import PunctCapSegModelONNX

            self._punctuator = PunctCapSegModelONNX.from_pretrained(
                "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
            )
        texts = [str(chunk.get("text", "")) for chunk in chunks if isinstance(chunk, dict)]
        if not texts:
            return raw
        punctuated = self._punctuator.infer(texts)
        index = 0
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = "".join(punctuated[index])
                if "unk" not in text.lower():
                    chunk["text"] = text
                index += 1
        raw["text"] = "".join(str(chunk.get("text", "")) for chunk in chunks if isinstance(chunk, dict))
        return raw


def _parse_cuda_device(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError(f"Unsupported device: {device}")
    return int(device.split(":", 1)[1])


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def batch_size_sequence(start: int, configured_fallbacks: list[int]) -> list[int]:
    values = [start]
    current = start
    while current > 1:
        current = max(1, current // 2)
        values.append(current)
    for value in configured_fallbacks:
        if value >= 1:
            values.append(value)
    seen: set[int] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
