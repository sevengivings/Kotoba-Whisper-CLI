from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from kotoba_standalone.types import ProcessOptions


warnings.filterwarnings(
    "ignore",
    message=r"The input name `inputs` is deprecated\. Please make sure to use `input_features` instead\.",
    category=FutureWarning,
    module=r"transformers\.models\.whisper\.generation_whisper",
)


class TranscriptionDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    raw: dict[str, Any]
    batch_size_used: int
    device_name: str
    torch_version: str
    torch_cuda_version: str | None
    word_timestamps_used: bool


class KotobaTranscriber:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None
        self._word_timestamps_available = True

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "Standalone transcription dependencies are not installed. "
                "Run 'uv sync --group transcribe --group cuda', then retry."
            ) from exc

        if not torch.cuda.is_available():
            raise TranscriptionDependencyError("CUDA is not available. Standalone transcription currently requires NVIDIA CUDA.")

        self._torch = torch
        device_index = _parse_cuda_device(self.options.model_device)
        self.device_name = torch.cuda.get_device_name(device_index)
        self.torch_version = torch.__version__
        self.torch_cuda_version = torch.version.cuda

        dtype = torch.float16 if self.options.model_dtype == "float16" else torch.float32
        processor = AutoProcessor.from_pretrained(self.options.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.options.model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        model.to(self.options.model_device)
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=self.options.model_device,
            batch_size=self.options.batch_size,
        )

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Transcriber is not loaded")
        for batch_size in batch_size_sequence(self.options.batch_size):
            try:
                timestamp_mode: bool | str = "word" if self._word_timestamps_available else True
                word_timestamps_used = self._word_timestamps_available
                try:
                    raw = self._transcribe_with_timestamp_mode(wav_path, batch_size, timestamp_mode)
                except IndexError:
                    if not self._word_timestamps_available:
                        raise
                    self._word_timestamps_available = False
                    word_timestamps_used = False
                    raw = self._transcribe_with_timestamp_mode(wav_path, batch_size, True)
                return TranscriptionResult(
                    raw=_json_safe(raw),
                    batch_size_used=batch_size,
                    device_name=self.device_name,
                    torch_version=self.torch_version,
                    torch_cuda_version=self.torch_cuda_version,
                    word_timestamps_used=word_timestamps_used,
                )
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    self._torch.cuda.empty_cache()
                    continue
                raise
        raise RuntimeError("GPU memory exhausted for all fallback batch sizes")

    def _transcribe_with_timestamp_mode(self, wav_path: str, batch_size: int, timestamp_mode: bool | str) -> Any:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Transcriber is not loaded")
        with self._torch.inference_mode():
            return self._pipe(
                wav_path,
                chunk_length_s=self.options.chunk_length_s,
                batch_size=batch_size,
                return_timestamps=timestamp_mode,
                generate_kwargs={"language": self.options.language, "task": "transcribe"},
            )


def extract_raw_chunks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("chunks"), list):
        return raw["chunks"]
    collected: list[dict[str, Any]] = []
    for key, value in raw.items():
        if str(key).startswith("chunks") and isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))
    return collected


def batch_size_sequence(start: int) -> list[int]:
    values: list[int] = []
    current = max(1, start)
    while current >= 1:
        values.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    return values


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _parse_cuda_device(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError(f"Unsupported device: {device}")
    return int(device.split(":", 1)[1])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
