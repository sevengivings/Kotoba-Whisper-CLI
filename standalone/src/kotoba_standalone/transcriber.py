from __future__ import annotations

import warnings
import wave
from array import array
from dataclasses import dataclass
from typing import Any

from kotoba_standalone.types import ProcessOptions


warnings.filterwarnings(
    "ignore",
    message=r"The input name `inputs` is deprecated\. Please make sure to use `input_features` instead\.",
    category=FutureWarning,
    module=r"transformers\.models\.whisper\.generation_whisper",
)
warnings.filterwarnings(
    "ignore",
    message=r"From v4\.47 onwards, when a model cache is to be returned.*",
    category=FutureWarning,
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
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "Standalone transcription dependencies are not installed. "
                "Run 'uv sync --group transcribe --group cuda', then retry."
            ) from exc

        transformers_logging.set_verbosity_error()
        if not torch.cuda.is_available():
            raise TranscriptionDependencyError("CUDA is not available. Standalone transcription currently requires NVIDIA CUDA.")

        self._torch = torch
        device_index = _parse_cuda_device(self.options.model_device)
        self.device_name = torch.cuda.get_device_name(device_index)
        self.torch_version = torch.__version__
        self.torch_cuda_version = torch.version.cuda

        dtype = torch.float16 if self.options.model_dtype == "float16" else torch.float32
        processor = AutoProcessor.from_pretrained(self.options.model_name)
        processor.feature_extractor.return_attention_mask = True
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.options.model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        model.generation_config.forced_decoder_ids = None
        model.generation_config.return_legacy_cache = True
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

    def transcribe_raw(self, wav_path: str) -> dict[str, Any]:
        return self.transcribe(wav_path).raw

    def _transcribe_with_timestamp_mode(self, wav_path: str, batch_size: int, timestamp_mode: bool | str) -> Any:
        if self._pipe is None or self._torch is None:
            raise RuntimeError("Transcriber is not loaded")
        audio_input = read_wav_mono_float32(wav_path)
        with self._torch.inference_mode():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                return self._pipe(
                    audio_input,
                    chunk_length_s=self.options.chunk_length_s,
                    batch_size=batch_size,
                    return_timestamps=timestamp_mode,
                    generate_kwargs={
                        "language": self.options.language,
                        "task": "transcribe",
                        "return_legacy_cache": True,
                    },
                )


def read_wav_mono_float32(wav_path: str) -> dict[str, Any]:
    import numpy as np

    with wave.open(wav_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        if sample_width != 2:
            raise RuntimeError(f"Expected 16-bit PCM WAV: {wav_path}")
        raw = wav_file.readframes(wav_file.getnframes())
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array("h", samples[::channels])
    audio = np.asarray(samples, dtype=np.float32) / 32768.0
    return {"array": audio, "sampling_rate": sample_rate}


def extract_raw_chunks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("chunks"), list):
        return raw["chunks"]
    collected: list[dict[str, Any]] = []
    for key, value in raw.items():
        if str(key).startswith("chunks") and isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))
    return collected


def offset_raw_chunks(raw_chunks: list[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    offset_chunks: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if timestamp is None:
            continue
        try:
            start, end = _offset_timestamp(timestamp, offset_s)
        except (TypeError, ValueError):
            continue
        copied = dict(chunk)
        copied["timestamp"] = [start, end]
        copied.pop("timestamps", None)
        offset_chunks.append(copied)
    return offset_chunks


def _offset_timestamp(timestamp: Any, offset_s: float) -> tuple[float, float]:
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = 0.0 if timestamp[0] is None else float(timestamp[0])
        end = start if timestamp[1] is None else float(timestamp[1])
        return start + offset_s, end + offset_s
    if isinstance(timestamp, dict):
        start = 0.0 if timestamp.get("start") is None else float(timestamp["start"])
        end = start if timestamp.get("end") is None else float(timestamp["end"])
        return start + offset_s, end + offset_s
    raise ValueError(f"Unsupported timestamp format: {timestamp!r}")


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
