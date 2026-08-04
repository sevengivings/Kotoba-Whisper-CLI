from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

from kotoba_standalone.transcriber import (
    TranscriptionDependencyError,
    TranscriptionResult,
    read_wav_mono_float32,
    read_wav_mono_float32_segment,
)
from kotoba_standalone.types import ProcessOptions


DEFAULT_FASTER_KOTOBA_MODEL = "RoachLin/kotoba-whisper-v2.2-faster"


class FasterKotobaTranscriber:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self._model: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None
        self.compute_type = "int8"

    def load(self) -> None:
        try:
            _install_pyav_stub()
            import ctranslate2
            import faster_whisper
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "faster-whisper dependencies are not installed. "
                "Run 'uv sync --group transcribe --group faster', then retry."
            ) from exc

        model_device = self.options.model_device.strip() or "cpu"
        if model_device != "cpu":
            raise TranscriptionDependencyError("Faster Kotoba CPU backend only supports the 'cpu' processing device.")

        self.device_name = "cpu"
        self.torch_version = f"faster-whisper {faster_whisper.__version__}; ctranslate2 {ctranslate2.__version__}"
        self.compute_type = "int8"
        self._model = WhisperModel(
            self.options.faster_model_name,
            device="cpu",
            compute_type=self.compute_type,
        )

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        audio_input = read_wav_mono_float32(wav_path)
        return self.transcribe_array(audio_input["array"])

    def transcribe_segment(self, wav_path: str, start_s: float, end_s: float) -> TranscriptionResult:
        audio_input = read_wav_mono_float32_segment(wav_path, start_s, end_s)
        return self.transcribe_array(audio_input["array"])

    def transcribe_array(self, audio: Any) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Transcriber is not loaded")

        segments, _info = self._model.transcribe(
            audio,
            language="ja" if self.options.language == "japanese" else self.options.language,
            task="transcribe",
            vad_filter=False,
            word_timestamps=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
        )
        chunks = [
            {
                "timestamp": [float(segment.start), float(segment.end)],
                "text": str(segment.text).strip(),
            }
            for segment in segments
            if str(segment.text).strip()
        ]
        return TranscriptionResult(
            raw={"chunks": chunks},
            batch_size_used=1,
            device_name=self.device_name,
            torch_version=self.torch_version,
            torch_cuda_version=self.torch_cuda_version,
            word_timestamps_used=False,
        )


def _install_pyav_stub() -> None:
    if "av" in sys.modules:
        return

    av_stub = types.ModuleType("av")
    av_stub.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
    av_stub.audio = types.SimpleNamespace(
        resampler=types.SimpleNamespace(AudioResampler=object),
        fifo=types.SimpleNamespace(AudioFifo=object),
    )
    av_stub.error = types.SimpleNamespace(InvalidDataError=Exception)

    def disabled_open(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("PyAV decoding is disabled; Kotoba passes WAV audio directly.")

    av_stub.open = disabled_open
    sys.modules["av"] = av_stub
