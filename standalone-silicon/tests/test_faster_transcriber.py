from __future__ import annotations

import sys
import types
import wave
from types import SimpleNamespace

from kotoba_standalone.faster_transcriber import FasterKotobaTranscriber, _install_pyav_stub
from kotoba_standalone.types import ProcessOptions


def test_install_pyav_stub_allows_module_spec(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "av", raising=False)

    _install_pyav_stub()

    assert sys.modules["av"].__spec__ is not None


def test_faster_transcriber_converts_segments_to_raw_chunks(tmp_path, monkeypatch) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type

        def transcribe(self, audio, **kwargs):
            del audio, kwargs
            return (
                [
                    SimpleNamespace(start=0.0, end=0.5, text=" さあ "),
                    SimpleNamespace(start=0.5, end=1.0, text=""),
                ],
                SimpleNamespace(language="ja"),
            )

    fake_faster = types.ModuleType("faster_whisper")
    fake_faster.__version__ = "test"
    fake_faster.WhisperModel = FakeWhisperModel
    fake_ctranslate2 = types.ModuleType("ctranslate2")
    fake_ctranslate2.__version__ = "test"
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)

    transcriber = FasterKotobaTranscriber(ProcessOptions(asr_backend="faster-kotoba", model_device="cpu"))
    transcriber.load()
    result = transcriber.transcribe(str(wav_path))

    assert result.raw == {"chunks": [{"timestamp": [0.0, 0.5], "text": "さあ"}]}
    assert result.batch_size_used == 1
    assert result.device_name == "cpu"
    assert not result.word_timestamps_used
