from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

from kotoba_standalone.mlx_transcriber import MlxKotobaTranscriber, ensure_mlx_weights_name, mlx_language_code
from kotoba_standalone.types import ProcessOptions


def test_ensure_mlx_weights_name_links_converted_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    ensure_mlx_weights_name(model_dir)

    assert (model_dir / "weights.safetensors").exists()


def test_mlx_language_code_maps_common_names() -> None:
    assert mlx_language_code("japanese") == "ja"
    assert mlx_language_code("English") == "en"
    assert mlx_language_code("ja") == "ja"


def test_mlx_transcriber_converts_segments_to_raw_chunks(tmp_path: Path, monkeypatch) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")

    calls: list[dict] = []

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__version__ = "test"
    fake_core = types.ModuleType("mlx.core")
    fake_core.cpu = object()
    fake_core.gpu = object()
    fake_core.set_default_device = lambda device: calls.append({"device": device})
    fake_mlx.core = fake_core

    fake_whisper = types.ModuleType("mlx_whisper")
    fake_whisper.__version__ = "test"

    def fake_transcribe(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"text": "さあ", "segments": [{"start": 0.0, "end": 0.5, "text": " さあ "}]}

    fake_whisper.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_whisper)

    transcriber = MlxKotobaTranscriber(
        ProcessOptions(asr_backend="kotoba-mlx", mlx_model_path=str(model_dir), mlx_device="gpu")
    )
    transcriber.load()
    result = transcriber.transcribe(str(wav_path))

    assert result.raw == {"text": "さあ", "chunks": [{"timestamp": [0.0, 0.5], "text": "さあ"}]}
    assert result.device_name == "mlx-gpu"
    assert result.batch_size_used == 1
    assert not result.word_timestamps_used
