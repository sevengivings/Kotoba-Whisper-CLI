from __future__ import annotations

import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

from kotoba_standalone.qwen_mlx_transcriber import (
    Qwen3MlxTranscriber,
    qwen_mlx_chunks_to_raw_chunks,
    qwen_mlx_result_to_raw,
    qwen_mlx_result_to_raw_with_meta,
)
from kotoba_standalone.types import ProcessOptions


def test_qwen_mlx_chunks_to_raw_chunks_accepts_dict_items() -> None:
    assert qwen_mlx_chunks_to_raw_chunks([{"text": "こんにちは", "start": 1.0, "end": 1.5}]) == [
        {"timestamp": [1.0, 1.5], "text": "こんにちは"}
    ]


def test_qwen_mlx_chunks_to_raw_chunks_accepts_timestamp_items() -> None:
    assert qwen_mlx_chunks_to_raw_chunks([{"text": "はい", "timestamp": [2.0, 2.3]}]) == [
        {"timestamp": [2.0, 2.3], "text": "はい"}
    ]


def test_qwen_mlx_result_to_raw_falls_back_to_duration() -> None:
    result = SimpleNamespace(text="さあ", language="ja", chunks=None, segments=None)

    assert qwen_mlx_result_to_raw(result, fallback_duration_s=3.7) == {
        "text": "さあ",
        "language": "ja",
        "chunks": [{"timestamp": [0.0, 3.7], "text": "さあ"}],
    }


def test_qwen_mlx_transcriber_uses_session_api(tmp_path: Path, monkeypatch) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    calls: list[dict] = []

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__version__ = "test"
    fake_core = types.ModuleType("mlx.core")
    fake_core.float16 = object()
    fake_core.bfloat16 = object()
    fake_core.float32 = object()
    fake_mlx.core = fake_core

    class FakeSession:
        def __init__(self, model, dtype):
            calls.append({"model": model, "dtype": dtype})

        def transcribe(self, audio, **kwargs):
            calls.append({"audio": audio, "kwargs": kwargs})
            return SimpleNamespace(
                text="こんにちは",
                language="ja",
                chunks=[{"text": "こんにちは", "start": 0.0, "end": 0.5}],
                segments=None,
            )

    fake_qwen = types.ModuleType("mlx_qwen3_asr")
    fake_qwen.Session = FakeSession

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", fake_qwen)

    transcriber = Qwen3MlxTranscriber(
        ProcessOptions(asr_backend="qwen3-mlx", qwen_mlx_model_name="Qwen/Qwen3-ASR-1.7B")
    )
    transcriber.load()
    result = transcriber.transcribe(str(wav_path))

    assert calls[0]["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert calls[1]["kwargs"]["language"] == "ja"
    assert calls[1]["kwargs"]["return_chunks"] is True
    assert calls[1]["kwargs"]["return_timestamps"] is True
    assert calls[1]["kwargs"]["forced_aligner"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert result.raw == {"text": "こんにちは", "language": "ja", "chunks": [{"timestamp": [0.0, 0.5], "text": "こんにちは"}]}
    assert result.device_name == "mlx-gpu"
    assert result.word_timestamps_used is False

    transcriber.unload()
    assert transcriber._session is None
    assert transcriber._aligner is None
    assert transcriber._mlx is None


def test_qwen_mlx_result_to_raw_preserves_punctuation_on_aligned_segments() -> None:
    result = SimpleNamespace(
        text="お？どう。",
        language="ja",
        segments=[
            {"text": "お", "start": 0.0, "end": 0.3},
            {"text": "どう", "start": 0.4, "end": 0.8},
        ],
        chunks=None,
    )

    raw, word_timestamps_used = qwen_mlx_result_to_raw_with_meta(result, fallback_duration_s=1.0)

    assert word_timestamps_used is True
    assert raw["chunks"] == [
        {"timestamp": [0.0, 0.3], "text": "お？"},
        {"timestamp": [0.4, 0.8], "text": "どう。"},
    ]


def test_qwen_mlx_transcriber_reuses_forced_aligner_and_clears_holder_on_unload(
    tmp_path: Path, monkeypatch
) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    events: list[object] = []

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__version__ = "test"
    fake_core = types.ModuleType("mlx.core")
    fake_core.float16 = "f16"
    fake_core.bfloat16 = "bf16"
    fake_core.float32 = "f32"
    fake_mlx.core = fake_core

    class FakeSession:
        def __init__(self, model, dtype):
            del model, dtype

        def transcribe(self, audio, **kwargs):
            del audio
            events.append(kwargs.get("forced_aligner"))
            return SimpleNamespace(
                text="お？どう。",
                language="ja",
                segments=[
                    {"text": "お", "start": 0.0, "end": 0.3},
                    {"text": "どう", "start": 0.4, "end": 0.8},
                ],
                chunks=None,
            )

    class FakeAligner:
        def __init__(self, model_path, dtype):
            self.model_path = model_path
            self.dtype = dtype

    class FakeHolder:
        @classmethod
        def clear(cls) -> None:
            events.append("holder_cleared")

    fake_qwen = types.ModuleType("mlx_qwen3_asr")
    fake_qwen.Session = FakeSession
    fake_aligner_mod = types.ModuleType("mlx_qwen3_asr.forced_aligner")
    fake_aligner_mod.ForcedAligner = FakeAligner
    fake_load_models = types.ModuleType("mlx_qwen3_asr.load_models")
    fake_load_models._ModelHolder = FakeHolder

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", fake_qwen)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr.forced_aligner", fake_aligner_mod)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr.load_models", fake_load_models)

    transcriber = Qwen3MlxTranscriber(
        ProcessOptions(
            asr_backend="qwen3-mlx",
            qwen_mlx_model_name="Qwen/Qwen3-ASR-1.7B",
            qwen_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
            qwen_mlx_return_timestamps=True,
        )
    )
    transcriber.load()
    result = transcriber.transcribe(str(wav_path))
    transcriber.unload()

    assert isinstance(events[0], FakeAligner)
    assert events[0].model_path == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert result.word_timestamps_used is True
    assert result.raw["chunks"] == [
        {"timestamp": [0.0, 0.3], "text": "お？"},
        {"timestamp": [0.4, 0.8], "text": "どう。"},
    ]
    assert events[1] == "holder_cleared"
