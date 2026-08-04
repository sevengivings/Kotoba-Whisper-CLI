from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from kotoba_standalone.transcriber import KotobaTranscriber, TranscriptionDependencyError
from kotoba_standalone.types import ProcessOptions


class FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(return_attention_mask=False)
        self.tokenizer = object()


class FakeModel:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(forced_decoder_ids=["test"], return_legacy_cache=False)
        self.device = None

    def to(self, device: str) -> None:
        self.device = device


def install_fake_transcriber_dependencies(monkeypatch: pytest.MonkeyPatch, cuda_available: bool) -> dict:
    calls: dict = {}

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.__version__ = "test-torch"
    torch.version = SimpleNamespace(cuda=None)
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda index: f"Fake CUDA {index}",
        empty_cache=lambda: None,
    )

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeProcessor:
            calls["processor_model_name"] = model_name
            return FakeProcessor()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> FakeModel:
            calls["model_name"] = model_name
            calls["model_kwargs"] = kwargs
            model = FakeModel()
            calls["model"] = model
            return model

    def fake_pipeline(**kwargs: object) -> object:
        calls["pipeline_kwargs"] = kwargs
        return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForSpeechSeq2Seq = FakeAutoModel
    transformers.AutoProcessor = FakeAutoProcessor
    transformers.pipeline = fake_pipeline

    transformers_utils = types.ModuleType("transformers.utils")
    transformers_logging = SimpleNamespace(set_verbosity_error=lambda: None)
    transformers_utils.logging = transformers_logging

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", transformers_logging)
    return calls


def test_kotoba_transcriber_allows_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_transcriber_dependencies(monkeypatch, cuda_available=False)

    transcriber = KotobaTranscriber(ProcessOptions(model_device="cpu", model_dtype="float16"))
    transcriber.load()

    assert transcriber.device_name == "cpu"
    assert calls["model_kwargs"]["torch_dtype"] == "float32"
    assert calls["model"].device == "cpu"
    assert calls["pipeline_kwargs"]["device"] == "cpu"
    assert calls["pipeline_kwargs"]["torch_dtype"] == "float32"


def test_kotoba_transcriber_explains_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_transcriber_dependencies(monkeypatch, cuda_available=False)

    transcriber = KotobaTranscriber(ProcessOptions(model_device="cuda:0"))

    with pytest.raises(TranscriptionDependencyError, match="Choose 'cpu'"):
        transcriber.load()
