from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from kotoba_standalone.translate.ollama import (
    OllamaModelError,
    OllamaUnavailableError,
    assert_ollama_model_available,
    build_system_prompt,
    check_ollama_available,
    default_output_srt,
    get_ollama_models,
    make_batches,
    parse_batch_translation,
    resolve_output_srt,
)
from kotoba_standalone.types import TranslationOptions


def test_default_output_srt_replaces_ja_suffix() -> None:
    assert default_output_srt(Path("sample.ja.srt")) == Path("sample.ko.srt")


def test_resolve_output_srt_accepts_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    assert resolve_output_srt(Path("sample.whisperx.ja.srt"), output_dir) == output_dir / "sample.whisperx.ko.srt"


def test_make_batches_defaults_to_subtitle_count() -> None:
    texts = [f"subtitle {index}" for index in range(120)]

    batches = make_batches(texts, batch_size=50, text_split_size=0)

    assert [len(batch) for batch in batches] == [50, 50, 20]
    assert batches[-1][-1] == 119


def test_parse_batch_translation_maps_local_numbers_to_original_indices() -> None:
    result = "[1] 안녕하세요\n[2] 괜찮아요"

    parsed = parse_batch_translation(result, [10, 11])

    assert parsed == {10: "안녕하세요", 11: "괜찮아요"}


def test_korean_prompt_can_request_strict_informal_style() -> None:
    prompt = build_system_prompt("japanese", "korean", batch_mode=True, korean_style="strict-banmal")

    assert "strict informal Korean speech style" in prompt
    assert "Translate each line separately" in prompt


def test_check_ollama_available_reports_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_urlopen(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)

    with pytest.raises(OllamaUnavailableError, match="Ollama is not reachable"):
        check_ollama_available(TranslationOptions(model="test"))


def test_get_ollama_models_parses_tags_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"models":[{"name":"a:model"},{"name":"b:model"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())

    assert get_ollama_models(TranslationOptions(model="a:model")) == ["a:model", "b:model"]


def test_assert_ollama_model_available_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kotoba_standalone.translate.ollama.get_ollama_models", lambda options: ["a:model"])

    with pytest.raises(OllamaModelError, match="was not found"):
        assert_ollama_model_available(TranslationOptions(model="missing:model"))
