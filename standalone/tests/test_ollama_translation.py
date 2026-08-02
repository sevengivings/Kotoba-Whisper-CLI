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
    clean_translated_subtitle_text,
    default_output_srt,
    format_ollama_model_choice,
    get_ollama_models,
    is_likely_translation_model,
    make_batches,
    parse_batch_translation,
    resolve_output_srt,
    source_based_translation_override,
    sort_ollama_models_for_translation,
    translation_model_label,
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


def test_sort_ollama_models_prioritizes_likely_translation_models() -> None:
    models = [
        "qwen3:14b",
        "hf.co/mradermacher/Hy-MT2-1.8B-GGUF:Q4_K_M",
        "hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M",
        "glm-4.7-flash:latest",
    ]

    sorted_models = sort_ollama_models_for_translation(models)

    assert sorted_models[0] == "hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M"
    assert sorted_models[1] == "hf.co/mradermacher/Hy-MT2-1.8B-GGUF:Q4_K_M"
    assert is_likely_translation_model("hf.co/mradermacher/translategemma-12b-it-GGUF:Q8_0")
    assert not is_likely_translation_model("hf.co/mradermacher/Hy-MT2-1.8B-GGUF:Q4_K_M")
    assert not is_likely_translation_model("qwen3:14b")
    assert translation_model_label("hf.co/mradermacher/Hy-MT2-1.8B-GGUF:Q4_K_M") == "실험용/비추천"
    assert format_ollama_model_choice("qwen3:14b").startswith("[번역 미확인]")


def test_parse_batch_translation_maps_local_numbers_to_original_indices() -> None:
    result = "[1] 안녕하세요\n[2] 괜찮아요"

    parsed = parse_batch_translation(result, [10, 11])

    assert parsed == {10: "안녕하세요", 11: "괜찮아요"}


def test_parse_batch_translation_removes_nested_number_labels() -> None:
    result = "1. [1] 안녕하세요\n2. [2] 괜찮아요"

    parsed = parse_batch_translation(result, [10, 11])

    assert parsed == {10: "안녕하세요", 11: "괜찮아요"}


def test_parse_batch_translation_splits_inline_numbered_lines() -> None:
    result = "[1] 안녕하세요 [2] 괜찮아요"

    parsed = parse_batch_translation(result, [10, 11])

    assert parsed == {10: "안녕하세요", 11: "괜찮아요"}


def test_clean_translated_subtitle_text_removes_leading_batch_labels() -> None:
    assert clean_translated_subtitle_text("[41] 이건 뭐예요?") == "이건 뭐예요?"


def test_source_based_translation_override_keeps_short_un_as_eung() -> None:
    assert source_based_translation_override("うん、") == "응"
    assert source_based_translation_override("うん。") == "응"
    assert source_based_translation_override(" うん ") == "응"
    assert source_based_translation_override("うん、うん。") == "응"
    assert source_based_translation_override("うん、うん、うん、うん、うん、うん、うん。") == "응"
    assert source_based_translation_override("うん、いいよ。") is None


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
