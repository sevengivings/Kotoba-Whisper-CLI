from __future__ import annotations

from pathlib import Path

from kotoba_standalone.translate.ollama import build_system_prompt, default_output_srt, make_batches, parse_batch_translation


def test_default_output_srt_replaces_ja_suffix() -> None:
    assert default_output_srt(Path("sample.ja.srt")) == Path("sample.ko.srt")


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

