from __future__ import annotations

import importlib.util
from pathlib import Path


def load_translate_module():
    module_path = Path(__file__).parents[1] / "tools" / "translate-srt-ollama.py"
    spec = importlib.util.spec_from_file_location("translate_srt_ollama", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load translate-srt-ollama.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_batches_defaults_to_subtitle_count() -> None:
    module = load_translate_module()
    texts = [f"subtitle {index}" for index in range(120)]

    batches = module.make_batches(texts, batch_size=50, text_split_size=0)

    assert [len(batch) for batch in batches] == [50, 50, 20]
    assert batches[0][0] == 0
    assert batches[-1][-1] == 119


def test_batch_translate_is_enabled_by_default() -> None:
    module = load_translate_module()
    parser = module.build_parser()

    args = parser.parse_args(["input.ja.srt", "--model", "gemma3:4b"])

    assert args.batch_translate is True
    assert args.batch_size == 50
    assert args.korean_style == "polite"


def test_no_batch_translate_disables_batch_mode() -> None:
    module = load_translate_module()
    parser = module.build_parser()

    args = parser.parse_args(["input.ja.srt", "--model", "gemma3:4b", "--no-batch-translate"])

    assert args.batch_translate is False


def test_korean_prompt_requires_polite_speech_by_default() -> None:
    module = load_translate_module()

    prompt = module.build_system_prompt("japanese", "korean", batch_mode=True)

    assert "always use polite Korean speech style" in prompt
    assert "Do not use banmal" in prompt
    assert "Translate each line separately" in prompt


def test_korean_prompt_can_request_banmal() -> None:
    module = load_translate_module()

    prompt = module.build_system_prompt("japanese", "korean", batch_mode=True, korean_style="banmal")

    assert "use natural informal Korean speech style" in prompt
    assert "also known as banmal" in prompt
    assert "Avoid polite Korean endings" in prompt
    assert "always use polite Korean speech style" not in prompt


def test_korean_prompt_can_request_strict_banmal() -> None:
    module = load_translate_module()

    prompt = module.build_system_prompt("japanese", "korean", batch_mode=True, korean_style="strict-banmal")

    assert "use strict informal Korean speech style" in prompt
    assert "must be non-polite and casual" in prompt
    assert "Do not use Korean polite sentence endings" in prompt
    assert "감사합니다" in prompt
    assert "translate ありがとうございます as 고마워" in prompt
    assert "always use polite Korean speech style" not in prompt


def test_non_korean_prompt_does_not_force_korean_politeness() -> None:
    module = load_translate_module()

    prompt = module.build_system_prompt("japanese", "english", batch_mode=False)

    assert "always use polite Korean speech style" not in prompt


def test_make_batches_can_still_limit_by_text_size() -> None:
    module = load_translate_module()
    texts = ["aaaaa", "bbbbb", "ccccc"]

    batches = module.make_batches(texts, batch_size=50, text_split_size=11)

    assert batches == [[0], [1], [2]]
