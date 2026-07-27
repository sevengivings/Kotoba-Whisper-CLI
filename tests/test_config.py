from pathlib import Path

from app.config import load_config


def test_config_loading() -> None:
    config = load_config(Path("config/config.yaml"))
    assert config.model.name == "kotoba-tech/kotoba-whisper-v2.2"
    assert config.inference.batch_size == 8
    assert config.inference.word_timestamps is True
    assert config.inference.word_max_gap_s == 0.5
    assert config.inference.vad_min_speech_duration_s == 0.25
    assert config.inference.vad_padding_s == 0.4
    assert config.inference.subtitle_merge_gap_s == 0.5
    assert config.inference.fallback_subtitle_max_duration_s == 5.0
    assert config.inference.fallback_subtitle_chars_per_second == 5.0
    assert config.inference.fallback_subtitle_padding_s == 0.4
    assert config.inference.filter_short_repeated_phrases is True
    assert "\u3054\u3081\u3093\u3002" in config.inference.filtered_short_phrases
    assert "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\u3002" in config.inference.filtered_short_phrases
    assert config.inference.filtered_short_phrase_max_duration_s == 1.6
    assert config.paths.output.as_posix() == "/workspace/output"
