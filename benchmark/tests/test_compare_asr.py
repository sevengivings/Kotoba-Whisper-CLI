from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "compare_asr.py"
SPEC = importlib.util.spec_from_file_location("benchmark_compare_asr", MODULE_PATH)
assert SPEC and SPEC.loader
compare_asr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_asr
SPEC.loader.exec_module(compare_asr)


def test_select_clips_filters_requested_ids() -> None:
    clips = [
        compare_asr.Clip("R01", 1.0, True, ["speech"]),
        compare_asr.Clip("N01", 1.0, False, ["non_speech"]),
    ]

    selected = compare_asr.select_clips(clips, ["N01"])

    assert [clip.id for clip in selected] == ["N01"]


def test_read_result_finds_nested_clip_outputs(tmp_path: Path) -> None:
    root = tmp_path / "kotoba-mlx"
    nested = root / "R01"
    nested.mkdir(parents=True)
    (nested / "R01.ja.txt").write_text("こんにちは。\n", encoding="utf-8")
    (nested / "R01.process.json").write_text(
        json.dumps({"status": "success", "processing_seconds": 2.5, "subtitle_count": 1}),
        encoding="utf-8",
    )

    result = compare_asr.read_result(
        compare_asr.WorkDir("kotoba", root),
        compare_asr.Clip("R01", 5.0, True, []),
    )

    assert result["status"] == "success"
    assert result["real_time_factor"] == 0.5
    assert result["text_chars"] == 6


def test_compare_clip_marks_non_speech_hallucination(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "N01.ja.txt").write_text("何か聞こえます。", encoding="utf-8")
    (right / "N01.ja.txt").write_text("", encoding="utf-8")

    row = compare_asr.compare_clip(
        compare_asr.Clip("N01", 10.0, False, ["non_speech"]),
        compare_asr.WorkDir("left", left),
        compare_asr.WorkDir("right", right),
    )

    assert row["hallucination_candidate"] == {"left": True, "right": False}
    assert row["text_equal"] is False


def test_summary_markdown_includes_labels_and_preview() -> None:
    markdown = compare_asr.summary_markdown(
        {
            "left": {"label": "kotoba", "path": "/tmp/kotoba"},
            "right": {"label": "qwen", "path": "/tmp/qwen"},
            "results": [
                {
                    "clip_id": "R01",
                    "duration": 5.0,
                    "kotoba": {"status": "success", "processing_seconds": 1.0, "text_chars": 4, "text_preview": "こんにちは"},
                    "qwen": {"status": "success", "processing_seconds": 2.0, "text_chars": 5, "text_preview": "こんばんは"},
                    "text_similarity": 0.5,
                    "hallucination_candidate": {"kotoba": False, "qwen": False},
                }
            ],
        }
    )

    assert "| Clip | Duration | kotoba status | qwen status |" in markdown
    assert "- kotoba: こんにちは" in markdown
