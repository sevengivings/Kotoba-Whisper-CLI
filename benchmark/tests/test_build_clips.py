from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_clips.py"
SPEC = importlib.util.spec_from_file_location("benchmark_build_clips", MODULE_PATH)
assert SPEC and SPEC.loader
build_clips = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_clips)


def test_select_source_clips_filters_source_and_clip() -> None:
    source = {
        "id": "source_a",
        "clips": [
            {"id": "R01", "start": "00:00:00", "end": "00:00:10"},
            {"id": "R02", "start": "00:00:10", "end": "00:00:20"},
        ],
    }

    selected = build_clips.select_source_clips(source, "source_a", "R02")

    assert [clip["id"] for clip in selected] == ["R02"]


def test_select_silence_clips_supports_synthetic_source_filter() -> None:
    selected = build_clips.select_silence_clips("synthetic", None)

    assert [clip["id"] for clip in selected] == ["S01_silence_30s", "S02_silence_60s"]


def test_silence_manifest_entry_marks_expected_speech_false(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio" / "S01_silence_30s.wav"
    entry = build_clips.silence_manifest_entry(
        {
            "id": "S01_silence_30s",
            "duration": 30.0,
            "tags": ["silence", "non_speech", "hallucination_test"],
        },
        wav_path,
        tmp_path,
        {"duration": 30.0},
    )

    assert entry["source_id"] == "synthetic"
    assert entry["expected_speech"] is False
    assert entry["path"] == "audio/S01_silence_30s.wav"
