from __future__ import annotations

from pathlib import Path

import pytest

from kotoba_standalone.pipeline import process_video, validate_silence_threshold
from kotoba_standalone.types import ProcessOptions, ProgressEvent


def test_validate_silence_threshold_accepts_db_values() -> None:
    assert validate_silence_threshold("-42dB") == "-42dB"


def test_validate_silence_threshold_rejects_missing_db_suffix() -> None:
    with pytest.raises(ValueError):
        validate_silence_threshold("-42")


def test_process_video_emits_progress_events(tmp_path: Path) -> None:
    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    events: list[ProgressEvent] = []

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out"), progress=events.append)

    assert result.status == "missing_transcription_dependency"
    assert result.output_dir == tmp_path / "out"
    assert result.wav_path == tmp_path / "out" / "ja_short_test.standalone.wav"
    assert (tmp_path / "out" / "ja_short_test.standalone.wav").exists()
    assert [event.stage for event in events] == ["prepare", "extract_audio", "probe_audio", "load_model", "dependency"]
