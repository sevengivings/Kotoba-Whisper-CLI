from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import ReferenceSegment, cer, reference_to_srt, write_reference
from tools.evaluate import evaluate, evaluate_intervals, vad_intervals


def test_cer_handles_japanese_text() -> None:
    assert cer("今日はカメラです", "今日はカメラです") == 0.0
    assert round(cer("今日はカメラです", "今日はカメラ"), 3) == 0.25


def test_evaluate_intervals_reports_vad_precision_and_recall() -> None:
    result = evaluate_intervals(
        [(10.0, 20.0), (30.0, 35.0)],
        [(9.0, 18.0), (30.0, 36.0), (40.0, 42.0)],
    )

    assert result["overlap_seconds"] == 13.0
    assert result["recall"] == 0.8667
    assert result["precision"] == 0.7647
    assert result["false_positive_seconds"] == 4.0
    assert result["missed_speech_seconds"] == 2.0


def test_evaluate_reads_reference_srt_and_vad_json(tmp_path: Path) -> None:
    segments = [
        ReferenceSegment(1.0, 2.0, "female_01", "えっ？", ["short"]),
        ReferenceSegment(4.0, 6.0, "male_01", "今日はテストです。", ["clean"]),
    ]
    reference_path = tmp_path / "sample.reference.json"
    srt_path = tmp_path / "sample.ja.srt"
    vad_path = tmp_path / "sample.vad.json"
    write_reference(reference_path, name="sample", duration=8.0, segments=segments)
    srt_path.write_text(reference_to_srt(segments), encoding="utf-8")
    vad_path.write_text(
        json.dumps({"transcription_spans": [{"start": 1.0, "end": 2.0}, {"start": 4.0, "end": 6.0}]}),
        encoding="utf-8",
    )

    result = evaluate(reference_path, srt_path=srt_path, vad_json_path=vad_path)

    assert result["srt"]["cer"] == 0.0
    assert result["vad"]["recall"] == 1.0
    assert result["vad"]["precision"] == 1.0


def test_vad_intervals_prefers_transcription_spans() -> None:
    payload = {
        "raw_speech_spans": [{"start": 0, "end": 10}],
        "transcription_spans": [{"start": 1, "end": 2}],
    }

    assert vad_intervals(payload) == [(1.0, 2.0)]
