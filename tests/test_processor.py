from pathlib import Path
import wave
from array import array

from app.media import estimate_silence_threshold, is_supported_media
from app.processor import (
    _offset_raw_chunks,
    extract_raw_chunks,
    load_job_options,
    source_disposition_for,
    unique_destination,
    validate_completion,
    write_json_atomic,
)
from app.transcriber import batch_size_sequence


class _Validation:
    enabled = True
    minimum_coverage_ratio = 0.70
    maximum_uncovered_tail_seconds = 300
    suspicious_result_destination = "failed"


class _Config:
    validation = _Validation()


def test_supported_extensions_case_insensitive() -> None:
    assert is_supported_media(Path("sample.MP4"))
    assert is_supported_media(Path("sample.wav"))
    assert not is_supported_media(Path("sample.txt"))


def test_unique_destination_avoids_overwrite(tmp_path: Path) -> None:
    original = tmp_path / "sample.mp4"
    original.write_text("x", encoding="utf-8")
    candidate = unique_destination(original)
    assert candidate != original
    assert candidate.name.startswith("sample_")
    assert candidate.suffix == ".mp4"


def test_process_json_atomic_write(tmp_path: Path) -> None:
    output = tmp_path / "sample.process.json"
    write_json_atomic(output, {"status": "success"})
    assert output.exists()
    assert not output.with_suffix(".json.part").exists()


def test_completion_validation() -> None:
    assert validate_completion(1000, 800, _Config()) == "success"
    assert validate_completion(1000, 600, _Config()) == "suspicious_incomplete"


def test_source_disposition_for_copied_success() -> None:
    assert source_disposition_for("success", True, _Config()) == "deleted"
    assert source_disposition_for("success", False, _Config()) == "archive"
    assert source_disposition_for("suspicious_incomplete", True, _Config()) == "failed"


def test_oom_batch_sequence() -> None:
    assert batch_size_sequence(8, [4, 2, 1]) == [8, 4, 2, 1]
    assert batch_size_sequence(6, [4, 2, 1]) == [6, 3, 1, 4, 2]


def test_extract_raw_chunks_from_speaker_keys() -> None:
    raw = {
        "text": "all",
        "chunks/SPEAKER_00": [{"timestamp": [0, 1], "text": "A"}],
        "chunks/SPEAKER_01": [{"timestamp": [1, 2], "text": "B"}],
    }
    assert len(extract_raw_chunks(raw)) == 2


def test_offset_raw_chunks() -> None:
    chunks = _offset_raw_chunks(
        [
            {"timestamp": [0, 1], "text": "A"},
            {"timestamps": {"start": 2, "end": 3}, "text": "B"},
        ],
        10,
    )

    assert chunks == [
        {"timestamp": [10.0, 11.0], "text": "A"},
        {"timestamp": [12.0, 13.0], "text": "B"},
    ]


def test_offset_raw_chunks_clamps_early_start() -> None:
    chunks = _offset_raw_chunks(
        [
            {"timestamp": [0, 1], "text": "A"},
            {"timestamp": [2, 3], "text": "B"},
        ],
        10,
        minimum_start_s=10.5,
    )

    assert chunks == [
        {"timestamp": [10.5, 11.0], "text": "A"},
        {"timestamp": [12.0, 13.0], "text": "B"},
    ]


def test_load_job_options_accepts_silence_overrides(tmp_path: Path) -> None:
    options_path = tmp_path / "sample.mp4.options.json"
    options_path.write_text(
        (
            '{"silence_threshold_db": "-35dB", '
            '"min_silence_duration_s": 0.4, '
            '"auto_silence_threshold": true, '
            '"delete_source_on_success": true}'
        ),
        encoding="utf-8",
    )

    assert load_job_options(options_path) == {
        "silence_threshold_db": "-35dB",
        "min_silence_duration_s": 0.4,
        "auto_silence_threshold": True,
        "delete_source_on_success": True,
    }


def test_load_job_options_accepts_utf8_bom(tmp_path: Path) -> None:
    options_path = tmp_path / "sample.mp4.options.json"
    options_path.write_text('{"silence_threshold_db": "-35dB"}', encoding="utf-8-sig")

    assert load_job_options(options_path) == {"silence_threshold_db": "-35dB"}


def test_load_job_options_rejects_invalid_threshold(tmp_path: Path) -> None:
    options_path = tmp_path / "sample.mp4.options.json"
    options_path.write_text('{"silence_threshold_db": "-35"}', encoding="utf-8")

    try:
        load_job_options(options_path)
    except ValueError as exc:
        assert "silence_threshold_db" in str(exc)
    else:
        raise AssertionError("Expected invalid silence threshold to fail")


def test_estimate_silence_threshold_from_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    samples = array("h")
    samples.extend([260] * 16000)
    samples.extend([2600] * 16000)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    estimate = estimate_silence_threshold(wav_path)

    assert estimate.threshold_db == "-35dB"
    assert round(estimate.noise_floor_db) == -42
    assert round(estimate.speech_level_db) == -22
    assert estimate.analyzed_frame_count > 0
