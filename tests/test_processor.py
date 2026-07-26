from pathlib import Path

from app.media import is_supported_media
from app.processor import _offset_raw_chunks, extract_raw_chunks, unique_destination, validate_completion, write_json_atomic
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
