from __future__ import annotations

from types import SimpleNamespace

from kotoba_standalone.qwen_transcriber import qwen_language_name, qwen_result_to_raw, qwen_timestamps_to_chunks


def test_qwen_language_name_maps_common_aliases() -> None:
    assert qwen_language_name("japanese") == "Japanese"
    assert qwen_language_name("ja") == "Japanese"
    assert qwen_language_name("English") == "English"


def test_qwen_timestamps_to_chunks_accepts_dict_items() -> None:
    chunks = qwen_timestamps_to_chunks([{"text": "こんにちは", "start_time": 1.0, "end_time": 1.5}])

    assert chunks == [{"timestamp": [1.0, 1.5], "text": "こんにちは"}]


def test_qwen_timestamps_to_chunks_accepts_object_items() -> None:
    item = SimpleNamespace(text="はい", start_time=2.0, end_time=2.3)

    assert qwen_timestamps_to_chunks([item]) == [{"timestamp": [2.0, 2.3], "text": "はい"}]


def test_qwen_result_to_raw_accepts_forced_align_result_items() -> None:
    aligned = SimpleNamespace(
        items=[
            SimpleNamespace(text="hello", start_time=0.2, end_time=0.6),
            SimpleNamespace(text="there", start_time=0.7, end_time=1.1),
        ]
    )
    result = SimpleNamespace(text="hello there", language="English", time_stamps=aligned)

    assert qwen_result_to_raw(result, fallback_duration_s=5.0)["chunks"] == [
        {"timestamp": [0.2, 0.6], "text": "hello"},
        {"timestamp": [0.7, 1.1], "text": "there"},
    ]


def test_qwen_result_to_raw_keeps_punctuation_between_aligned_words() -> None:
    aligned = SimpleNamespace(items=[
        SimpleNamespace(text="お", start_time=0.0, end_time=0.3),
        SimpleNamespace(text="どう", start_time=0.4, end_time=0.8),
    ])
    result = SimpleNamespace(text="お？どう。", language="Japanese", time_stamps=aligned)

    assert [chunk["text"] for chunk in qwen_result_to_raw(result)["chunks"]] == ["お？", "どう。"]


def test_qwen_timestamps_to_chunks_accepts_tuple_items() -> None:
    assert qwen_timestamps_to_chunks([("ねえ", 3.0, 3.4)]) == [{"timestamp": [3.0, 3.4], "text": "ねえ"}]


def test_qwen_result_to_raw_falls_back_to_single_chunk_without_timestamps() -> None:
    result = SimpleNamespace(text="こんにちは", language="Japanese", time_stamps=[])

    assert qwen_result_to_raw(result, fallback_duration_s=4.2) == {
        "text": "こんにちは",
        "language": "Japanese",
        "chunks": [{"timestamp": [0.0, 4.2], "text": "こんにちは"}],
    }
