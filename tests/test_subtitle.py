from app.subtitle import chunks_to_srt, format_srt_time, normalize_chunks


def test_format_srt_time() -> None:
    assert format_srt_time(1.2) == "00:00:01,200"
    assert format_srt_time(3661.234) == "01:01:01,234"
    assert format_srt_time(-1) == "00:00:00,000"


def test_srt_numbering_and_cleanup() -> None:
    chunks = normalize_chunks(
        [
            {"timestamp": [1.2, 4.5], "text": " 今日はよろしくお願いします 。 "},
            {"timestamp": [4.7, 7.1], "text": "こちらこそ、よろしくお願いします。"},
        ]
    )
    assert chunks_to_srt(chunks) == (
        "1\n"
        "00:00:01,200 --> 00:00:04,500\n"
        "今日はよろしくお願いします。\n\n"
        "2\n"
        "00:00:04,700 --> 00:00:07,100\n"
        "こちらこそ、よろしくお願いします。\n"
    )


def test_normalize_chunks_filters_and_repairs() -> None:
    chunks = normalize_chunks(
        [
            {"timestamp": [-1, 0], "text": ""},
            {"timestamp": [-1, 1], "text": "A"},
            {"timestamp": [0.5, 0.4], "text": "B"},
            {"timestamp": [2, 3], "text": "B"},
        ]
    )
    assert len(chunks) == 2
    assert chunks[0].start == 0
    assert chunks[0].end == 1
    assert chunks[1].start >= chunks[0].end
    assert chunks[1].end > chunks[1].start

