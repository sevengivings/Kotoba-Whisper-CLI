from dataclasses import dataclass

from app.subtitle import (
    SubtitleChunk,
    chunks_to_srt,
    format_srt_time,
    group_chunks_by_timing,
    filter_short_repeated_phrases,
    normalize_chunks,
    split_chunks_on_silence,
    split_text_for_spans,
)


@dataclass(frozen=True)
class Silence:
    start: float
    end: float


def test_format_srt_time() -> None:
    assert format_srt_time(1.2) == "00:00:01,200"
    assert format_srt_time(3661.234) == "01:01:01,234"
    assert format_srt_time(-1) == "00:00:00,000"


def test_srt_numbering_and_cleanup() -> None:
    chunks = normalize_chunks(
        [
            {"timestamp": [1.2, 4.5], "text": " Hello , world ! "},
            {"timestamp": [4.7, 7.1], "text": "Again."},
        ]
    )
    assert chunks_to_srt(chunks) == (
        "1\n"
        "00:00:01,200 --> 00:00:04,500\n"
        "Hello, world!\n\n"
        "2\n"
        "00:00:04,700 --> 00:00:07,100\n"
        "Again.\n"
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


def test_group_chunks_by_timing_uses_word_timestamps() -> None:
    words = normalize_chunks(
        [
            {"timestamp": [0.0, 0.3], "text": "Hello"},
            {"timestamp": [0.3, 0.6], "text": " world."},
            {"timestamp": [1.4, 1.7], "text": "Again"},
        ]
    )

    grouped = group_chunks_by_timing(words, max_gap_s=0.5, max_duration_s=4.0, max_chars=80)

    assert [(chunk.start, chunk.end, chunk.text) for chunk in grouped] == [
        (0.0, 0.6, "Hello world."),
        (1.4, 1.7, "Again"),
    ]


def test_filter_short_repeated_phrases_removes_short_standalone_phrase() -> None:
    chunks = [
        SubtitleChunk(0.0, 1.0, "\u3054\u3081\u3093\u3002"),
        SubtitleChunk(2.0, 5.0, "\u3054\u3081\u3093\u3002"),
        SubtitleChunk(6.0, 7.0, "\u3054\u3081\u3093\u3001\u5f85\u3063\u3066\u3002"),
        SubtitleChunk(8.0, 9.0, "\u306f\u3044\u3002"),
    ]

    filtered = filter_short_repeated_phrases(
        chunks,
        ["\u3054\u3081\u3093\u3002"],
        max_duration_s=1.6,
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in filtered] == [
        (2.0, 5.0, "\u3054\u3081\u3093\u3002"),
        (6.0, 7.0, "\u3054\u3081\u3093\u3001\u5f85\u3063\u3066\u3002"),
        (8.0, 9.0, "\u306f\u3044\u3002"),
    ]


def test_group_chunks_by_timing_merges_unfinished_fragments() -> None:
    fragments = normalize_chunks(
        [
            {"timestamp": [0.0, 1.0], "text": "もし"},
            {"timestamp": [2.2, 3.0], "text": "も"},
            {"timestamp": [4.0, 4.8], "text": "し。"},
            {"timestamp": [4.8, 6.0], "text": "次。"},
        ]
    )

    grouped = group_chunks_by_timing(fragments, max_gap_s=2.0, max_duration_s=8.0, max_chars=80)

    assert [(chunk.start, chunk.end, chunk.text) for chunk in grouped] == [
        (0.0, 4.8, "もしもし。"),
        (4.8, 6.0, "次。"),
    ]


def test_split_chunks_on_silence_splits_long_chunk() -> None:
    chunks = normalize_chunks(
        [
            {
                "timestamp": [0, 12],
                "text": "First sentence? Second sentence. Third sentence.",
            }
        ]
    )

    split = split_chunks_on_silence(
        chunks,
        [Silence(3.0, 4.0), Silence(8.0, 9.0)],
        min_subtitle_duration_s=0.8,
    )

    assert [(chunk.start, chunk.end) for chunk in split] == [(0, 3.0), (4.0, 8.0), (9.0, 12)]
    assert [chunk.text for chunk in split] == [
        "First sentence?",
        "Second sentence.",
        "Third sentence.",
    ]


def test_split_text_for_spans_falls_back_to_character_count() -> None:
    assert split_text_for_spans("abcdefgh", [1, 1]) == ["abcd", "efgh"]


def test_split_text_for_spans_handles_more_spans_than_sentences() -> None:
    assert split_text_for_spans("abcdef", [1, 1, 1]) == ["ab", "cd", "ef"]
