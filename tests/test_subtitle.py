from dataclasses import dataclass

from app.subtitle import (
    SubtitleChunk,
    chunks_to_srt,
    format_srt_time,
    filter_punctuation_only_chunks,
    group_chunks_by_timing,
    filter_short_repeated_phrases,
    filter_standalone_phrases,
    normalize_chunks,
    shift_subtitle_timings,
    split_chunks_on_silence,
    split_text_for_spans,
    tighten_fallback_subtitle_durations,
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
        SubtitleChunk(0.0, 1.0, "\u3059\u307f\u307e\u305b\u3093\u3002"),
        SubtitleChunk(2.0, 5.0, "\u3059\u307f\u307e\u305b\u3093\u3002"),
        SubtitleChunk(6.0, 7.0, "\u3059\u307f\u307e\u305b\u3093\u3001\u5f85\u3063\u3066\u3002"),
        SubtitleChunk(8.0, 9.0, "\u306f\u3044\u3002"),
    ]

    filtered = filter_short_repeated_phrases(
        chunks,
        ["\u3059\u307f\u307e\u305b\u3093\u3002"],
        max_duration_s=1.6,
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in filtered] == [
        (2.0, 5.0, "\u3059\u307f\u307e\u305b\u3093\u3002"),
        (6.0, 7.0, "\u3059\u307f\u307e\u305b\u3093\u3001\u5f85\u3063\u3066\u3002"),
        (8.0, 9.0, "\u306f\u3044\u3002"),
    ]


def test_filter_standalone_phrases_removes_exact_phrase_regardless_duration() -> None:
    chunks = [
        SubtitleChunk(0.0, 1.0, "\u3054\u3081\u3093\u3002"),
        SubtitleChunk(2.0, 5.0, "\u3054\u3081\u3093\u3002"),
        SubtitleChunk(6.0, 7.0, "\u3054\u3081\u3093\u3001\u5f85\u3063\u3066\u3002"),
        SubtitleChunk(8.0, 9.0, "\u306f\u3044\u3002"),
    ]

    filtered = filter_standalone_phrases(chunks, ["\u3054\u3081\u3093\u3002"])

    assert filtered == chunks[2:]


def test_filter_punctuation_only_chunks_removes_orphan_punctuation() -> None:
    chunks = [
        SubtitleChunk(0.0, 0.8, "."),
        SubtitleChunk(1.0, 1.8, "\u3002"),
        SubtitleChunk(2.0, 2.8, "??"),
        SubtitleChunk(3.0, 3.8, "\u3001"),
        SubtitleChunk(4.0, 5.0, "\u305d\u3046\u3002"),
        SubtitleChunk(6.0, 7.0, "3\u304b\u6708"),
    ]

    filtered = filter_punctuation_only_chunks(chunks)

    assert filtered == chunks[4:]


def test_tighten_fallback_subtitle_durations_caps_short_text() -> None:
    chunks = [
        SubtitleChunk(10.0, 25.0, "\u3042\u308c\u3001\u9375\u958b\u3051\u3063\u3071\u3058\u3083\u3093\u3002"),
        SubtitleChunk(30.0, 32.0, "\u9577\u3081\u306e\u6587\u7ae0\u306f\u305d\u306e\u307e\u307e\u3002"),
    ]

    tightened = tighten_fallback_subtitle_durations(
        chunks,
        min_duration_s=0.8,
        max_duration_s=5.0,
        chars_per_second=7.0,
        padding_s=0.4,
    )

    assert tightened[0].start == 10.0
    assert round(tightened[0].end, 3) == 12.114
    assert tightened[0].text == chunks[0].text
    assert tightened[1] == chunks[1]


def test_shift_subtitle_timings_offsets_start_and_end() -> None:
    chunks = [
        SubtitleChunk(1.0, 2.0, "A"),
        SubtitleChunk(3.0, 4.0, "B"),
    ]

    shifted = shift_subtitle_timings(chunks, 0.5)

    assert shifted == [
        SubtitleChunk(1.5, 2.5, "A"),
        SubtitleChunk(3.5, 4.5, "B"),
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
