from __future__ import annotations

from kotoba_standalone.media import SilenceSpan, parse_silencedetect_output, speech_spans_from_silences


def test_parse_silencedetect_output_pairs_start_and_end() -> None:
    stderr = """
[silencedetect @ 000001] silence_start: 1.234
[silencedetect @ 000001] silence_end: 2.500 | silence_duration: 1.266
[silencedetect @ 000001] silence_start: 4
[silencedetect @ 000001] silence_end: 5.25 | silence_duration: 1.25
"""

    assert parse_silencedetect_output(stderr) == [
        SilenceSpan(1.234, 2.5),
        SilenceSpan(4.0, 5.25),
    ]


def test_speech_spans_from_silences_applies_padding_and_max_duration() -> None:
    spans = speech_spans_from_silences(
        duration_s=10.0,
        silences=[SilenceSpan(2.0, 3.0), SilenceSpan(7.0, 8.0)],
        min_duration_s=0.25,
        max_duration_s=3.0,
        padding_s=0.2,
        merge_gap_s=0.0,
    )

    assert spans == [
        SilenceSpan(0.0, 2.2),
        SilenceSpan(2.8, 5.8),
        SilenceSpan(5.8, 7.2),
        SilenceSpan(7.8, 10.0),
    ]
