import wave
from array import array
from pathlib import Path

from app.media import SilenceSpan, parse_silencedetect_output, refine_speech_span_starts, speech_spans_from_silences


def test_parse_silencedetect_output() -> None:
    stderr = """
    [silencedetect @ 000001] silence_start: 3.25
    [silencedetect @ 000001] silence_end: 4.10 | silence_duration: 0.85
    [silencedetect @ 000001] silence_start: 8
    [silencedetect @ 000001] silence_end: 9.5 | silence_duration: 1.5
    """

    silences = parse_silencedetect_output(stderr)

    assert [(silence.start, silence.end) for silence in silences] == [(3.25, 4.1), (8.0, 9.5)]


def test_speech_spans_from_silences_splits_long_spans() -> None:
    silences = parse_silencedetect_output("silence_start: 5\nsilence_end: 6")

    spans = speech_spans_from_silences(20, silences, min_duration_s=1, max_duration_s=7)

    assert [(span.start, span.end) for span in spans] == [(0.0, 5.0), (6.0, 13.0), (13.0, 20.0)]


def test_speech_spans_from_silences_pads_and_merges_close_spans() -> None:
    silences = parse_silencedetect_output(
        "silence_start: 5\nsilence_end: 6\nsilence_start: 7\nsilence_end: 8"
    )

    spans = speech_spans_from_silences(
        12,
        silences,
        min_duration_s=0.5,
        max_duration_s=30,
        padding_s=0.25,
        merge_gap_s=1.5,
    )

    assert [(span.start, span.end) for span in spans] == [(0.0, 12.0)]


def test_speech_spans_from_silences_keeps_short_speech_with_padding() -> None:
    silences = parse_silencedetect_output(
        "silence_start: 0\nsilence_end: 16.889\nsilence_start: 17.463\nsilence_end: 30"
    )

    spans = speech_spans_from_silences(
        30,
        silences,
        min_duration_s=0.25,
        max_duration_s=30,
        padding_s=0.4,
        merge_gap_s=2.0,
    )

    assert [(round(span.start, 3), round(span.end, 3)) for span in spans] == [(16.489, 17.863)]


def test_refine_speech_span_starts_moves_start_to_onset(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    samples = array("h")
    samples.extend([0] * 16000)
    samples.extend([3000] * 16000)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    refinement = refine_speech_span_starts(
        wav_path,
        [SilenceSpan(0.0, 2.0)],
        "-42dB",
        threshold_offset_db=3.0,
        pre_roll_s=0.08,
        frame_duration_s=0.03,
        min_consecutive_frames=2,
        max_adjustment_s=2.0,
    )

    assert refinement.adjusted_count == 1
    assert round(refinement.spans[0].start, 2) == 0.91
    assert refinement.spans[0].end == 2.0
