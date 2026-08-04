from __future__ import annotations

import json
import wave
from pathlib import Path

from kotoba_standalone.subtitle import (
    SubtitleChunk,
    SubtitleQualityIssue,
    analyze_subtitle_quality,
    annotate_chunks_with_quality,
    apply_tail_refinements,
    audio_stats_for_chunks,
    filter_likely_hallucinations,
    normalize_phrase,
    recommended_action_for_flags,
    split_long_subtitle_candidates,
    split_long_subtitle_text,
    tail_retranscribe_candidate_indexes,
    text_similarity,
)


def test_normalize_phrase_removes_trailing_punctuation() -> None:
    assert normalize_phrase("ありがとうございました。") == "ありがとうございました"
    assert normalize_phrase(" ごめん。 ") == "ごめん"


def test_filter_likely_hallucinations_drops_blocked_phrase_and_near_zero_long_text() -> None:
    chunks = [
        SubtitleChunk(0.0, 1.0, "ありがとうございました"),
        SubtitleChunk(2.0, 2.01, "一日当たってるこれ気持ちいい"),
        SubtitleChunk(3.0, 4.0, "先生いっぱい触って"),
    ]

    kept, dropped = filter_likely_hallucinations(chunks)

    assert kept == [chunks[2]]
    assert [issue.text for issue in dropped] == ["ありがとうございました", "一日当たってるこれ気持ちいい"]
    assert dropped[0].recommended_action == "drop"


def test_analyze_subtitle_quality_flags_flat_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "flat.wav"
    write_silent_wav(wav_path, duration_s=2.0)
    chunks = [SubtitleChunk(0.0, 1.2, "ありがとうございました")]

    issues = analyze_subtitle_quality(chunks, wav_path)

    assert issues
    assert "flat_audio" in issues[0].flags
    assert "blocked_phrase" in issues[0].flags


def test_audio_stats_for_chunks_handles_out_of_range_timestamp(tmp_path: Path) -> None:
    wav_path = tmp_path / "short.wav"
    write_silent_wav(wav_path, duration_s=0.5)

    stats = audio_stats_for_chunks(wav_path, [SubtitleChunk(10.0, 11.0, "text")])

    assert stats[0]["frame_count"] == 0


def test_quality_issue_json_is_serializable(tmp_path: Path) -> None:
    wav_path = tmp_path / "flat.wav"
    write_silent_wav(wav_path, duration_s=1.0)

    issues = analyze_subtitle_quality([SubtitleChunk(0.0, 0.01, "長すぎる字幕")], wav_path)

    json.dumps([issue.__dict__ for issue in issues], ensure_ascii=False)


def test_recommended_action_distinguishes_drop_and_resegment_candidates() -> None:
    assert recommended_action_for_flags(["weak_audio_peak", "low_chars_per_second"]) == "drop_candidate"
    assert (
        recommended_action_for_flags(["strong_audio", "fragmented_energy", "low_chars_per_second"])
        == "resegment_candidate"
    )
    assert recommended_action_for_flags(["long_subtitle_split_candidate"]) == "split_candidate"


def test_split_long_subtitle_text_prefers_japanese_phrase_boundaries() -> None:
    text = "\u5148\u751f\u306e\u3046\u3061\u3093\u306b\u5165\u308c\u3066\u304f\u3060\u3055\u3044\u3044\u3063\u3071\u3044\u7a81\u304d\u51fa\u3057\u3066\u3054\u3089\u3093\u3046\u3093\u3053\u308c"
    parts = split_long_subtitle_text(
        text,
        max_parts=3,
        max_chars=24,
    )

    assert parts == [
        "\u5148\u751f\u306e\u3046\u3061\u3093\u306b\u5165\u308c\u3066\u304f\u3060\u3055\u3044",
        "\u3044\u3063\u3071\u3044\u7a81\u304d\u51fa\u3057\u3066\u3054\u3089\u3093",
        "\u3046\u3093\u3053\u308c",
    ]


def test_split_long_subtitle_candidates_distributes_time() -> None:
    chunk = SubtitleChunk(
        10.0,
        32.0,
        "\u5148\u751f\u306e\u3046\u3061\u3093\u306b\u5165\u308c\u3066\u304f\u3060\u3055\u3044\u3044\u3063\u3071\u3044\u7a81\u304d\u51fa\u3057\u3066\u3054\u3089\u3093\u3046\u3093\u3053\u308c",
    )
    issues = [
        type(
            "Issue",
            (),
            {
                "index": 1,
                "recommended_action": "split_candidate",
            },
        )()
    ]

    chunks, applied = split_long_subtitle_candidates([chunk], issues)  # type: ignore[arg-type]

    assert len(chunks) == 3
    assert chunks[0].start == 10.0
    assert chunks[-1].end == 32.0
    assert applied[0].recommended_action == "split_applied"


def test_annotate_chunks_with_quality_appends_tag_on_new_line() -> None:
    chunk = SubtitleChunk(0.0, 1.0, "text")
    issue = type(
        "Issue",
        (),
        {
            "index": 1,
            "recommended_action": "review",
            "flags": ["flag_a", "flag_b"],
        },
    )()

    annotated = annotate_chunks_with_quality([chunk], [issue])  # type: ignore[arg-type]

    assert annotated[0].text == "text\n[quality: review; flag_a, flag_b]"


def test_tail_retranscribe_candidate_indexes_selects_long_resegment_candidates() -> None:
    issues = [
        SubtitleQualityIssue(1, 0.0, 2.0, 2.0, "short", ["early_start_candidate"], "review"),
        SubtitleQualityIssue(2, 0.0, 15.0, 15.0, "long", ["long_duration_short_text"], "drop_candidate"),
        SubtitleQualityIssue(3, 0.0, 20.0, 20.0, "blocked", ["blocked_phrase"], "drop"),
        SubtitleQualityIssue(4, 0.0, 22.0, 22.0, "long text", ["strong_audio"], "resegment_candidate"),
    ]

    assert tail_retranscribe_candidate_indexes(issues) == {2, 4}


def test_text_similarity_handles_partial_tail_match() -> None:
    assert text_similarity(
        "\u5148\u751f\u306e\u3046\u3061\u3093\u306b\u5165\u308c\u3066\u304f\u3060\u3055\u3044",
        "\u5165\u308c\u3066\u304f\u3060\u3055\u3044",
    ) == 1.0
    assert text_similarity("\u5148\u751f", "\u3054\u98ef") < 0.5


def test_apply_tail_refinements_moves_start_forward() -> None:
    chunks = [SubtitleChunk(10.0, 30.0, "text")]

    refined, applied = apply_tail_refinements(
        chunks,
        {
            1: {
                "new_start": 25.25,
                "tail_start": 25.0,
                "similarity": 0.8,
            }
        },
    )

    assert refined[0] == SubtitleChunk(25.25, 30.0, "text")
    assert applied[0].recommended_action == "tail_refine_applied"


def write_silent_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    frame_count = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
