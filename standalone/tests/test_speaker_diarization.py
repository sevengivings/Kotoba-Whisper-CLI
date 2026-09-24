from __future__ import annotations

from kotoba_standalone.speaker_diarization import SpeakerTurn, group_chunks_by_speaker, speaker_for_chunk
from kotoba_standalone.subtitle import SubtitleChunk, normalize_chunks


def test_grouping_breaks_at_speaker_change_and_keeps_repeated_words() -> None:
    raw = [
        {"timestamp": [0.0, 0.4], "text": "はい"},
        {"timestamp": [0.5, 0.9], "text": "はい"},
        {"timestamp": [1.0, 1.4], "text": "いいえ"},
        {"timestamp": [1.5, 1.9], "text": "そう"},
    ]
    turns = [SpeakerTurn(0.0, 0.95, "SPEAKER_00"), SpeakerTurn(0.95, 2.0, "SPEAKER_01")]

    grouped = group_chunks_by_speaker(normalize_chunks(raw, deduplicate=False), turns)

    assert len(grouped) == 2
    assert grouped[0].text.count("はい") == 2
    assert grouped[1].text == "いいえそう"


def test_speaker_assignment_uses_largest_overlap() -> None:
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(0.8, 1.5, "SPEAKER_01")]

    assert speaker_for_chunk(SubtitleChunk(0.9, 1.4, "test"), turns) == "SPEAKER_01"
    assert speaker_for_chunk(SubtitleChunk(2.0, 2.5, "test"), turns) is None


def test_short_aligned_token_gets_readable_display_duration() -> None:
    chunks = [SubtitleChunk(0.0, 0.01, "あ"), SubtitleChunk(1.0, 1.5, "はい")]
    turns = [SpeakerTurn(0.0, 0.1, "SPEAKER_00"), SpeakerTurn(1.0, 1.5, "SPEAKER_01")]

    grouped = group_chunks_by_speaker(chunks, turns)

    assert grouped[0].end == 0.65
    assert grouped[0].end <= grouped[1].start


def test_sentence_final_particle_stays_with_previous_speaker() -> None:
    chunks = [
        SubtitleChunk(0.0, 0.5, "どう"),
        SubtitleChunk(0.5, 0.7, "だ？"),
        SubtitleChunk(0.7, 1.3, "何？"),
    ]
    turns = [SpeakerTurn(0.0, 0.5, "SPEAKER_00"), SpeakerTurn(0.5, 1.3, "SPEAKER_01")]

    grouped = group_chunks_by_speaker(chunks, turns)

    assert grouped[0].text == "どうだ？"
    assert grouped[1].text == "何？"


def test_very_short_turn_uses_time_from_following_cue() -> None:
    chunks = [SubtitleChunk(0.0, 0.01, "はい。"), SubtitleChunk(0.01, 1.5, "いいえ。")]
    turns = [SpeakerTurn(0.0, 0.01, "SPEAKER_00"), SpeakerTurn(0.01, 1.5, "SPEAKER_01")]

    grouped = group_chunks_by_speaker(chunks, turns)

    assert grouped[0].end == 0.5
    assert grouped[1].start == 0.5
