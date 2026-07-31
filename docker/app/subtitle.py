from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubtitleChunk:
    start: float
    end: float
    text: str


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+([\u3002\uff01\uff1f\u3001,.!?])", r"\1", text)
    text = re.sub(r"([\u300c\u300e\uff08])\s+", r"\1", text)
    text = re.sub(r"\s+([\u300d\u300f\uff09])", r"\1", text)
    return text


def normalize_chunks(raw_chunks: list[dict[str, Any]]) -> list[SubtitleChunk]:
    chunks: list[SubtitleChunk] = []
    previous_end = 0.0
    previous_text = ""

    for item in raw_chunks:
        timestamp = item.get("timestamp") or item.get("timestamps")
        if timestamp is None:
            continue
        start, end = _parse_timestamp(timestamp)
        text = clean_text(str(item.get("text", "")))
        if not text:
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        if end <= start:
            end = start + 0.01
        if start < previous_end:
            start = previous_end
            if end <= start:
                end = start + 0.01
        if text == previous_text:
            continue
        chunks.append(SubtitleChunk(start=start, end=end, text=text))
        previous_end = end
        previous_text = text

    return chunks


def group_chunks_by_timing(
    chunks: list[SubtitleChunk],
    max_gap_s: float,
    max_duration_s: float,
    max_chars: int,
) -> list[SubtitleChunk]:
    grouped: list[SubtitleChunk] = []
    current: SubtitleChunk | None = None

    for chunk in chunks:
        if current is None:
            current = chunk
            continue
        gap = chunk.start - current.end
        merged_text = _join_text(current.text, chunk.text)
        should_break = (
            gap > max_gap_s
            or chunk.end - current.start > max_duration_s
            or len(merged_text) > max_chars
            or _ends_sentence(current.text)
        )
        if should_break:
            grouped.append(current)
            current = chunk
        else:
            current = SubtitleChunk(current.start, chunk.end, merged_text)

    if current is not None:
        grouped.append(current)
    return grouped


def filter_short_repeated_phrases(
    chunks: list[SubtitleChunk],
    phrases: list[str],
    max_duration_s: float,
) -> list[SubtitleChunk]:
    normalized_phrases = {_normalize_phrase(phrase) for phrase in phrases}
    filtered: list[SubtitleChunk] = []
    for chunk in chunks:
        duration = chunk.end - chunk.start
        if duration <= max_duration_s and _normalize_phrase(chunk.text) in normalized_phrases:
            continue
        filtered.append(chunk)
    return filtered


def filter_standalone_phrases(
    chunks: list[SubtitleChunk],
    phrases: list[str],
) -> list[SubtitleChunk]:
    normalized_phrases = {_normalize_phrase(phrase) for phrase in phrases}
    if not normalized_phrases:
        return chunks
    return [chunk for chunk in chunks if _normalize_phrase(chunk.text) not in normalized_phrases]


def filter_punctuation_only_chunks(chunks: list[SubtitleChunk]) -> list[SubtitleChunk]:
    return [chunk for chunk in chunks if not _is_punctuation_only(chunk.text)]


def tighten_fallback_subtitle_durations(
    chunks: list[SubtitleChunk],
    min_duration_s: float,
    max_duration_s: float,
    chars_per_second: float,
    padding_s: float,
) -> list[SubtitleChunk]:
    tightened: list[SubtitleChunk] = []
    min_shorten_delta_s = 0.1
    for chunk in chunks:
        duration = chunk.end - chunk.start
        target_duration = _fallback_target_duration(
            chunk.text,
            min_duration_s,
            max_duration_s,
            chars_per_second,
            padding_s,
        )
        if duration <= target_duration or duration - target_duration < min_shorten_delta_s:
            tightened.append(chunk)
            continue
        tightened.append(SubtitleChunk(chunk.start, chunk.start + target_duration, chunk.text))
    return tightened


def shift_subtitle_timings(chunks: list[SubtitleChunk], offset_s: float) -> list[SubtitleChunk]:
    if offset_s == 0:
        return chunks
    return [
        SubtitleChunk(max(0.0, chunk.start + offset_s), max(0.0, chunk.end + offset_s), chunk.text)
        for chunk in chunks
    ]


def clamp_subtitle_timings(chunks: list[SubtitleChunk], media_duration_s: float) -> list[SubtitleChunk]:
    if media_duration_s <= 0:
        return chunks
    clamped: list[SubtitleChunk] = []
    for chunk in chunks:
        if chunk.start >= media_duration_s:
            continue
        end = min(chunk.end, media_duration_s)
        if end > chunk.start:
            clamped.append(SubtitleChunk(chunk.start, end, chunk.text))
    return clamped


def split_chunks_on_silence(
    chunks: list[SubtitleChunk],
    silences: list[Any],
    min_subtitle_duration_s: float,
) -> list[SubtitleChunk]:
    split_chunks: list[SubtitleChunk] = []
    for chunk in chunks:
        internal_silences = [
            silence
            for silence in silences
            if silence.start > chunk.start
            and silence.end < chunk.end
            and silence.start - chunk.start >= min_subtitle_duration_s
            and chunk.end - silence.end >= min_subtitle_duration_s
        ]
        if not internal_silences:
            split_chunks.append(chunk)
            continue

        spans = _speech_spans_for_chunk(chunk, internal_silences, min_subtitle_duration_s)
        if len(spans) <= 1:
            split_chunks.append(chunk)
            continue

        pieces = split_text_for_spans(chunk.text, [end - start for start, end in spans])
        for (start, end), text in zip(spans, pieces):
            cleaned = clean_text(text)
            if cleaned:
                split_chunks.append(SubtitleChunk(start=start, end=end, text=cleaned))
    return split_chunks


def _speech_spans_for_chunk(
    chunk: SubtitleChunk,
    silences: list[Any],
    min_duration_s: float,
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = chunk.start
    for silence in silences:
        if silence.start - cursor >= min_duration_s:
            spans.append((cursor, silence.start))
        cursor = max(cursor, silence.end)
    if chunk.end - cursor >= min_duration_s:
        spans.append((cursor, chunk.end))
    return spans


def split_text_for_spans(text: str, durations: list[float]) -> list[str]:
    if len(durations) <= 1:
        return [text]
    units = _text_units(text)
    if len(units) <= 1 or len(units) < len(durations):
        return _split_text_by_character_count(text, len(durations))

    total_duration = sum(durations)
    total_chars = sum(len(unit) for unit in units)
    pieces: list[str] = []
    unit_index = 0
    for index, duration in enumerate(durations):
        remaining_groups = len(durations) - index
        if remaining_groups == 1:
            pieces.append("".join(units[unit_index:]))
            break
        target_chars = max(1, round(total_chars * duration / total_duration))
        current: list[str] = []
        current_chars = 0
        max_unit_index = len(units) - (remaining_groups - 1)
        while unit_index < max_unit_index:
            unit = units[unit_index]
            if current and current_chars + len(unit) > target_chars:
                break
            current.append(unit)
            current_chars += len(unit)
            unit_index += 1
        if not current:
            current.append(units[unit_index])
            unit_index += 1
        pieces.append("".join(current))
    return pieces


def _text_units(text: str) -> list[str]:
    units = re.findall(r".+?[\u3002\uff01\uff1f!?]+|.+?[\u3001,]+|.+", text)
    return [unit.strip() for unit in units if unit.strip()]


def _split_text_by_character_count(text: str, count: int) -> list[str]:
    if count <= 1:
        return [text]
    length = len(text)
    return [
        text[round(length * index / count) : round(length * (index + 1) / count)]
        for index in range(count)
    ]


def _join_text(left: str, right: str) -> str:
    separator = " " if _needs_space_between(left, right) else ""
    return clean_text(left + separator + right)


def _needs_space_between(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[\u3002\uff01\uff1f.!?]\s*$", text))


def _normalize_phrase(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[\s\u3000]+", "", text)
    return text


def _is_punctuation_only(text: str) -> bool:
    text = re.sub(r"[\s\u3000]+", "", clean_text(text))
    return bool(text) and not any(char.isalnum() for char in text)


def _fallback_target_duration(
    text: str,
    min_duration_s: float,
    max_duration_s: float,
    chars_per_second: float,
    padding_s: float,
) -> float:
    readable_chars = len(re.sub(r"[\s\u3000]+", "", clean_text(text)))
    estimated = readable_chars / chars_per_second + padding_s
    return max(min_duration_s, min(max_duration_s, estimated))


def _parse_timestamp(timestamp: Any) -> tuple[float, float]:
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = 0.0 if timestamp[0] is None else float(timestamp[0])
        end = start if timestamp[1] is None else float(timestamp[1])
        return start, end
    if isinstance(timestamp, dict):
        start = 0.0 if timestamp.get("start") is None else float(timestamp["start"])
        end = start if timestamp.get("end") is None else float(timestamp["end"])
        return start, end
    raise ValueError(f"Unsupported timestamp format: {timestamp!r}")


def chunks_to_srt(chunks: list[SubtitleChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        blocks.append(
            f"{index}\n{format_srt_time(chunk.start)} --> {format_srt_time(chunk.end)}\n{chunk.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def chunks_to_txt(chunks: list[SubtitleChunk]) -> str:
    return "\n".join(chunk.text for chunk in chunks) + ("\n" if chunks else "")


def chunks_to_json(chunks: list[SubtitleChunk]) -> list[dict[str, Any]]:
    return [{"start": c.start, "end": c.end, "text": c.text} for c in chunks]
