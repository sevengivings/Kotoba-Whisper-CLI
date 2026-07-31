from __future__ import annotations

import math
import re
import statistics
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubtitleChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleQualityIssue:
    index: int
    start: float
    end: float
    duration: float
    text: str
    flags: list[str]
    recommended_action: str
    audio: dict[str, float | int] | None = None


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


def normalize_phrase(text: str) -> str:
    normalized = clean_text(text)
    normalized = re.sub(r"[\s\u3000]+", "", normalized)
    normalized = re.sub(r"[\u3002\u3001\uff0c\uff0e,.!?\uff01\uff1f\u2026]+$", "", normalized)
    return normalized


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
        chunks.append(SubtitleChunk(start, end, text))
        previous_end = end
        previous_text = text
    return chunks


def filter_likely_hallucinations(chunks: list[SubtitleChunk]) -> tuple[list[SubtitleChunk], list[SubtitleQualityIssue]]:
    kept: list[SubtitleChunk] = []
    dropped: list[SubtitleQualityIssue] = []
    for index, chunk in enumerate(chunks, 1):
        flags = hallucination_flags(chunk)
        should_drop = bool(
            "blocked_phrase" in flags
            or "punctuation_only" in flags
            or ("near_zero_duration" in flags and len(normalize_phrase(chunk.text)) >= 5)
        )
        if should_drop:
            dropped.append(
                SubtitleQualityIssue(
                    index=index,
                    start=chunk.start,
                    end=chunk.end,
                    duration=chunk.end - chunk.start,
                    text=chunk.text,
                    flags=flags,
                    recommended_action="drop",
                )
            )
        else:
            kept.append(chunk)
    return kept, dropped


def split_long_subtitle_candidates(
    chunks: list[SubtitleChunk],
    issues: list[SubtitleQualityIssue],
    target_duration_s: float = 8.0,
    max_chars: int = 28,
) -> tuple[list[SubtitleChunk], list[SubtitleQualityIssue]]:
    split_indexes = {issue.index for issue in issues if issue.recommended_action == "split_candidate"}
    if not split_indexes:
        return chunks, []
    result: list[SubtitleChunk] = []
    applied: list[SubtitleQualityIssue] = []
    for index, chunk in enumerate(chunks, 1):
        if index not in split_indexes:
            result.append(chunk)
            continue
        parts = split_long_subtitle_text(chunk.text, max_parts=max(2, math.ceil((chunk.end - chunk.start) / target_duration_s)), max_chars=max_chars)
        if len(parts) <= 1:
            result.append(chunk)
            continue
        split_chunks = _distribute_subtitle_time(chunk, parts)
        result.extend(split_chunks)
        applied.append(
            SubtitleQualityIssue(
                index=index,
                start=chunk.start,
                end=chunk.end,
                duration=chunk.end - chunk.start,
                text=chunk.text,
                flags=["split_applied"],
                recommended_action="split_applied",
                audio={"part_count": len(split_chunks)},
            )
        )
    return result, applied


def tail_retranscribe_candidate_indexes(issues: list[SubtitleQualityIssue]) -> set[int]:
    return {
        issue.index
        for issue in issues
        if 10.0 <= issue.duration <= 30.0
        and issue.recommended_action in {"drop_candidate", "resegment_candidate"}
        and "blocked_phrase" not in issue.flags
        and "near_zero_duration" not in issue.flags
    }


def apply_tail_refinements(
    chunks: list[SubtitleChunk],
    refinements: dict[int, dict[str, Any]],
) -> tuple[list[SubtitleChunk], list[SubtitleQualityIssue]]:
    if not refinements:
        return chunks, []
    result: list[SubtitleChunk] = []
    applied: list[SubtitleQualityIssue] = []
    for index, chunk in enumerate(chunks, 1):
        refinement = refinements.get(index)
        if refinement is None:
            result.append(chunk)
            continue
        new_start = float(refinement["new_start"])
        if new_start <= chunk.start or new_start >= chunk.end:
            result.append(chunk)
            continue
        adjusted = SubtitleChunk(new_start, chunk.end, chunk.text)
        result.append(adjusted)
        applied.append(
            SubtitleQualityIssue(
                index=index,
                start=chunk.start,
                end=chunk.end,
                duration=chunk.end - chunk.start,
                text=chunk.text,
                flags=["tail_refine_applied"],
                recommended_action="tail_refine_applied",
                audio={
                    "new_start": round(new_start, 3),
                    "tail_start": round(float(refinement["tail_start"]), 3),
                    "similarity": round(float(refinement["similarity"]), 3),
                },
            )
        )
    return result, applied


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_phrase(left)
    right_norm = normalize_phrase(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    left_grams = _char_ngrams(left_norm, 2)
    right_grams = _char_ngrams(right_norm, 2)
    if not left_grams or not right_grams:
        return 1.0 if left_norm == right_norm else 0.0
    intersection = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    return intersection / union if union else 0.0


def split_long_subtitle_text(text: str, max_parts: int, max_chars: int = 28) -> list[str]:
    text = clean_text(text)
    if max_parts <= 1 or len(normalize_phrase(text)) <= max_chars:
        return [text]
    units = _split_text_units(text)
    if len(units) <= 1:
        return _split_text_by_character_count(text, max_parts)
    return _merge_units_for_balanced_parts(units, max_parts=max_parts, max_chars=max_chars)


def annotate_chunks_with_quality(chunks: list[SubtitleChunk], issues: list[SubtitleQualityIssue]) -> list[SubtitleChunk]:
    issue_map = {issue.index: issue for issue in issues}
    annotated: list[SubtitleChunk] = []
    for index, chunk in enumerate(chunks, 1):
        issue = issue_map.get(index)
        if issue is None:
            annotated.append(chunk)
            continue
        tag = f"[quality: {issue.recommended_action}; {', '.join(issue.flags)}]"
        annotated.append(SubtitleChunk(chunk.start, chunk.end, f"{chunk.text}\n{tag}"))
    return annotated


def hallucination_flags(chunk: SubtitleChunk) -> list[str]:
    flags: list[str] = []
    duration = chunk.end - chunk.start
    normalized = normalize_phrase(chunk.text)
    if duration < 0.3:
        flags.append("near_zero_duration")
    if _is_punctuation_only(chunk.text):
        flags.append("punctuation_only")
    if normalized in _BLOCKED_STANDALONE_PHRASES:
        flags.append("blocked_phrase")
    if duration >= 8.0 and len(normalized) <= 30:
        flags.append("long_duration_short_text")
    if _looks_like_garbage_cjk(normalized):
        flags.append("possible_garbage_cjk")
    return flags


def analyze_subtitle_quality(chunks: list[SubtitleChunk], wav_path: Path | None = None) -> list[SubtitleQualityIssue]:
    audio_stats = audio_stats_for_chunks(wav_path, chunks) if wav_path is not None and wav_path.exists() else [None] * len(chunks)
    issues: list[SubtitleQualityIssue] = []
    for index, (chunk, audio) in enumerate(zip(chunks, audio_stats, strict=False), 1):
        flags = hallucination_flags(chunk)
        duration = chunk.end - chunk.start
        normalized = normalize_phrase(chunk.text)
        chars_per_second = len(normalized) / duration if duration > 0 else float("inf")
        if duration >= 12.0 and len(normalized) >= 25:
            flags.append("long_subtitle_split_candidate")
        if duration >= 8.0 and chars_per_second < 2.0:
            flags.append("low_chars_per_second")
        if audio is not None:
            if duration <= 3.0 and audio["max_db"] < -45.0:
                flags.append("flat_audio")
            if duration >= 8.0 and audio["max_db"] < -22.0 and audio["p90_db"] < -30.0:
                flags.append("weak_audio_peak")
            if duration >= 8.0 and audio["max_db"] >= -22.0 and audio["p90_db"] >= -30.0:
                flags.append("strong_audio")
            if duration >= 8.0 and audio["speech_ratio"] < 0.18 and audio["longest_island_s"] < 0.8:
                flags.append("low_speech_density")
            if duration >= 8.0 and audio["longest_island_s"] < 0.8:
                flags.append("fragmented_energy")
            if 1.5 <= duration < 8.0 and audio["leading_quiet_s"] >= 0.7 and "blocked_phrase" not in flags:
                flags.append("early_start_candidate")
        if not flags:
            continue
        action = recommended_action_for_flags(flags)
        issues.append(
            SubtitleQualityIssue(
                index=index,
                start=chunk.start,
                end=chunk.end,
                duration=duration,
                text=chunk.text,
                flags=flags,
                recommended_action=action,
                audio=audio,
            )
        )
    return issues


def quality_issues_to_json(issues: list[SubtitleQualityIssue]) -> list[dict[str, Any]]:
    return [
        {
            "index": issue.index,
            "start": issue.start,
            "end": issue.end,
            "duration": issue.duration,
            "text": issue.text,
            "flags": issue.flags,
            "recommended_action": issue.recommended_action,
            "audio": issue.audio,
        }
        for issue in issues
    ]


def recommended_action_for_flags(flags: list[str]) -> str:
    if "blocked_phrase" in flags or "punctuation_only" in flags:
        return "drop"
    if "near_zero_duration" in flags:
        return "drop"
    if "flat_audio" in flags and "low_chars_per_second" in flags:
        return "drop_candidate"
    if (
        ("weak_audio_peak" in flags or ("fragmented_energy" in flags and "strong_audio" not in flags))
        and "low_chars_per_second" in flags
    ):
        return "drop_candidate"
    if "strong_audio" in flags and "fragmented_energy" in flags and "low_chars_per_second" in flags:
        return "resegment_candidate"
    if "long_subtitle_split_candidate" in flags:
        return "split_candidate"
    if "early_start_candidate" in flags:
        return "refine_start_candidate"
    return "review"


def audio_stats_for_chunks(wav_path: Path, chunks: list[SubtitleChunk], frame_duration_s: float = 0.05) -> list[dict[str, float | int]]:
    with wave.open(str(wav_path), "rb") as wav:
        frame_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            return [_empty_audio_stats() for _ in chunks]
        return [_audio_stats_for_chunk(wav, frame_rate, channels, chunk, frame_duration_s) for chunk in chunks]


def group_chunks_by_timing(
    chunks: list[SubtitleChunk],
    max_gap_s: float = 0.5,
    max_duration_s: float = 6.0,
    max_chars: int = 42,
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


def chunks_to_srt(chunks: list[SubtitleChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        blocks.append(f"{index}\n{format_srt_time(chunk.start)} --> {format_srt_time(chunk.end)}\n{chunk.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def chunks_to_txt(chunks: list[SubtitleChunk]) -> str:
    return "\n".join(chunk.text for chunk in chunks) + ("\n" if chunks else "")


def chunks_to_json(chunks: list[SubtitleChunk]) -> list[dict[str, Any]]:
    return [{"start": chunk.start, "end": chunk.end, "text": chunk.text} for chunk in chunks]


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


def _join_text(left: str, right: str) -> str:
    separator = " " if _needs_space_between(left, right) else ""
    return clean_text(left + separator + right)


def _needs_space_between(left: str, right: str) -> bool:
    return bool(left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum())


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[\u3002\uff01\uff1f.!?]\s*$", text))


def _distribute_subtitle_time(chunk: SubtitleChunk, parts: list[str]) -> list[SubtitleChunk]:
    total_duration = chunk.end - chunk.start
    weights = [max(1, len(normalize_phrase(part))) for part in parts]
    total_weight = sum(weights)
    cursor = chunk.start
    result: list[SubtitleChunk] = []
    for index, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        if index == len(parts) - 1:
            end = chunk.end
        else:
            end = cursor + total_duration * weight / total_weight
        result.append(SubtitleChunk(cursor, end, part))
        cursor = end
    return result


def _split_text_units(text: str) -> list[str]:
    boundaries = (
        r"[\u3002\uff01\uff1f.!?]+|"
        r"(?:くださいね|ください|下さい|ごらん|ちょうだい|していいよ|していい|"
        r"いいの|いいよ|だよ|だね|なの|ですか|ますか|じゃん|ねえ|はい|うん)"
    )
    units: list[str] = []
    cursor = 0
    for match in re.finditer(boundaries, text):
        end = match.end()
        unit = clean_text(text[cursor:end])
        if unit:
            units.append(unit)
        cursor = end
    tail = clean_text(text[cursor:])
    if tail:
        units.append(tail)
    return units or [text]


def _merge_units_for_balanced_parts(units: list[str], max_parts: int, max_chars: int) -> list[str]:
    total_chars = sum(len(normalize_phrase(unit)) for unit in units)
    natural_parts = min(max_parts, len(units))
    target_parts = min(max_parts, max(2, math.ceil(total_chars / max_chars), natural_parts))
    parts: list[str] = []
    current = ""
    remaining_units = list(units)
    while remaining_units:
        remaining_groups = max(1, target_parts - len(parts))
        remaining_chars = sum(len(normalize_phrase(unit)) for unit in remaining_units)
        target_chars = max(1, math.ceil(remaining_chars / remaining_groups))
        unit = remaining_units.pop(0)
        candidate = current + unit
        if current and len(normalize_phrase(candidate)) > target_chars and len(parts) < target_parts - 1:
            parts.append(current)
            current = unit
        else:
            current = candidate
    if current:
        parts.append(current)
    if len(parts) > max_parts:
        return _split_text_by_character_count("".join(parts), max_parts)
    return parts


def _split_text_by_character_count(text: str, parts: int) -> list[str]:
    normalized_length = len(text)
    if parts <= 1 or normalized_length <= 1:
        return [text]
    chunk_size = math.ceil(normalized_length / parts)
    return [text[index : index + chunk_size] for index in range(0, normalized_length, chunk_size)]


def _char_ngrams(text: str, size: int) -> set[str]:
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(0, len(text) - size + 1)}


def _audio_stats_for_chunk(
    wav: wave.Wave_read,
    frame_rate: int,
    channels: int,
    chunk: SubtitleChunk,
    frame_duration_s: float,
) -> dict[str, float | int]:
    duration = max(0.0, chunk.end - chunk.start)
    if duration <= 0:
        return _empty_audio_stats()
    start_frame = max(0, int(chunk.start * frame_rate))
    if start_frame >= wav.getnframes():
        return _empty_audio_stats()
    wav.setpos(start_frame)
    raw = wav.readframes(max(1, int(duration * frame_rate)))
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array("h", samples[::channels])
    samples_per_frame = max(1, int(frame_rate * frame_duration_s))
    levels: list[float] = []
    for offset in range(0, len(samples), samples_per_frame):
        frame = samples[offset : offset + samples_per_frame]
        if not frame:
            continue
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        levels.append(20 * math.log10(max(rms, 1.0) / 32768.0))
    if not levels:
        return _empty_audio_stats()
    median_db = statistics.median(levels)
    max_db = max(levels)
    threshold = median_db + 8.0
    active = [level > threshold for level in levels]
    speech_frames = sum(1 for value in active if value)
    longest_island = _longest_true_run(active) * frame_duration_s
    leading_quiet = _leading_quiet_frames(levels, threshold) * frame_duration_s
    return {
        "frame_count": len(levels),
        "median_db": round(median_db, 3),
        "max_db": round(max_db, 3),
        "p90_db": round(sorted(levels)[min(len(levels) - 1, int(len(levels) * 0.9))], 3),
        "speech_ratio": round(speech_frames / len(levels), 3),
        "longest_island_s": round(longest_island, 3),
        "leading_quiet_s": round(leading_quiet, 3),
    }


def _empty_audio_stats() -> dict[str, float | int]:
    return {
        "frame_count": 0,
        "median_db": -120.0,
        "max_db": -120.0,
        "p90_db": -120.0,
        "speech_ratio": 0.0,
        "longest_island_s": 0.0,
        "leading_quiet_s": 0.0,
    }


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _leading_quiet_frames(levels: list[float], threshold: float) -> int:
    count = 0
    for level in levels:
        if level > threshold:
            break
        count += 1
    return count


def _is_punctuation_only(text: str) -> bool:
    normalized = re.sub(r"[\s\u3000]+", "", text)
    return bool(normalized) and not re.search(r"[\w\u3040-\u30ff\u3400-\u9fff]", normalized)


def _looks_like_garbage_cjk(text: str) -> bool:
    if len(text) < 8:
        return False
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    kana_count = len(re.findall(r"[\u3040-\u30ff]", text))
    return cjk_count >= 8 and kana_count == 0 and cjk_count / max(1, len(text)) >= 0.8


_BLOCKED_STANDALONE_PHRASES = {
    "ごめん",
    "ありがとうございました",
    "ありがとうございます",
    "ご視聴ありがとうございました",
}
