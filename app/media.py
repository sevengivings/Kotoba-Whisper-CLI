from __future__ import annotations

import json
import math
import re
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


@dataclass(frozen=True)
class SilenceSpan:
    start: float
    end: float


@dataclass(frozen=True)
class SpeechStartRefinement:
    spans: list[SilenceSpan]
    adjusted_count: int
    average_adjustment_s: float
    maximum_adjustment_s: float


@dataclass(frozen=True)
class SilenceThresholdEstimate:
    threshold_db: str
    noise_floor_db: float
    speech_level_db: float
    analyzed_frame_count: int


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract_audio(input_file: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-af",
        "aresample=async=1:first_pts=0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg failed without stderr"
        raise RuntimeError(f"FFmpeg audio extraction failed for {input_file}: {stderr}")


def extract_audio_segment(source_wav_path: Path, segment_wav_path: Path, start: float, end: float) -> None:
    segment_wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source_wav_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(segment_wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg segment extraction failed without stderr"
        raise RuntimeError(f"FFmpeg segment extraction failed for {source_wav_path}: {stderr}")


def probe_duration_seconds(input_file: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def detect_silences(
    wav_path: Path,
    threshold_db: str,
    min_duration_s: float,
) -> list[SilenceSpan]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(wav_path),
        "-vn",
        "-af",
        f"silencedetect=noise={threshold_db}:d={min_duration_s}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg silencedetect failed without stderr"
        raise RuntimeError(f"FFmpeg silence detection failed for {wav_path}: {stderr}")
    return parse_silencedetect_output(result.stderr)


def estimate_silence_threshold(
    wav_path: Path,
    min_threshold_db: float = -42.0,
    max_threshold_db: float = -28.0,
    frame_duration_s: float = 0.03,
) -> SilenceThresholdEstimate:
    frame_levels = _wav_frame_dbfs(wav_path, frame_duration_s)
    if not frame_levels:
        threshold = _format_db(min_threshold_db)
        return SilenceThresholdEstimate(threshold, min_threshold_db, min_threshold_db, 0)

    sorted_levels = sorted(frame_levels)
    noise_floor = _percentile(sorted_levels, 20)
    speech_level = _percentile(sorted_levels, 85)
    dynamic_range = max(0.0, speech_level - noise_floor)
    threshold = noise_floor + max(6.0, min(14.0, dynamic_range * 0.35))
    threshold = max(min_threshold_db, min(max_threshold_db, threshold))
    return SilenceThresholdEstimate(
        _format_db(threshold),
        round(noise_floor, 2),
        round(speech_level, 2),
        len(frame_levels),
    )


def refine_speech_span_starts(
    wav_path: Path,
    spans: list[SilenceSpan],
    silence_threshold_db: str,
    threshold_offset_db: float,
    pre_roll_s: float,
    frame_duration_s: float,
    min_consecutive_frames: int,
    max_adjustment_s: float,
) -> SpeechStartRefinement:
    if not spans:
        return SpeechStartRefinement([], 0, 0.0, 0.0)
    if frame_duration_s <= 0 or min_consecutive_frames < 1 or max_adjustment_s <= 0:
        return SpeechStartRefinement(spans, 0, 0.0, 0.0)

    onset_threshold_db = _parse_db(silence_threshold_db) + threshold_offset_db
    refined: list[SilenceSpan] = []
    adjustments: list[float] = []
    for span in spans:
        search_end = min(span.end, span.start + max_adjustment_s)
        onset = _find_first_speech_frame(
            wav_path,
            span.start,
            search_end,
            onset_threshold_db,
            frame_duration_s,
            min_consecutive_frames,
        )
        if onset is None:
            refined.append(span)
            continue
        new_start = max(span.start, min(onset - pre_roll_s, span.end))
        if span.end - new_start <= 0:
            refined.append(span)
            continue
        adjustment = new_start - span.start
        if adjustment <= 0:
            refined.append(span)
            continue
        refined.append(SilenceSpan(new_start, span.end))
        adjustments.append(adjustment)

    if not adjustments:
        return SpeechStartRefinement(refined, 0, 0.0, 0.0)
    return SpeechStartRefinement(
        refined,
        len(adjustments),
        round(sum(adjustments) / len(adjustments), 3),
        round(max(adjustments), 3),
    )


def parse_silencedetect_output(stderr: str) -> list[SilenceSpan]:
    silences: list[SilenceSpan] = []
    current_start: float | None = None
    for line in stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            current_start = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and current_start is not None:
            end = float(end_match.group(1))
            if end > current_start:
                silences.append(SilenceSpan(current_start, end))
            current_start = None
    return silences


def _find_first_speech_frame(
    wav_path: Path,
    start_s: float,
    end_s: float,
    threshold_db: float,
    frame_duration_s: float,
    min_consecutive_frames: int,
) -> float | None:
    if end_s <= start_s:
        return None
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        if sample_width != 2:
            raise RuntimeError(f"Speech start refinement expects 16-bit PCM WAV: {wav_path}")
        samples_per_frame = max(1, int(frame_rate * frame_duration_s))
        start_frame = max(0, int(start_s * frame_rate))
        end_frame = max(start_frame, int(end_s * frame_rate))
        wav_file.setpos(min(start_frame, wav_file.getnframes()))
        consecutive = 0
        current_frame = start_frame
        while current_frame < end_frame:
            raw = wav_file.readframes(min(samples_per_frame, end_frame - current_frame))
            if not raw:
                break
            level = _pcm16_dbfs(raw, channels)
            if level >= threshold_db:
                consecutive += 1
                if consecutive >= min_consecutive_frames:
                    onset_frame = current_frame - samples_per_frame * (min_consecutive_frames - 1)
                    return max(start_s, onset_frame / frame_rate)
            else:
                consecutive = 0
            current_frame += len(raw) // (sample_width * channels)
    return None


def _wav_frame_dbfs(wav_path: Path, frame_duration_s: float) -> list[float]:
    levels: list[float] = []
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        if sample_width != 2:
            raise RuntimeError(f"Auto silence threshold expects 16-bit PCM WAV: {wav_path}")
        samples_per_frame = max(1, int(frame_rate * frame_duration_s))
        while True:
            raw = wav_file.readframes(samples_per_frame)
            if not raw:
                break
            levels.append(_pcm16_dbfs(raw, channels))
    return levels


def _pcm16_dbfs(raw: bytes, channels: int) -> float:
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array("h", samples[::channels])
    if not samples:
        return -100.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    if mean_square <= 0:
        return -100.0
    rms = math.sqrt(mean_square)
    return 20.0 * math.log10(rms / 32768.0)


def _parse_db(value: str) -> float:
    match = re.match(r"^(-?\d+(?:\.\d+)?)dB$", value)
    if not match:
        raise ValueError(f"Invalid dB value: {value}")
    return float(match.group(1))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _format_db(value: float) -> str:
    rounded = int(round(value))
    return f"{rounded}dB"


def speech_spans_from_silences(
    duration_s: float,
    silences: list[SilenceSpan],
    min_duration_s: float,
    max_duration_s: float,
    padding_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[SilenceSpan]:
    speech_spans: list[SilenceSpan] = []
    cursor = 0.0
    for silence in silences:
        if silence.start - cursor >= min_duration_s:
            speech_spans.append(_padded_span(cursor, silence.start, duration_s, padding_s))
        cursor = max(cursor, silence.end)
    if duration_s - cursor >= min_duration_s:
        speech_spans.append(_padded_span(cursor, duration_s, duration_s, padding_s))
    return _split_merged_spans(speech_spans, max_duration_s, merge_gap_s)


def _padded_span(start: float, end: float, duration_s: float, padding_s: float) -> SilenceSpan:
    return SilenceSpan(max(0.0, start - padding_s), min(duration_s, end + padding_s))


def _split_merged_spans(
    spans: list[SilenceSpan],
    max_duration_s: float,
    merge_gap_s: float,
) -> list[SilenceSpan]:
    merged: list[SilenceSpan] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        can_merge = (
            span.start - previous.end <= merge_gap_s
            and span.end - previous.start <= max_duration_s
        )
        if can_merge:
            merged[-1] = SilenceSpan(previous.start, span.end)
        else:
            merged.append(span)

    split: list[SilenceSpan] = []
    for span in merged:
        split.extend(_split_span(span.start, span.end, max_duration_s))
    return split


def _split_span(start: float, end: float, max_duration_s: float) -> list[SilenceSpan]:
    spans: list[SilenceSpan] = []
    cursor = start
    while end - cursor > max_duration_s:
        next_end = cursor + max_duration_s
        spans.append(SilenceSpan(cursor, next_end))
        cursor = next_end
    if end > cursor:
        spans.append(SilenceSpan(cursor, end))
    return spans
