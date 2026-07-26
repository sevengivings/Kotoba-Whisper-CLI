from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


@dataclass(frozen=True)
class SilenceSpan:
    start: float
    end: float


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
