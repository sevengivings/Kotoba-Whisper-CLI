from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
FFMPEG_PATH_ENV = "KOTOBA_FFMPEG_PATH"


class FFmpegAudioExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SilenceSpan:
    start: float
    end: float


@dataclass(frozen=True)
class SilenceThresholdEstimate:
    threshold_db: str
    noise_floor_db: float
    speech_level_db: float
    analyzed_frame_count: int


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def ffmpeg_exe() -> str:
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path)
        raise FileNotFoundError(f"{FFMPEG_PATH_ENV} points to a missing ffmpeg executable: {configured}")

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str | None:
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured:
        configured_path = Path(configured).expanduser()
        candidate = configured_path.with_name("ffprobe.exe" if configured_path.suffix.lower() == ".exe" else "ffprobe")
        if candidate.exists():
            return str(candidate)

    path_probe = shutil.which("ffprobe")
    if path_probe:
        return path_probe

    ffmpeg_path = Path(ffmpeg_exe())
    candidate = ffmpeg_path.with_name("ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe")
    if candidate.exists():
        return str(candidate)
    return None


def extract_audio(input_file: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = ffmpeg_exe()
    except FileNotFoundError as exc:
        raise FFmpegAudioExtractionError(_format_audio_extraction_error(input_file, str(exc), None)) from exc
    command = [
        ffmpeg,
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
    result = _run_media_command(command)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg failed without stderr"
        raise FFmpegAudioExtractionError(_format_audio_extraction_error(input_file, stderr, ffmpeg))


def extract_audio_segment(source_wav_path: Path, segment_wav_path: Path, start: float, end: float) -> None:
    segment_wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(),
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
    result = _run_media_command(command)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg segment extraction failed without stderr"
        raise RuntimeError(f"FFmpeg segment extraction failed for {source_wav_path}: {stderr}")


def probe_duration_seconds(input_file: Path) -> float | None:
    probe = ffprobe_exe()
    if probe is None:
        return None
    command = [
        probe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_file),
    ]
    result = _run_media_command(command)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise RuntimeError(f"Invalid WAV frame rate: {wav_path}")
        return wav_file.getnframes() / frame_rate


def detect_silences(wav_path: Path, threshold_db: str, min_duration_s: float) -> list[SilenceSpan]:
    command = [
        ffmpeg_exe(),
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
    result = _run_media_command(command)
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
    return normalize_speech_spans(
        duration_s,
        speech_spans,
        min_duration_s,
        max_duration_s,
        0.0,
        merge_gap_s,
    )


def normalize_speech_spans(
    duration_s: float,
    spans: list[SilenceSpan],
    min_duration_s: float,
    max_duration_s: float,
    padding_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[SilenceSpan]:
    normalized = [
        _padded_span(span.start, span.end, duration_s, padding_s)
        for span in spans
        if span.end - span.start >= min_duration_s
    ]
    return _split_merged_spans(normalized, max_duration_s, merge_gap_s)


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
    return f"{int(round(value))}dB"


def _padded_span(start: float, end: float, duration_s: float, padding_s: float) -> SilenceSpan:
    return SilenceSpan(max(0.0, start - padding_s), min(duration_s, end + padding_s))


def _split_merged_spans(spans: list[SilenceSpan], max_duration_s: float, merge_gap_s: float) -> list[SilenceSpan]:
    merged: list[SilenceSpan] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        overlaps = span.start <= previous.end
        can_merge = overlaps or (
            span.start - previous.end <= merge_gap_s and span.end - previous.start <= max_duration_s
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


def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _format_audio_extraction_error(input_file: Path, stderr: str, selected_ffmpeg: str | None) -> str:
    details = _tail_lines(stderr, 14)
    lines = [
        f"FFmpeg could not extract audio from: {input_file}",
    ]
    if selected_ffmpeg:
        lines.append(f"Selected FFmpeg: {selected_ffmpeg}")
        if _is_imageio_ffmpeg(selected_ffmpeg):
            lines.extend(
                [
                    "",
                    "The bundled FFmpeg failed while decoding this file.",
                    "This often happens with damaged AAC audio packets. Some external FFmpeg builds can skip those packets and finish.",
                    f"Install FFmpeg and add it to PATH, or set {FFMPEG_PATH_ENV} to a known-good ffmpeg.exe path.",
                    r"Example: set KOTOBA_FFMPEG_PATH=C:\Python\Faster-Whisper-XXL\ffmpeg.exe",
                    "한국어 안내: FFmpeg를 설치해서 PATH에 추가하거나, 정상 작동하는 ffmpeg.exe 경로를 KOTOBA_FFMPEG_PATH로 지정해 주세요.",
                    r"예시: set KOTOBA_FFMPEG_PATH=C:\Python\Faster-Whisper-XXL\ffmpeg.exe",
                ]
            )
    else:
        lines.extend(
            [
                "",
                f"Set {FFMPEG_PATH_ENV} to an existing ffmpeg.exe path, or install FFmpeg and add it to PATH.",
                f"한국어 안내: {FFMPEG_PATH_ENV}에 기존 ffmpeg.exe 경로를 지정하거나, FFmpeg를 설치해서 PATH에 추가해 주세요.",
            ]
        )
    lines.extend(["", "FFmpeg output:", details])
    return "\n".join(lines)


def _is_imageio_ffmpeg(ffmpeg: str) -> bool:
    return "imageio_ffmpeg" in ffmpeg.replace("\\", "/").lower()


def _tail_lines(text: str, max_lines: int) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "ffmpeg failed without stderr"
