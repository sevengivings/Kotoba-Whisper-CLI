from __future__ import annotations

from pathlib import Path

import pytest

import kotoba_standalone.media as media
from kotoba_standalone.media import (
    FFmpegAudioExtractionError,
    SilenceSpan,
    extract_audio,
    normalize_speech_spans,
    parse_silencedetect_output,
    speech_spans_from_silences,
)


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


def test_normalize_speech_spans_filters_pads_merges_and_splits() -> None:
    spans = normalize_speech_spans(
        duration_s=10.0,
        spans=[SilenceSpan(0.1, 0.2), SilenceSpan(1.0, 2.0), SilenceSpan(2.3, 4.0)],
        min_duration_s=0.25,
        max_duration_s=2.0,
        padding_s=0.2,
        merge_gap_s=0.0,
    )

    assert spans == [
        SilenceSpan(0.8, 2.8),
        SilenceSpan(2.8, 4.2),
    ]


def test_ffmpeg_exe_prefers_configured_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = tmp_path / "custom-ffmpeg.exe"
    configured.write_text("", encoding="utf-8")
    monkeypatch.setenv(media.FFMPEG_PATH_ENV, str(configured))
    monkeypatch.setattr(media.shutil, "which", lambda name: None)

    assert media.ffmpeg_exe() == str(configured)


def test_ffmpeg_exe_rejects_missing_configured_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(media.FFMPEG_PATH_ENV, r"C:\missing\ffmpeg.exe")

    with pytest.raises(FileNotFoundError):
        media.ffmpeg_exe()


def test_ffmpeg_exe_prefers_path_before_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(media.FFMPEG_PATH_ENV, raising=False)
    monkeypatch.setattr(media.shutil, "which", lambda name: r"C:\tools\ffmpeg.exe" if name == "ffmpeg" else None)
    monkeypatch.setattr(media.imageio_ffmpeg, "get_ffmpeg_exe", lambda: r"C:\bundled\ffmpeg.exe")

    assert media.ffmpeg_exe() == r"C:\tools\ffmpeg.exe"


def test_ffprobe_exe_uses_path_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda name: r"C:\tools\ffprobe.exe" if name == "ffprobe" else None)

    assert media.ffprobe_exe() == r"C:\tools\ffprobe.exe"


def test_ffprobe_exe_prefers_configured_ffmpeg_sibling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_text("", encoding="utf-8")
    ffprobe.write_text("", encoding="utf-8")
    monkeypatch.setenv(media.FFMPEG_PATH_ENV, str(ffmpeg))
    monkeypatch.setattr(media.shutil, "which", lambda name: r"C:\path\ffprobe.exe" if name == "ffprobe" else None)

    assert media.ffprobe_exe() == str(ffprobe)


def test_extract_audio_error_suggests_external_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "broken.mp4"
    wav_path = tmp_path / "broken.wav"
    input_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: r"C:\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg.exe")

    class Result:
        returncode = 1
        stderr = "aac bitstream error\nInvalid data found when processing input"

    monkeypatch.setattr(media, "_run_media_command", lambda command: Result())

    with pytest.raises(FFmpegAudioExtractionError) as exc_info:
        extract_audio(input_file, wav_path)

    message = str(exc_info.value)
    assert "bundled FFmpeg failed" in message
    assert media.FFMPEG_PATH_ENV in message
    assert "한국어 안내" in message
    assert "Invalid data found" in message
