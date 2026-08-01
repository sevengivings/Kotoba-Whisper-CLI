from __future__ import annotations

import sys
from pathlib import Path

from kotoba_standalone.launcher import (
    LauncherOptions,
    LauncherTranslationOptions,
    build_process_command,
    build_translate_command,
    existing_korean_subtitles,
    find_existing_japanese_subtitles,
    format_elapsed_korean,
    media_filetypes,
    validate_input_path,
)


def test_format_elapsed_korean() -> None:
    assert format_elapsed_korean(0) == "0초"
    assert format_elapsed_korean(18 * 60 + 20) == "18분 20초"
    assert format_elapsed_korean(3661) == "1시간 1분 1초"


def test_build_process_command_defaults_to_pyannote(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(input_path=tmp_path / "sample.mp4", output_dir=tmp_path / "out")
    )

    assert command[:4] == [sys.executable, "-m", "kotoba_standalone.cli", "process"]
    assert str(tmp_path / "sample.mp4") in command
    assert "--auto-silence-threshold" not in command
    assert command[command.index("--vad-engine") + 1] == "pyannote"
    assert "--translate" not in command


def test_build_process_command_adds_translation_options(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
            translate=True,
            translation_model="chosen:model",
            korean_style="strict-banmal",
        )
    )

    assert "--translate" in command
    assert command[command.index("--translation-model") + 1] == "chosen:model"
    assert command[command.index("--korean-style") + 1] == "strict-banmal"


def test_build_translate_command_writes_to_output_dir(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.ja.srt"
    command = build_translate_command(
        srt_path,
        LauncherTranslationOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
            translation_model="chosen:model",
            korean_style="banmal",
        ),
    )

    assert command[:4] == [sys.executable, "-m", "kotoba_standalone.cli", "translate"]
    assert str(srt_path) in command
    assert command[command.index("--output") + 1] == str(tmp_path / "out")
    assert command[command.index("--model") + 1] == "chosen:model"
    assert command[command.index("--korean-style") + 1] == "banmal"


def test_build_process_command_does_not_expose_ffmpeg_vad_options(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
        )
    )

    assert command[command.index("--vad-engine") + 1] == "pyannote"
    assert "--auto-silence-threshold" not in command
    assert "--report-subtitle-quality" not in command
    assert "--drop-likely-hallucinations" not in command
    assert "--split-long-subtitles" not in command
    assert "--annotate-subtitle-quality" not in command
    assert "--tail-retranscribe-long-subtitles" not in command


def test_media_filetypes_include_video_and_audio_patterns() -> None:
    filetypes = media_filetypes()

    assert "*.mp4" in filetypes[0][1]
    assert "*.wav" in filetypes[0][1]


def test_validate_input_path_accepts_supported_media_file(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_text("", encoding="utf-8")

    assert validate_input_path(media) is None


def test_validate_input_path_rejects_non_media_file(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("", encoding="utf-8")

    assert validate_input_path(document) == "동영상 또는 음성 파일만 선택할 수 있습니다."


def test_validate_input_path_rejects_folder_without_media(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("", encoding="utf-8")

    assert validate_input_path(tmp_path) == "선택한 폴더에 처리 가능한 동영상 또는 음성 파일이 없습니다."


def test_find_existing_japanese_subtitles_for_media_file(tmp_path: Path) -> None:
    media = tmp_path / "media" / "sample.mp4"
    output_dir = tmp_path / "out"
    media.parent.mkdir()
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    expected = output_dir / "sample.ja.srt"
    expected.write_text("", encoding="utf-8")

    assert find_existing_japanese_subtitles(media, output_dir) == [expected]


def test_find_existing_japanese_subtitles_for_folder_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    first = output_dir / "a.ja.srt"
    second = output_dir / "b.ja.srt"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    (output_dir / "b.ko.srt").write_text("", encoding="utf-8")

    assert find_existing_japanese_subtitles(input_dir, output_dir) == [first, second]


def test_existing_korean_subtitles_reports_existing_outputs(tmp_path: Path) -> None:
    input_srt = tmp_path / "sample.ja.srt"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "sample.ko.srt"
    existing.write_text("", encoding="utf-8")

    assert existing_korean_subtitles([input_srt], output_dir) == [existing]
