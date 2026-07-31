from __future__ import annotations

import sys
from pathlib import Path

from kotoba_standalone.launcher import LauncherOptions, build_process_command, format_elapsed_korean


def test_format_elapsed_korean() -> None:
    assert format_elapsed_korean(0) == "0초"
    assert format_elapsed_korean(18 * 60 + 20) == "18분 20초"
    assert format_elapsed_korean(3661) == "1시간 1분 1초"


def test_build_process_command_defaults_to_process_with_auto_threshold(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(input_path=tmp_path / "sample.mp4", output_dir=tmp_path / "out")
    )

    assert command[:4] == [sys.executable, "-m", "kotoba_standalone.cli", "process"]
    assert str(tmp_path / "sample.mp4") in command
    assert "--auto-silence-threshold" in command
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
