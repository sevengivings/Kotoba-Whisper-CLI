from __future__ import annotations

from pathlib import Path

from kotoba_standalone.settings import (
    load_launcher_state,
    load_saved_translation_model,
    save_launcher_state,
    save_translation_model,
)


def test_save_and_load_translation_model(tmp_path: Path) -> None:
    path = tmp_path / "translation-defaults.json"

    save_translation_model("chosen:model", path)

    assert load_saved_translation_model(path) == "chosen:model"


def test_load_saved_translation_model_ignores_missing_file(tmp_path: Path) -> None:
    assert load_saved_translation_model(tmp_path / "missing.json") is None


def test_save_and_load_launcher_state(tmp_path: Path) -> None:
    path = tmp_path / "launcher-state.json"
    state = {
        "last_output_dir": "D:/Output",
        "last_model_device": "cuda:0",
        "external_ffmpeg_path": "C:/Tools/ffmpeg.exe",
        "ollama_host": "ollama.local",
        "ollama_port": 11435,
    }

    save_launcher_state(state, path)

    assert load_launcher_state(path) == state
