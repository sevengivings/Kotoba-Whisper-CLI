from __future__ import annotations

from pathlib import Path

from kotoba_standalone.settings import load_saved_translation_model, save_translation_model


def test_save_and_load_translation_model(tmp_path: Path) -> None:
    path = tmp_path / "translation-defaults.json"

    save_translation_model("chosen:model", path)

    assert load_saved_translation_model(path) == "chosen:model"


def test_load_saved_translation_model_ignores_missing_file(tmp_path: Path) -> None:
    assert load_saved_translation_model(tmp_path / "missing.json") is None
