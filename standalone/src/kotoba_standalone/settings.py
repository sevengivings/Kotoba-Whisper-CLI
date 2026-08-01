from __future__ import annotations

import json
from pathlib import Path


DEFAULT_TRANSLATION_MODEL = "hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M"
DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "config" / "translation-defaults.json"
LAUNCHER_STATE_PATH = Path(__file__).resolve().parents[2] / "config" / "launcher-state.json"


def load_saved_translation_model(path: Path = DEFAULTS_PATH) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    model = data.get("ollama_model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def save_translation_model(model: str, path: Path = DEFAULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ollama_model": model}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_launcher_state(path: Path = LAUNCHER_STATE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_launcher_state(state: dict, path: Path = LAUNCHER_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
