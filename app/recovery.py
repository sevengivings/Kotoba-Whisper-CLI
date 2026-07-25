from __future__ import annotations

import json
from pathlib import Path


def read_attempts(state_file: Path) -> dict[str, int]:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_attempts(state_file: Path, attempts: dict[str, int]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp = state_file.with_suffix(state_file.suffix + ".part")
    temp.write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(state_file)


def increment_attempt(state_file: Path, key: str) -> int:
    attempts = read_attempts(state_file)
    attempts[key] = attempts.get(key, 0) + 1
    write_attempts(state_file, attempts)
    return attempts[key]

