from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> int:
    health_file = Path("/workspace/processing/.health.json")
    if not health_file.exists():
        print("health file does not exist")
        return 1
    try:
        data = json.loads(health_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid health json: {exc}")
        return 1
    age = time.time() - health_file.stat().st_mtime
    if age > 120:
        print(f"health file is stale: {age:.1f}s")
        return 1
    required = {
        "model_loaded": True,
        "gpu_available": True,
        "watcher_running": True,
    }
    for key, expected in required.items():
        if data.get(key) is not expected:
            print(f"bad health value {key}={data.get(key)!r}")
            return 1
    if data.get("status") not in {"ready"}:
        print(f"bad status: {data.get('status')!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

