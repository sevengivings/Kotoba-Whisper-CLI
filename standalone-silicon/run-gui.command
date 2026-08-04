#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [[ -x "$SCRIPT_DIR/.venv/bin/kotoba-launcher" ]]; then
  exec "$SCRIPT_DIR/.venv/bin/kotoba-launcher"
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/kotoba-launcher" ]]; then
  exec "$VIRTUAL_ENV/bin/kotoba-launcher"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --no-sync kotoba-launcher
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  exec "$SCRIPT_DIR/.venv/bin/python" -m kotoba_standalone.launcher
fi

if command -v python >/dev/null 2>&1 && python -c "import kotoba_standalone.launcher" >/dev/null 2>&1; then
  exec python -m kotoba_standalone.launcher
fi

echo "Kotoba Standalone Silicon environment was not found."
echo "Run ./install-silicon.sh first, then try again."
read -r -p "Press Enter to close."
exit 1
