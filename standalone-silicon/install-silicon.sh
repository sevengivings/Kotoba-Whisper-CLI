#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

WITH_PYANNOTE=true
WITH_KOTOBA_MPS=true
WITH_KOTOBA_MLX=true
WITH_MLX_MODEL=true
WITH_QWEN_MLX=true

for arg in "$@"; do
  case "$arg" in
    --with-pyannote)
      WITH_PYANNOTE=true
      ;;
    --without-pyannote)
      WITH_PYANNOTE=false
      ;;
    --with-kotoba-mps)
      WITH_KOTOBA_MPS=true
      WITH_PYANNOTE=true
      ;;
    --without-kotoba-mps)
      WITH_KOTOBA_MPS=false
      ;;
    --with-kotoba-mlx)
      WITH_KOTOBA_MLX=true
      ;;
    --without-kotoba-mlx)
      WITH_KOTOBA_MLX=false
      WITH_MLX_MODEL=false
      ;;
    --with-mlx-model)
      WITH_KOTOBA_MLX=true
      WITH_MLX_MODEL=true
      ;;
    --without-mlx-model)
      WITH_MLX_MODEL=false
      ;;
    --with-qwen-mlx)
      WITH_QWEN_MLX=true
      ;;
    --without-qwen-mlx)
      WITH_QWEN_MLX=false
      ;;
    -h|--help)
      echo "Usage: ./install-silicon.sh [--without-pyannote] [--without-kotoba-mps] [--without-kotoba-mlx] [--without-mlx-model] [--without-qwen-mlx]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is intended for macOS on Apple Silicon."
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: this Mac is not reporting arm64. Apple Silicon is recommended."
fi

if ! command -v uv >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing uv with Homebrew..."
    brew install uv
  else
    echo "uv was not found. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

if ! python3.12 -c "import tkinter" >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    if ! brew list --versions python-tk@3.12 >/dev/null 2>&1; then
      echo "Installing Tk support for Python 3.12 with Homebrew..."
      brew install python-tk@3.12
    fi
  fi
fi

if ! python3.12 -c "import tkinter" >/dev/null 2>&1; then
  echo "Python 3.12 Tk support is not available."
  echo "Install it with: brew install python-tk@3.12"
  exit 1
fi

groups=(--group faster)
if [[ "$WITH_KOTOBA_MPS" == "true" ]]; then
  groups+=(--group transcribe --group torch)
fi
if [[ "$WITH_PYANNOTE" == "true" ]]; then
  if [[ "$WITH_KOTOBA_MPS" != "true" ]]; then
    groups+=(--group transcribe --group torch)
  fi
  groups+=(--group pyannote)
fi
if [[ "$WITH_KOTOBA_MLX" == "true" ]]; then
  groups+=(--group mlx)
fi
if [[ "$WITH_QWEN_MLX" == "true" ]]; then
  groups+=(--group qwen-mlx)
fi

echo "Installing Kotoba Standalone Silicon packages..."
uv sync --python 3.12 "${groups[@]}"

if [[ "$WITH_KOTOBA_MLX" == "true" && "$WITH_MLX_MODEL" == "true" ]]; then
  echo
  echo "Preparing Kotoba-Whisper v2.2 MLX model..."
  "$SCRIPT_DIR/tools/convert-kotoba-v22-mlx.sh"
fi

echo
echo "Setup completed."
echo "Start the launcher with: ./run-gui.command"
echo "CLI sample: uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --output-dir ./tmp-output"
