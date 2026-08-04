#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="${MLX_EXAMPLES_DIR:-}"
OUTPUT_DIR="${1:-$ROOT_DIR/models/kotoba-whisper-v2.2-mlx-q4}"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Run ./install-silicon.sh first."
  exit 1
fi

if [[ -e "$OUTPUT_DIR/weights.safetensors" || -e "$OUTPUT_DIR/weights.npz" ]]; then
  echo "Converted model already exists: $OUTPUT_DIR"
  exit 0
fi

uv sync --inexact --group faster --group transcribe --group torch --group pyannote --group mlx

if [[ -z "$EXAMPLES_DIR" ]]; then
  EXAMPLES_DIR="$(mktemp -d /tmp/mlx-examples.XXXXXX)"
  git clone --depth 1 https://github.com/ml-explore/mlx-examples.git "$EXAMPLES_DIR"
fi

uv run --no-sync python "$EXAMPLES_DIR/whisper/convert.py" \
  --torch-name-or-path kotoba-tech/kotoba-whisper-v2.2 \
  --mlx-path "$OUTPUT_DIR" \
  -q \
  --q-bits 4

if [[ -f "$OUTPUT_DIR/model.safetensors" && ! -e "$OUTPUT_DIR/weights.safetensors" ]]; then
  ln -s model.safetensors "$OUTPUT_DIR/weights.safetensors" 2>/dev/null || cp "$OUTPUT_DIR/model.safetensors" "$OUTPUT_DIR/weights.safetensors"
fi

echo "Converted model: $OUTPUT_DIR"
