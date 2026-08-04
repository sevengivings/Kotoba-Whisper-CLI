# Kotoba Standalone Silicon

Language: [한국어](README.md) | **English**

Experimental standalone edition for local Japanese subtitle extraction on Apple Silicon Macs (M1 or newer). This tree is intentionally separate from the Windows/CUDA-oriented `standalone/` edition.

## Current Default Path

- GUI ASR: `Kotoba-Whisper v2.2 MLX`
- VAD: bundled `pyannote/segmentation-3.0`
- GUI VAD device: `mps`
- Translation: Korean translation is available when Ollama is installed
- CLI comparison paths: Kotoba faster CPU, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX

The first goal is a reliable end-to-end Apple Silicon workflow. The GUI is fixed to the measured-fastest `Kotoba-Whisper v2.2 MLX + pyannote MPS` path, while other engines remain available from the CLI for comparison.

## Install

From Terminal:

```bash
cd standalone-silicon
./install-silicon.sh
```

If `uv` is missing and Homebrew is available, the installer tries `brew install uv`. If Tk support for the GUI is missing, it also tries `brew install python-tk@3.12`. Without Homebrew, install uv and a Python build with Tk support first.

Install only the lighter FFmpeg VAD path without pyannote:

```bash
./install-silicon.sh --without-pyannote
```

The default install prepares faster CPU, pyannote, the Kotoba Transformers MPS path, Kotoba-Whisper v2.2 MLX dependencies, and the converted MLX model.

Skip the MPS experiment path:

```bash
./install-silicon.sh --without-kotoba-mps
```

Skip the MLX experiment path:

```bash
./install-silicon.sh --without-kotoba-mlx
```

Install MLX dependencies but skip model conversion:

```bash
./install-silicon.sh --without-mlx-model
```

Qwen3-ASR is not installed into the default environment because its dependencies conflict with the standard transcription group. Use the separate `.venv-qwen` environment when needed.

## GUI

```bash
./run-gui.command
```

You can also open it from Finder. For the first run, Terminal is easier because macOS security or dependency errors remain visible.

GUI transcription runs with `Kotoba-Whisper v2.2 MLX + pyannote MPS`. Engine and processing-device selectors are hidden in the GUI. Use the CLI `--asr-backend` and `--model-device` options to compare other engines.

Korean translation controls stay disabled until the Ollama server and an installed translation model are verified. Click `Ollama Check` or `Ollama Models` in the GUI, then enable `Run Korean translation too`.

## CLI

Short sample:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --output-dir ./tmp-output
```

Folder input:

```bash
uv run --no-sync kotoba process ~/Movies --output-dir ./tmp-output
```

MPS experiment:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba \
  --model-device mps \
  --model-dtype float32
```

FFmpeg VAD comparison:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --vad-engine ffmpeg
```

Kotoba-Whisper v2.2 MLX experiment:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba-mlx
```

The default install creates the converted model at `models/kotoba-whisper-v2.2-mlx-q4`. To rebuild it or use another location, run `./tools/convert-kotoba-v22-mlx.sh [output-dir]` and pass `--mlx-model-path`.

## Translation

After installing Ollama and pulling a translation model:

```bash
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --translate
```

## Development Note

This folder intentionally tolerates duplication with `standalone/`. Once the Apple Silicon path is stable, common logic such as `media`, `subtitle`, and `translate` can be extracted.
