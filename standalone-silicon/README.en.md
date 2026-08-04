# Kotoba Standalone Silicon

Language: [한국어](README.md) | **English**

Experimental standalone edition for local Japanese subtitle extraction on Apple Silicon Macs (M1 or newer). This tree is intentionally separate from the Windows/CUDA-oriented `standalone/` edition.

## Current Default Path

- GUI ASR: `Kotoba-Whisper v2.2 MLX`
- VAD: bundled `pyannote/segmentation-3.0`
- GUI VAD device: `mps`
- Translation: Korean translation is available when Ollama is installed
- GUI/CLI comparison paths: Kotoba faster CPU, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX, Qwen3-ASR MLX

The first goal is a reliable end-to-end Apple Silicon workflow. The GUI defaults to the stable `Kotoba-Whisper v2.2 MLX + pyannote MPS` path, and also lets you choose Qwen3-ASR 0.6B/1.7B MLX from the transcription engine selector.

## Install

From Terminal:

```bash
cd standalone-silicon
./install-silicon.sh
```

Installation needs network access for Python packages, Kotoba MLX model conversion, and model downloads. The Kotoba MLX conversion step also needs `git` because it fetches `mlx-examples`.

If `uv` is missing and Homebrew is available, the installer tries `brew install uv`. If Tk support for the GUI is missing, it also tries `brew install python-tk@3.12`. Without Homebrew, install uv, git, and a Python 3.12 build with Tk support first.

Install only the lighter FFmpeg VAD path without pyannote:

```bash
./install-silicon.sh --without-pyannote
```

The default install prepares faster CPU, pyannote, the Kotoba Transformers MPS path, Kotoba-Whisper v2.2 MLX, Qwen3-ASR MLX dependencies, and the converted Kotoba MLX model.

The clean-install path has been verified with GUI startup, Kotoba MLX transcription, and Qwen3-ASR 0.6B MLX transcription.

Skip the MPS experiment path:

```bash
./install-silicon.sh --without-kotoba-mps
```

Skip the Kotoba MLX path:

```bash
./install-silicon.sh --without-kotoba-mlx
```

Install MLX dependencies but skip model conversion:

```bash
./install-silicon.sh --without-mlx-model
```

Skip Qwen3-ASR MLX dependencies:

```bash
./install-silicon.sh --without-qwen-mlx
```

Qwen3-ASR MLX Python dependencies are included in the default install. The Qwen3-ASR 0.6B/1.7B model weights are not downloaded during installation; they are downloaded into the Hugging Face cache on first use from the GUI or CLI.

## GUI

```bash
./run-gui.command
```

You can also open it from Finder. For the first run, Terminal is easier because macOS security or dependency errors remain visible.

GUI transcription defaults to `Kotoba-Whisper v2.2 MLX + pyannote MPS`. The transcription engine selector includes `Qwen3-ASR 0.6B MLX (고속실험)` and `Qwen3-ASR 1.7B MLX`. The processing-device selector remains hidden; pyannote uses MPS when available.

The first Qwen3-ASR MLX run can take longer while the selected model is downloaded. Later runs reuse the Hugging Face cache.

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

Kotoba-Whisper v2.2 MLX:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba-mlx
```

The default install creates the converted model at `models/kotoba-whisper-v2.2-mlx-q4`. To rebuild it or use another location, run `./tools/convert-kotoba-v22-mlx.sh [output-dir]` and pass `--mlx-model-path`.

Qwen3-ASR 1.7B MLX experiment:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend qwen3-mlx
```

Use the smaller 0.6B model first when you want a faster smoke test:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend qwen3-mlx \
  --qwen-mlx-model-name Qwen/Qwen3-ASR-0.6B
```

## Translation

After installing Ollama and pulling a translation model:

```bash
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --translate
```

## Development Note

This folder intentionally tolerates duplication with `standalone/`. Once the Apple Silicon path is stable, common logic such as `media`, `subtitle`, and `translate` can be extracted.
