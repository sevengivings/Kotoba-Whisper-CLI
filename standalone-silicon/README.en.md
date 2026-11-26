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

## Updating An Existing Install

If you already use this checkout, pull the latest code and sync the Python environment again. New features can add Python dependencies, so `git pull` alone may not be enough.

```bash
cd ~/Kotoba-Whisper-CLI
git pull

cd standalone-silicon
uv sync
```

If your first install used options for `./install-silicon.sh`, you can rerun the installer with the same options when needed.

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

By default, Qwen3-ASR MLX also extracts word-level timestamps using `Qwen/Qwen3-ForcedAligner-0.6B` so subtitles split into short cues instead of long VAD spans (pass `--no-qwen-mlx-timestamps` to disable).

### Qwen Speaker-Aware Subtitles

In the GUI, selecting `Qwen3-ASR 0.6B MLX (고속실험)` or `Qwen3-ASR 1.7B MLX` enables speaker diarization by default. You can choose `자동`, `2명`, `3명`, `4명`, or `5명` for `화자 분리 인원`, and both the checkbox state and speaker count are saved. Other engines do not provide reliable word timestamps in this workflow, so GUI speaker diarization is active only for Qwen3 engines.

From the CLI, pass `--diarize-speakers` (and optionally `--num-speakers`; omit it for automatic estimation):

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend qwen3-mlx \
  --model-device mps \
  --diarize-speakers \
  --num-speakers 3 \
  --translate
```

Subtitles are split when the detected speaker changes without printing speaker labels into the SRT text. Detected speaker turns and per-subtitle speaker assignments are written to `*.speakers.json`. Accurate speaker splits require word timestamps; overlapping speech or very short interjections may still be misattributed.

Diarization uses the bundled `pyannote/segmentation-3.0` checkpoint and `pyannote/wespeaker-voxceleb-resnet34-LM`, downloading the speaker embedding model and `Qwen/Qwen3-ForcedAligner-0.6B` on first use.

## Translation

Install Ollama and pull a recommended `Hy-MT3` model for your Apple Silicon unified memory (existing `Hy-MT2` models remain supported):

| Use case | Model | Ollama size | Unified memory starting point |
| --- | --- | ---: | --- |
| Best quality & speed (32GB+ unified memory) | `hf.co/mradermacher/Hy-MT3-30B-A3B-i1-GGUF:Q4_K_M` | 18GB | 32GB+ Mac recommended (3B active MoE; unsupported on 8/16GB, tight on 24GB) |
| Balanced / default recommended (16GB+ unified memory) | `hf.co/mradermacher/Hy-MT3-7B-i1-GGUF:Q4_K_M` | 4.6GB | 16GB+ Mac recommended (~5.3GB GPU footprint; 8GB Macs may hit memory pressure) |

- **Automatic unified/GPU memory release before translation**: After ASR and optional WhisperX alignment finish, `standalone-silicon` unloads those models and clears MLX and PyTorch (MPS/CUDA) caches before invoking Ollama, freeing unified memory for one-shot translation runs.
- **Hy-MT3 speech-style prompt handling**: Because `Hy-MT3` tends to ignore `polite` / `banmal` style instructions placed only in the `system` prompt (often defaulting to casual Korean), `standalone-silicon` automatically mirrors the selected Korean speech-style instruction (`polite`, `banmal`, or `strict-banmal`) into the `user` prompt whenever an `hy-mt3` model is used.

```bash
# 32GB+ unified memory Mac (best quality & speed, 3B active MoE)
ollama pull hf.co/mradermacher/Hy-MT3-30B-A3B-i1-GGUF:Q4_K_M

# 16GB unified memory Mac (verified on Apple M1 16GB)
ollama pull hf.co/mradermacher/Hy-MT3-7B-i1-GGUF:Q4_K_M

uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --translate
```

## Development Note

This folder intentionally tolerates duplication with `standalone/`. Once the Apple Silicon path is stable, common logic such as `media`, `subtitle`, and `translate` can be extracted.
