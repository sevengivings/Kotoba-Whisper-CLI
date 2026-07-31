# Kotoba Standalone

Language: [한국어](README.md) | **English**

Recommended Docker-free edition for Windows 11 and Linux.

This edition removes the Docker watcher and PowerShell/bash wrapper dependency. It uses a direct Python CLI so future desktop, web, or native UI layers can call the same pipeline code.

For model and third-party distribution notices, see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Easiest Windows Flow

If command lines feel intimidating, use the two Windows helper files first:

1. Double-click `install-windows.bat`.
2. After setup finishes, double-click `run-kotoba.bat`.
3. Pick a video file or folder in the small launcher window, then press `시작`.

`run-kotoba.bat` starts `kotoba-launcher`. The launcher lets you choose the input, output folder, Korean translation, and Ollama translation model from a simple window. Speech detection automatically uses the bundled pyannote model. Experimental subtitle quality post-processing remains CLI-only.

The command-line flow below is still useful for troubleshooting.

## Quick Setup

Install `uv`, then run from this folder:

```powershell
uv run kotoba --help
```

`ffmpeg` does not need to be installed separately. The standalone CLI uses the binary bundled by `imageio-ffmpeg`.

For transcription on NVIDIA CUDA:

```powershell
uv sync --group transcribe --group cuda --group pyannote
uv run --no-sync kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output"
```

To open the launcher from a terminal:

```powershell
uv run --group transcribe --group cuda kotoba-launcher
```

## Processing

Process one file:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output"
```

Process all supported media files directly inside one folder:

```powershell
uv run --no-sync kotoba process "D:\Videos" --output-dir ".\tmp-output"
```

Pyannote VAD pre-split is enabled by default. It detects human speech spans, transcribes them, and offsets timestamps back onto the original media timeline. The GUI does not expose the legacy FFmpeg volume controls. Advanced CLI users can compare the old engine with `--vad-engine ffmpeg` and optionally add `--auto-silence-threshold` or `--silence-threshold-db -42dB`.

### Default pyannote VAD

`install-windows.bat` installs the pyannote dependencies. To add them to an existing environment manually:

```powershell
uv sync --group transcribe --group cuda --group pyannote
```

The original `pyannote/segmentation-3.0` weights are bundled under the MIT license. The default model therefore needs no Hugging Face account, condition acceptance, token, or network download. The upstream `LICENSE`, model card, fixed revision `e66f3d3b9eb0873085418a7b813d3b369bf160bb`, and checksums are preserved in `src/kotoba_standalone/models/pyannote-segmentation-3.0`. Existing Windows installations can run `install-pyannote-windows.bat` to add the required Python dependencies.

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output"
```

The launcher always uses pyannote. The pyannote model is released from GPU memory before Kotoba is loaded. `--auto-silence-threshold` and `--silence-threshold-db` are ignored when pyannote is selected. Processing metadata records the selected engine, pyannote version, model, `bundled/local/huggingface` source, and detected speech span count. A `*.vad.json` file stores both the raw detected spans and the normalized spans sent to Kotoba. A custom remote gated model passed with `--pyannote-model` still requires the user's own access and token.

## Experimental Subtitle Quality Options (CLI Only)

Difficult media can make ASR create hallucinated subtitles from silence, repeated low-level sounds, or zero-duration chunks. These options are experimental and are off by default.

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --report-subtitle-quality --drop-likely-hallucinations --split-long-subtitles
```

- `--report-subtitle-quality`: writes a `*.subtitle-quality.json` report.
- `--drop-likely-hallucinations`: removes only high-confidence hallucination subtitles.
- `--split-long-subtitles`: splits long real subtitles on text boundaries.
- `--tail-retranscribe-long-subtitles`: retranscribes the last 5 seconds of suspicious 10-30 second subtitles and moves the start time forward when the tail text matches.
- `--tail-retranscribe-max-candidates 20`: limits tail retranscription attempts, prioritizing re-segmentation candidates and longer candidates.
- `--annotate-subtitle-quality`: appends quality tags as extra subtitle text lines.

The automatic drop pass is intentionally conservative:

- standalone `ごめん`, `ありがとうございました`, `ありがとうございます`, `ご視聴ありがとうございました`
- punctuation-only subtitles
- near-zero duration subtitles with long text

The report also marks review-only candidates such as `drop_candidate`, `refine_start_candidate`, `resegment_candidate`, and `split_candidate`.

`--tail-retranscribe-long-subtitles` only tries `drop_candidate` and `resegment_candidate` subtitles between 10 and 30 seconds. It records every attempt in `tail_refine_attempts` and applies a timing change only when the tail transcription is similar enough to the original subtitle. By default, it checks up to 20 candidates, prioritizing re-segmentation candidates and then longer candidates.

To inspect quality tags directly in Subtitle Edit or a video player:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --report-subtitle-quality --annotate-subtitle-quality
```

`--annotate-subtitle-quality` writes `[quality: ...]` lines into the SRT body, so use it only for review. If translation is enabled, those tags may become part of the translation input.

## Translation

Recommended Hy-MT2 Ollama models:

| Use case | Model | Ollama size | Practical VRAM starting point |
| --- | --- | ---: | --- |
| Best quality | `hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M` | 18GB | 24GB class GPU recommended |
| Minimum recommended | `hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M` | 4.6GB | 8GB class GPU minimum; 12GB+ is safer |

Pull the best-quality model:

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
```

Pull the minimum recommended model:

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

Translate an existing Japanese SRT:

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

Choose from models already downloaded in Ollama:

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model-choice
```

Process and translate in one command:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice
```

After translation succeeds, the selected or requested model is saved to `config/translation-defaults.json`. Later runs can omit `--model` or `--translation-model`.

When `kotoba process --translate` succeeds, the Korean subtitle is also copied next to the original media file. If `VideoName.srt` does not exist, it uses that name. If it already exists, it writes `VideoName.ko.srt` instead.

## Model Smoke Test

Before translating a long SRT:

```powershell
python .\tools\compare-ollama-models.py --timeout-seconds 30
```

On the current test machine, 30B Q4 produced the best Korean, and 7B Q4 was a usable faster middle-spec fallback. The 1.8B Q4 model is intentionally not recommended for this subtitle prompt.

## Tests

```powershell
uv run --group dev pytest
```
