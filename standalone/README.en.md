# Kotoba Standalone

Language: [한국어](README.md) | **English**

Experimental standalone edition for Windows 11 and Linux.

This edition removes the Docker watcher and PowerShell/bash wrapper dependency. It uses a direct Python CLI so future desktop, web, or native UI layers can call the same pipeline code.

For model and third-party distribution notices, see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Easiest Windows Flow

If command lines feel intimidating, use the two Windows helper files first:

1. Double-click `install-windows.bat`.
2. After setup finishes, double-click `run-kotoba.bat`.
3. Pick a video file or folder in the small launcher window, then press `시작`.

`run-kotoba.bat` starts `kotoba-launcher`. The launcher lets you choose the input, output folder, auto silence threshold, Korean translation, and Ollama translation model from a simple window.

The command-line flow below is still useful for troubleshooting.

## Quick Setup

Install `uv`, then run from this folder:

```powershell
uv run kotoba --help
```

`ffmpeg` does not need to be installed separately. The standalone CLI uses the binary bundled by `imageio-ffmpeg`.

For transcription on NVIDIA CUDA:

```powershell
uv sync --group transcribe --group cuda
uv run --group transcribe --group cuda kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output" --auto-silence-threshold
```

To open the launcher from a terminal:

```powershell
uv run --group transcribe --group cuda kotoba-launcher
```

## Processing

Process one file:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --auto-silence-threshold
```

Process all supported media files directly inside one folder:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos" --output-dir ".\tmp-output" --auto-silence-threshold
```

VAD pre-split is enabled by default. It detects silence, cuts the WAV into speech spans, transcribes each span, and offsets timestamps back onto the original media timeline.

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
uv run --group transcribe --group cuda kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice --auto-silence-threshold
```

After translation succeeds, the selected or requested model is saved to `standalone/config/translation-defaults.json`. Later runs can omit `--model` or `--translation-model`.

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
