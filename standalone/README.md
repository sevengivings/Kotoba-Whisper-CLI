# Kotoba Standalone

Experimental standalone edition for Windows 11 and Linux.

This edition is intended to remove the Docker watcher and PowerShell/bash wrapper dependency. It uses a direct Python CLI so future desktop, web, or native UI layers can call the same pipeline code.

Current status:

- `kotoba translate`: usable Ollama SRT translation command.
- `kotoba process`: direct media input, bundled ffmpeg audio extraction, VAD pre-split, lazy Kotoba model loading, and Japanese SRT/TXT/JSON writing when transcription dependencies are installed.
- Verified on Windows 11 + RTX 3090 with `torch==2.5.1+cu121` using `sample/ja_short_test.mp4`.
- Translation initially reuses the same Ollama prompt and batch behavior as the Docker edition.

## Setup

Install `uv`, then from this folder:

```powershell
uv run kotoba --help
```

`ffmpeg` does not need to be installed separately for the standalone CLI. It uses the ffmpeg binary bundled by `imageio-ffmpeg`.

Transcription dependencies are intentionally optional because CUDA PyTorch is large and hardware-specific. For now, install them only on a Windows/Linux machine with an NVIDIA GPU. The default standalone CUDA group follows the Docker edition's PyTorch/CUDA line (`torch==2.5.1`, CUDA 12.1 wheels):

```powershell
uv sync --group transcribe --group cuda
uv run --group transcribe --group cuda kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output"
```

If PyTorch/CUDA is missing, `kotoba process` still extracts WAV audio and returns `missing_transcription_dependency` with a clear message. Use `nvidia-smi` first to confirm that Windows or Linux can see the NVIDIA GPU.

Try direct media processing skeleton:

```powershell
uv run kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output"
```

You can also pass a folder. By default this processes supported media files directly inside that folder:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos" --output-dir ".\tmp-output" --auto-silence-threshold
```

VAD pre-split is enabled by default. It detects silence with the default `-42dB` threshold, cuts the WAV into speech spans, transcribes each span, and offsets timestamps back onto the original media timeline. You can ask the CLI to estimate the threshold from the full audio first:

```powershell
uv run --group transcribe --group cuda kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output" --auto-silence-threshold
```

Useful tuning flags:

- `--silence-threshold-db -42dB`: override the silence detector threshold.
- `--auto-silence-threshold`: analyze the WAV and choose a threshold before silence detection.
- `--no-vad-pre-split`: transcribe the whole WAV without VAD splitting.
- `--vad-max-segment-duration-s 30`: cap each speech segment length.
- `--vad-padding-s 0.4`: keep a little audio before and after each speech span.

For translation:

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
uv run kotoba translate "C:\Python\Kotoba-Whisper-CLI\output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
```

Recommended Hy-MT2 Ollama models:

| Use case | Model | Ollama size | Practical VRAM starting point |
| --- | --- | ---: | --- |
| Best quality | `hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M` | 18GB | 24GB class GPU recommended |
| Minimum recommended | `hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M` | 4.6GB | 8GB class GPU minimum; 12GB+ is safer |

The Ollama size is the downloaded model size, not an exact VRAM requirement. Runtime memory also depends on context length, KV cache, GPU offload behavior, and other running programs. For this subtitle prompt, `Hy-MT2-7B-GGUF:Q4_K_M` is the lowest model currently recommended. If it fails to load or the machine starts paging heavily, use another lower-quant 7B variant through Ollama and select it with `--model-choice`.

Middle-spec setup example:

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

Translation checks `http://localhost:11434/api/tags` before sending subtitle text. If Ollama is not running, the CLI exits with a short human-readable failure message instead of a Python traceback.

To choose from models already downloaded in Ollama:

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model-choice
```

For `process --translate`, use:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice
```

After a translation succeeds, the selected or requested model is saved to `standalone/config/translation-defaults.json`. Later runs can omit `--model` or `--translation-model`; the saved model is reused first, with the built-in default model used only when no saved model exists.

When `kotoba process --translate` succeeds, the Korean subtitle is also copied next to the original media file. If `VideoName.srt` does not exist, the copied subtitle uses that name. If it already exists, the CLI writes `VideoName.ko.srt` instead.

For a quick model sanity check before translating a long SRT:

```powershell
python .\tools\compare-ollama-models.py --timeout-seconds 30
```

On the current test machine, `Hy-MT2-30B-A3B-GGUF:Q4_K_M` produced the best Korean, and `Hy-MT2-7B-GGUF:Q4_K_M` was a usable faster middle-spec fallback. `Hy-MT2-1.8B-GGUF:Q4_K_M` is intentionally not recommended because it produced malformed Korean for this subtitle prompt.

Run tests:

```powershell
uv run --group dev pytest
```

## Design

The CLI is intentionally thin. UI layers should call `kotoba_standalone.pipeline.process_video` or `kotoba_standalone.translate.ollama.translate_srt` and receive progress through callbacks.
