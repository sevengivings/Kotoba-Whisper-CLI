# Kotoba Standalone

Experimental standalone edition for Windows 11 and Linux.

This edition is intended to remove the Docker watcher and PowerShell/bash wrapper dependency. It uses a direct Python CLI so future desktop, web, or native UI layers can call the same pipeline code.

Current status:

- `kotoba translate`: usable Ollama SRT translation command.
- `kotoba process`: direct media input, bundled ffmpeg audio extraction, lazy Kotoba model loading, and Japanese SRT/TXT/JSON writing when transcription dependencies are installed.
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

For translation:

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
uv run kotoba translate "C:\Python\Kotoba-Whisper-CLI\output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
```

Run tests:

```powershell
uv run --group dev pytest
```

## Design

The CLI is intentionally thin. UI layers should call `kotoba_standalone.pipeline.process_video` or `kotoba_standalone.translate.ollama.translate_srt` and receive progress through callbacks.
