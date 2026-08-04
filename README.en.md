# Kotoba-Whisper CLI

Language: [한국어](README.md) | **English**

A local tool that extracts Japanese subtitles from media and can translate them into Korean with Ollama. New installations should use the **Docker-free `standalone/` edition**.

## Easiest Windows Setup

1. Install the NVIDIA graphics driver if you want CUDA acceleration.
2. Double-click `standalone\install-kotoba.bat`.
3. If prompted that Python 3.12 is missing, confirm the install with `Y`.
4. Double-click `standalone\run-gui.bat` after setup finishes.
5. Select a media file or folder and press `시작`.

The setup batch file checks for `uv` and Python 3.12 and installs them with `winget` when needed. It asks before installing Python 3.12. The application uses a bundled ffmpeg binary. If CUDA is available, the launcher shows the CUDA device; otherwise it shows CPU only. Korean translation additionally requires [Ollama](https://ollama.com/download).

See the [standalone English guide](standalone/README.en.md) for detailed setup and usage.

## Command-Line Quick Start

```powershell
cd C:\Python\Kotoba-Whisper-CLI\standalone
uv sync --group transcribe --group cuda --group pyannote
uv run --no-sync kotoba process "D:\Videos\sample.mp4"
```

Speech detection uses the bundled pyannote VAD by default. The MIT-licensed model is included with both editions, so users do not need a Hugging Face account or token.

The upstream license, model card, revision, and checksums are preserved in the [standalone model directory](standalone/src/kotoba_standalone/models/pyannote-segmentation-3.0) and [Docker model directory](docker/app/models/pyannote-segmentation-3.0).

## Editions

| Edition | Recommended for | Description |
| --- | --- | --- |
| [Standalone](standalone/README.en.md) | New users | Direct uv workflow with Windows GUI and CLI |
| [Docker](docker/README.md) | Existing watcher users | Pyannote-default `input` folder watcher workflow |

Docker code, scripts, configuration, data directories, and model cache all live in `docker/`.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model and dependency licensing and redistribution notes.
