# Kotoba-Whisper CLI

Language: [한국어](README.md) | **English**

A local tool that extracts Japanese subtitles from media and can translate them into Korean with Ollama. New installations should use the **Docker-free `standalone/` edition**.

## Easiest Windows Setup

1. Install the NVIDIA graphics driver.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Double-click `standalone\install-windows.bat`.
4. Double-click `standalone\run-kotoba.bat` after setup finishes.
5. Select a media file or folder and press `시작`.

Python and ffmpeg do not need separate installations. uv manages Python, and the application uses a bundled ffmpeg binary. The NVIDIA driver is still required. Korean translation additionally requires [Ollama](https://ollama.com/download).

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
