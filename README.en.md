# Kotoba-Whisper CLI

Language: [한국어](README.md) | **English**

This project provides Japanese subtitle extraction and Korean subtitle translation workflows based on Kotoba-Whisper and Ollama.

There are two editions:

- Docker edition: the original folder-watcher workflow with PowerShell/bash helper scripts.
- Standalone edition: an experimental beginner-friendly `uv` based CLI without Docker or PowerShell wrapper dependency.

For new users who mainly want to process local media files directly, start with the standalone edition:

- [Standalone Korean guide](standalone/README.md)
- [Standalone English guide](standalone/README.en.md)

The recommended translation models are:

- Best quality: `hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M`
- Minimum recommended: `hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M`

