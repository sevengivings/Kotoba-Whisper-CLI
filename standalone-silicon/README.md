# Kotoba Standalone Silicon

언어: **한국어** | [English](README.en.md)

Apple Silicon Mac(M1 이상)에서 일본어 영상/음성 자막을 로컬로 만들기 위한 실험용 standalone 버전입니다. 기존 `standalone/` Windows/CUDA 경로와 분리해 개발합니다.

## 현재 기본 경로

- GUI ASR: `Kotoba-Whisper v2.2 MLX`
- VAD: 번들된 `pyannote/segmentation-3.0`
- GUI VAD 장치: `mps`
- 번역: Ollama가 설치되어 있으면 한국어 번역 가능
- CLI 비교 경로: Kotoba faster CPU, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX

첫 목표는 Apple Silicon에서 안정적으로 끝까지 실행되는 경로를 확보하는 것입니다. GUI는 실측상 가장 빠른 `Kotoba-Whisper v2.2 MLX + pyannote MPS` 경로로 고정하고, 다른 엔진 비교는 CLI에서 실행합니다.

## 설치

터미널에서:

```bash
cd standalone-silicon
./install-silicon.sh
```

`uv`가 없고 Homebrew가 있으면 설치 스크립트가 `brew install uv`를 시도합니다. GUI에 필요한 Tk가 빠져 있으면 `brew install python-tk@3.12`도 시도합니다. Homebrew가 없다면 먼저 uv와 Tk 지원 Python을 준비해야 합니다.

pyannote 없이 더 가벼운 FFmpeg VAD 경로만 설치하려면:

```bash
./install-silicon.sh --without-pyannote
```

기본 설치는 faster CPU, pyannote, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX 의존성과 MLX 변환 모델을 함께 준비합니다.

MPS 실험 경로를 빼고 설치하려면:

```bash
./install-silicon.sh --without-kotoba-mps
```

MLX 실험 경로를 빼고 설치하려면:

```bash
./install-silicon.sh --without-kotoba-mlx
```

MLX 의존성은 설치하되 변환 모델 준비만 건너뛰려면:

```bash
./install-silicon.sh --without-mlx-model
```

Qwen3-ASR은 의존성 충돌을 피하기 위해 기본 환경에 같이 설치하지 않습니다. 필요할 때 별도 `.venv-qwen` 환경을 사용합니다.

## GUI 실행

```bash
./run-gui.command
```

Finder에서 실행할 수도 있습니다. macOS 보안 설정에 따라 처음 실행 시 터미널에서 직접 실행하는 편이 원인 확인에 쉽습니다.

GUI 전사는 `Kotoba-Whisper v2.2 MLX + pyannote MPS` 조합으로 실행됩니다. 전사 엔진과 처리 장치 선택은 GUI에서 숨겨져 있으며, 다른 전사 엔진을 비교하려면 CLI의 `--asr-backend`와 `--model-device` 옵션을 사용하세요.

한국어 번역 옵션은 Ollama 서버와 설치된 번역 모델이 확인되기 전까지 비활성화됩니다. GUI에서 `Ollama 확인` 또는 `Ollama 모델`을 눌러 모델을 확인한 뒤 `한국어 번역까지 실행`을 켜세요.

## CLI 실행

짧은 샘플:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --output-dir ./tmp-output
```

폴더 처리:

```bash
uv run --no-sync kotoba process ~/Movies --output-dir ./tmp-output
```

MPS 실험:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba \
  --model-device mps \
  --model-dtype float32
```

FFmpeg VAD 비교:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --vad-engine ffmpeg
```

Kotoba-Whisper v2.2 MLX 실험:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba-mlx
```

변환 모델은 기본 설치 중 `models/kotoba-whisper-v2.2-mlx-q4`에 생성됩니다. 다시 만들거나 다른 위치를 쓰려면 `./tools/convert-kotoba-v22-mlx.sh [출력폴더]`를 실행하고 `--mlx-model-path`를 지정합니다.

## 번역

Ollama를 설치하고 번역 모델을 받은 뒤:

```bash
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --translate
```

## 개발 메모

이 폴더는 의도적으로 `standalone/`과 중복을 허용합니다. Apple Silicon 경로가 안정화된 뒤 `media`, `subtitle`, `translate` 같은 순수 로직부터 공통화합니다.
