# Kotoba Standalone Silicon

언어: **한국어** | [English](README.en.md)

Apple Silicon Mac(M1 이상)에서 일본어 영상/음성 자막을 로컬로 만들기 위한 실험용 standalone 버전입니다. 기존 `standalone/` Windows/CUDA 경로와 분리해 개발합니다.

## 현재 기본 경로

- GUI ASR: `Kotoba-Whisper v2.2 MLX`
- VAD: 번들된 `pyannote/segmentation-3.0`
- GUI VAD 장치: `mps`
- 번역: Ollama가 설치되어 있으면 한국어 번역 가능
- GUI/CLI 비교 경로: Kotoba faster CPU, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX, Qwen3-ASR MLX

첫 목표는 Apple Silicon에서 안정적으로 끝까지 실행되는 경로를 확보하는 것입니다. GUI 기본값은 실측상 안정적인 `Kotoba-Whisper v2.2 MLX + pyannote MPS` 경로이며, Qwen3-ASR 0.6B/1.7B MLX도 전사 엔진에서 선택할 수 있습니다.

## 설치

터미널에서:

```bash
cd standalone-silicon
./install-silicon.sh
```

설치 중 Python 패키지 설치, Kotoba MLX 모델 변환, 필요한 모델 다운로드가 진행되므로 네트워크 연결이 필요합니다. Kotoba MLX 변환 단계에서는 `mlx-examples`를 내려받기 위해 `git`도 필요합니다.

`uv`가 없고 Homebrew가 있으면 설치 스크립트가 `brew install uv`를 시도합니다. GUI에 필요한 Tk가 빠져 있으면 `brew install python-tk@3.12`도 시도합니다. Homebrew가 없다면 먼저 uv, git, Tk 지원 Python 3.12를 준비해야 합니다.

pyannote 없이 더 가벼운 FFmpeg VAD 경로만 설치하려면:

```bash
./install-silicon.sh --without-pyannote
```

기본 설치는 faster CPU, pyannote, Kotoba Transformers MPS, Kotoba-Whisper v2.2 MLX, Qwen3-ASR MLX 의존성과 Kotoba MLX 변환 모델을 함께 준비합니다.

클린 설치 기준으로 기본 설치 후 바로 GUI 실행, Kotoba MLX 전사, Qwen3-ASR 0.6B MLX 전사가 동작하는 것을 확인했습니다.

MPS 실험 경로를 빼고 설치하려면:

```bash
./install-silicon.sh --without-kotoba-mps
```

Kotoba MLX 경로를 빼고 설치하려면:

```bash
./install-silicon.sh --without-kotoba-mlx
```

MLX 의존성은 설치하되 변환 모델 준비만 건너뛰려면:

```bash
./install-silicon.sh --without-mlx-model
```

Qwen3-ASR MLX 의존성을 빼고 설치하려면:

```bash
./install-silicon.sh --without-qwen-mlx
```

Qwen3-ASR MLX 실행에 필요한 Python 의존성은 기본 설치에 포함됩니다. 다만 Qwen3-ASR 0.6B/1.7B 모델 본체는 설치 시 미리 받지 않고, GUI 또는 CLI에서 처음 선택해 실행할 때 Hugging Face 캐시에 다운로드됩니다.

## GUI 실행

```bash
./run-gui.command
```

Finder에서 실행할 수도 있습니다. macOS 보안 설정에 따라 처음 실행 시 터미널에서 직접 실행하는 편이 원인 확인에 쉽습니다.

GUI 전사는 기본적으로 `Kotoba-Whisper v2.2 MLX + pyannote MPS` 조합으로 실행됩니다. 전사 엔진에서 `Qwen3-ASR 0.6B MLX (고속실험)` 또는 `Qwen3-ASR 1.7B MLX`를 선택할 수 있습니다. 처리 장치 선택은 GUI에서 숨겨져 있으며, pyannote는 가능한 경우 MPS를 사용합니다.

Qwen3-ASR MLX를 처음 선택하면 모델 다운로드 때문에 첫 실행 시간이 길어질 수 있습니다. 이후 실행은 Hugging Face 캐시를 사용합니다.

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

Kotoba-Whisper v2.2 MLX:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend kotoba-mlx
```

변환 모델은 기본 설치 중 `models/kotoba-whisper-v2.2-mlx-q4`에 생성됩니다. 다시 만들거나 다른 위치를 쓰려면 `./tools/convert-kotoba-v22-mlx.sh [출력폴더]`를 실행하고 `--mlx-model-path`를 지정합니다.

Qwen3-ASR 1.7B MLX 실험:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend qwen3-mlx
```

더 작은 0.6B 모델로 속도를 먼저 확인하려면:

```bash
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 \
  --output-dir ./tmp-output \
  --asr-backend qwen3-mlx \
  --qwen-mlx-model-name Qwen/Qwen3-ASR-0.6B
```

## 번역

Ollama를 설치하고 번역 모델을 받은 뒤:

```bash
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run --no-sync kotoba process ../sample/ja_short_test.mp4 --translate
```

## 개발 메모

이 폴더는 의도적으로 `standalone/`과 중복을 허용합니다. Apple Silicon 경로가 안정화된 뒤 `media`, `subtitle`, `translate` 같은 순수 로직부터 공통화합니다.
