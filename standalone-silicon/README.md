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

번역 모델을 확인하면 `번역 추가 설정`에서 `없음`, `AV`, `애니메이션` 등의 프로필을 선택할 수 있습니다. `프로필 관리...`에서 추가 지시사항과 번역 참고 용어집을 편집할 수 있으며, 프로필은 `config/translation-profiles.json`에 저장됩니다. 용어집은 문맥을 고려하는 참고자료로 전달되며 기계적인 문자열 치환은 하지 않습니다.

프로필 관리의 `고급 설정` 탭에서는 원문 교정 규칙(`원문 => 교정문`)과 유령자막 필터(`정확히:문구`, `포함:문구`)를 지정할 수 있습니다. 이 규칙은 원본 SRT를 수정하지 않고 번역 입력에만 적용되며, 제외된 자막은 번역 metadata에 기록됩니다.

예를 들어 `ご視聴ありがとうございました`가 번역되어 나오는 `시청해주셔서 감사합니다` 자막을 제외하려면 `정확히:ご視聴ありがとうございました`를 입력합니다. 필터는 한국어 결과가 아니라 번역 전 일본어 원문을 기준으로 동작합니다.

기존 프로필을 영상별로 잠시 바꿔 쓰려면 `프로필 복제`로 복사본을 만든 뒤 고급 설정을 수정하고, 작업이 끝난 후 복사본을 삭제할 수 있습니다. 추가 지시사항과 원문 교정·필터 규칙에는 `클립보드 붙여넣기` 버튼이 있으며, 용어집은 여러 줄을 한 번에 붙여넣을 수 있습니다.

```text
先生 => 선생님
魔法 -> 마법
```

용어집 붙여넣기는 `원문 => 번역`, `원문 -> 번역`, 탭으로 구분한 `원문<TAB>번역<TAB>메모` 형식을 지원합니다. 잘못된 줄이 있으면 줄 번호를 표시하고 전체 붙여넣기를 취소합니다.

## 기존 설치 업데이트

이미 설치해서 사용 중인 경우에는 프로젝트를 최신 코드로 받은 뒤 Python 환경을 다시 동기화합니다. 새 기능이나 의존성이 추가된 경우 `git pull`만으로는 부족할 수 있습니다.

```bash
cd ~/Kotoba-Whisper-CLI
git pull

cd standalone-silicon
uv sync
```

설치 시 `./install-silicon.sh`에 옵션을 붙여 사용했다면, 필요에 따라 같은 옵션으로 설치 스크립트를 다시 실행해도 됩니다.

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

CLI에서는 프로필 이름을 직접 지정할 수도 있습니다.

```bash
uv run --no-sync kotoba translate ../sample/ja_short_test.ja.srt --translation-profile 애니메이션
```

## 개발 메모

이 폴더는 의도적으로 `standalone/`과 중복을 허용합니다. Apple Silicon 경로가 안정화된 뒤 `media`, `subtitle`, `translate` 같은 순수 로직부터 공통화합니다.
