# Kotoba-Whisper CLI

언어: **한국어** | [English](README.en.md)

일본어 영상에서 자막을 추출하고, 필요하면 Ollama로 한국어 자막까지 번역하는 로컬 도구입니다. 새로 설치하는 사용자에게는 **Docker가 필요 없는 `standalone/` 버전**을 권장합니다.

## 가장 쉬운 Windows 사용 방법

1. NVIDIA 그래픽카드 드라이버를 설치합니다.
2. `standalone\install-kotoba-kor.bat` 또는 `standalone\install-kotoba.bat`를 더블클릭합니다.
3. Python 3.12가 없다는 안내가 나오면 설치 여부를 확인하고 `Y`를 누릅니다.
4. 설치가 끝나면 `standalone\run-gui.bat`를 더블클릭합니다.
5. 화면에서 영상 파일 또는 폴더를 선택하고 `시작`을 누릅니다.

설치 배치 파일은 `uv`와 Python 3.12를 확인하고, 없으면 `winget`으로 설치합니다. Python 3.12 설치 전에는 사용자 확인을 한 번 받습니다. 프로그램은 패키지에 포함된 ffmpeg를 사용합니다. NVIDIA GPU가 있으면 CUDA 장치가 자동으로 표시되고, CUDA가 없으면 GUI는 CPU만 표시합니다. 한국어 번역을 사용하려면 [Ollama](https://ollama.com/download)가 추가로 필요합니다.

자세한 설명은 [standalone 한국어 설치 및 사용 안내](standalone/README.md)를 참고하세요.

## 명령어로 실행

PowerShell에서:

```powershell
cd C:\Python\Kotoba-Whisper-CLI\standalone
uv sync --group transcribe --group cuda --group pyannote
uv run --no-sync kotoba process "D:\Videos\sample.mp4"
```

음성 구간 검출은 기본적으로 번들된 pyannote VAD를 사용합니다.

기본 pyannote 모델은 MIT 라이선스에 따라 standalone과 Docker판에 포함되어 있으므로 Hugging Face 계정이나 토큰이 필요하지 않습니다. 원본 라이선스와 모델 카드는 [standalone 번들 모델 폴더](standalone/src/kotoba_standalone/models/pyannote-segmentation-3.0)와 [Docker 번들 모델 폴더](docker/app/models/pyannote-segmentation-3.0)에 보존되어 있습니다.

## 한국어 번역

Ollama를 설치한 뒤 권장 모델 중 하나를 받습니다.

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

자막 추출과 번역을 함께 실행하려면:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice
```

번역이 성공하면 한국어 자막은 작업 폴더뿐 아니라 원본 영상 옆에도 복사됩니다.

## 제공 버전

| 버전 | 권장 대상 | 설명 |
| --- | --- | --- |
| [standalone](standalone/README.md) | 새 사용자 | uv 기반 직접 실행, Windows GUI와 CLI 제공 |
| [Docker](docker/README.md) | 기존 watcher 사용자 | pyannote 기본, `input` 폴더 감시 방식 |

Docker 버전의 코드, 실행 스크립트, 설정, 데이터 폴더는 모두 `docker/`에 있습니다.

## 프로젝트 구조

```text
standalone/   권장 uv 기반 버전
docker/       기존 Docker watcher 버전
sample/       짧은 테스트 미디어
```

모델과 주요 라이브러리의 라이선스 및 재배포 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요.
