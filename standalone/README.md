# Kotoba Standalone 설치와 사용 방법

언어: **한국어** | [English](README.en.md)

이 문서는 컴퓨터에 익숙하지 않은 분도 따라 할 수 있도록, Docker를 쓰지 않는 권장 standalone 버전의 설치와 사용 방법을 설명합니다.

standalone 버전은 동영상 파일을 직접 지정해서 일본어 자막을 만들고, 필요하면 Ollama로 한국어 자막까지 번역합니다. `ffmpeg`는 따로 설치하지 않아도 됩니다.

모델과 주요 라이브러리의 배포/라이선스 고지는 루트의 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)를 참고하세요.

## 가장 쉬운 실행 방법

PowerShell 명령어가 부담스럽다면 아래 두 파일부터 사용하세요.

1. 한국어 메시지를 보려면 `install-kotoba-kor.bat`, 영어 메시지도 괜찮으면 `install-kotoba.bat`를 더블클릭합니다.
2. 설치가 끝나면 `run-gui.bat`를 더블클릭합니다.
3. 작은 창이 뜨면 영상 파일이나 폴더를 선택하고 `시작`을 누릅니다.

`run-gui.bat`는 내부적으로 `kotoba-launcher`를 실행합니다. 런처에서는 입력 파일/폴더, 작업 폴더, 전사 엔진, 한국어 번역 여부와 Ollama 번역 모델을 화면에서 고를 수 있습니다. 음성 구간 검출은 번들된 pyannote를 자동으로 사용합니다. 실험적인 자막 품질 후처리는 CLI에서만 사용할 수 있습니다.

명령어를 직접 쓰는 방법은 아래에 계속 정리해 두었습니다. 문제가 생겼을 때는 명령어 방식이 원인 파악에 더 편합니다.

## 무엇을 설치하나요?

필요한 것은 네 가지입니다.

1. NVIDIA 그래픽카드 드라이버
2. Python 실행 환경을 관리하는 `uv`
3. 자막 번역에 사용할 Ollama
4. 이 프로젝트의 standalone Python 환경

설치 배치 파일은 `uv`와 Python 3.12를 확인하고, 없으면 `winget`으로 설치합니다. Python 3.12 설치 전에는 사용자에게 한 번 확인합니다. NVIDIA CUDA GPU가 있으면 빠르게 처리할 수 있고, GPU가 없으면 GUI의 `처리 장치`에는 `cpu`만 표시됩니다. CPU 전사는 가능하지만 매우 느릴 수 있습니다.

GPU가 없는 Windows PC의 GUI는 `Kotoba-Whisper faster CPU` 전사 엔진을 자동으로 선택합니다. 이 엔진은 CTranslate2/faster-whisper 기반이며, 기본 모델은 `RoachLin/kotoba-whisper-v2.2-faster`입니다. 기존 CUDA 환경에서는 기본 `Kotoba-Whisper v2.2` 경로를 그대로 사용합니다.

## 1. NVIDIA GPU 확인

PowerShell을 열고 아래 명령을 실행합니다.

```powershell
nvidia-smi
```

그래픽카드 이름과 메모리 정보가 보이면 준비가 된 것입니다. 명령을 찾을 수 없거나 오류가 나면 NVIDIA 드라이버를 먼저 설치하거나 업데이트하세요.

## 2. uv 설치

Windows에서는 PowerShell에서 아래 명령을 실행합니다.

```powershell
winget install --id astral-sh.uv -e
```

설치 후 새 PowerShell을 열고 확인합니다.

```powershell
uv --version
```

## 3. 프로젝트 폴더로 이동

예를 들어 프로젝트가 `C:\Python\Kotoba-Whisper-CLI`에 있다면:

```powershell
cd C:\Python\Kotoba-Whisper-CLI\standalone
```

이 문서의 명령은 모두 `standalone` 폴더에서 실행한다고 생각하시면 됩니다.

## 4. Python 환경 설치

처음 한 번만 실행합니다.

```powershell
uv sync --python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --no-managed-python --no-python-downloads --group transcribe --group cuda --group pyannote
```

다운로드가 오래 걸릴 수 있습니다. PyTorch와 CUDA 관련 파일이 크기 때문입니다.

설치가 끝나면 도움말이 보이는지 확인합니다.

```powershell
uv run --no-sync kotoba --help
```

GUI 런처를 열려면:

```powershell
uv run --no-sync kotoba-launcher
```

Windows Smart App Control이 켜져 있으면 새로 내려받은 Python 패키지의 `.dll` 또는 `.pyd` 파일이 차단될 수 있습니다. 이 경우 관리자 권한 실행만으로 해결되지 않을 수 있으며, Windows 보안의 Smart App Control 설정을 확인해야 합니다.

## 5. 짧은 샘플로 자막 추출 테스트

```powershell
uv run --no-sync kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output"
```

성공하면 `tmp-output` 폴더에 아래와 같은 파일이 생깁니다.

- `ja_short_test.ja.srt`: 일본어 자막
- `ja_short_test.ja.txt`: 일본어 텍스트
- `ja_short_test.raw.json`: 원본 전사 결과
- `ja_short_test.process.json`: 처리 시간과 설정 정보

## 실제 동영상 처리

파일 하나를 처리하려면:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output"
```

폴더 안의 동영상을 한 번에 처리하려면:

```powershell
uv run --no-sync kotoba process "D:\Videos" --output-dir ".\tmp-output"
```

폴더 입력은 해당 폴더 바로 아래의 미디어 파일만 처리합니다. 하위 폴더까지 재귀적으로 들어가지는 않습니다.

## 자막 타이밍 옵션

기본값으로 pyannote VAD pre-split이 켜져 있습니다. 프로그램은 사람 음성 구간을 찾아 해당 구간을 전사한 뒤 원래 영상 시간으로 다시 맞춥니다.

자주 쓰는 옵션은 아래 정도입니다.

- `--no-vad-pre-split`: VAD 분할 없이 전체 오디오를 한 번에 전사합니다.
- `--vad-padding-s 0.4`: 말 앞뒤로 약간의 여유 시간을 붙입니다.

일반 사용자는 VAD 옵션을 지정할 필요가 없습니다. 기존 음량 기반 방식을 비교해야 하는 경우에만 CLI에서 `--vad-engine ffmpeg`를 사용합니다. 이때 `--auto-silence-threshold` 또는 `--silence-threshold-db -42dB`를 함께 지정할 수 있습니다. 이 고급 FFmpeg 옵션은 GUI에는 표시되지 않습니다.

### 기본 pyannote VAD

standalone은 배경 음악이나 지속적인 잡음에서도 사람 음성을 구분하기 위해 pyannote VAD를 기본으로 사용합니다.

기본 `install-kotoba.bat`는 pyannote 의존성까지 함께 설치합니다. 기존 standalone 환경에 pyannote만 추가하려면 개발자용 `tools\install-pyannote-windows.bat`를 한 번 실행하거나 다음 명령을 사용합니다.

```powershell
uv sync --group transcribe --group cuda --group pyannote
```

기본 모델 `pyannote/segmentation-3.0`의 원본 가중치는 MIT 라이선스에 따라 standalone에 포함되어 있습니다. 따라서 기본 모델을 사용할 때는 Hugging Face 계정, 사용 조건 동의 또는 토큰 로그인이 필요하지 않습니다.

- 원본 모델: <https://huggingface.co/pyannote/segmentation-3.0>
- 고정 revision: `e66f3d3b9eb0873085418a7b813d3b369bf160bb`
- 번들 위치: `src/kotoba_standalone/models/pyannote-segmentation-3.0`
- 원본 `LICENSE`, 모델 카드, 파일 checksum을 번들 폴더에 함께 보존합니다.

그다음 아래처럼 실행합니다.

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output"
```

GUI는 pyannote를 자동으로 사용하며 기존 FFmpeg 음량 기반 VAD 조절값은 표시하지 않습니다. 다만 일부 손상된 오디오 스트림을 우회할 수 있도록 상태 영역에서 외부 FFmpeg 실행 파일을 지정할 수 있습니다. pyannote 실행이 끝나면 모델을 GPU 메모리에서 내린 뒤 Kotoba를 로드하므로 두 모델이 GPU에 계속 함께 남지는 않습니다.

- `--vad-engine pyannote`: 기본값이며 `pyannote/segmentation-3.0`으로 사람 음성 구간을 검출합니다.
- `--vad-engine ffmpeg`: 기존 음량 기반 VAD를 CLI에서 비교할 때만 사용합니다.
- `--pyannote-model`: 다른 로컬 checkpoint 또는 Hugging Face 모델을 지정할 때 사용합니다. 원격 gated 모델은 해당 사용자가 별도로 접근 권한과 토큰을 준비해야 합니다.

pyannote를 선택하면 `--auto-silence-threshold`와 `--silence-threshold-db`는 사용하지 않습니다. 결과의 `*.process.json`에는 VAD 엔진, pyannote 버전, 모델 이름, `bundled/local/huggingface` 출처와 검출된 음성 구간 수가 기록됩니다. `*.vad.json`에는 pyannote가 처음 검출한 구간과 실제 Kotoba 전사에 사용한 구간이 모두 저장됩니다.

## 자막 품질 실험 옵션(CLI 전용)

난도가 높은 영상에서는 ASR이 무음이나 반복 소리에서 가짜 자막을 만들거나, 0.01초짜리 표시 불가능한 자막을 만들 수 있습니다. 아래 옵션은 아직 실험용이며 기본값은 꺼져 있습니다.

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --report-subtitle-quality --drop-likely-hallucinations --split-long-subtitles
```

- `--report-subtitle-quality`: `*.subtitle-quality.json` 리포트를 만듭니다.
- `--drop-likely-hallucinations`: 확실한 오탐 자막만 제거합니다.
- `--split-long-subtitles`: 긴 정상 자막을 텍스트 경계 기준으로 나눕니다.
- `--tail-retranscribe-long-subtitles`: 10~30초짜리 의심 자막의 뒤쪽 5초를 다시 전사해, 원문과 비슷하면 시작 시간을 뒤로 보정합니다.
- `--tail-retranscribe-max-candidates 20`: 뒤쪽 5초 재전사 시도 개수입니다. 재분할 후보와 긴 후보를 우선 처리합니다.
- `--annotate-subtitle-quality`: SRT 본문 아래 줄에 품질 태그를 붙입니다.

현재 자동 제거 대상은 보수적으로 제한되어 있습니다.

- `ごめん`, `ありがとうございました`, `ありがとうございます`, `ご視聴ありがとうございました` 단독 자막
- 구두점만 있는 자막
- 0.3초 미만인데 텍스트가 긴 표시 불가능 자막

리포트에는 삭제하지 않고 검토만 필요한 후보도 표시됩니다.

- `drop_candidate`: 긴 저밀도 오탐 의심
- `refine_start_candidate`: 자막 시작이 너무 빠른 의심
- `resegment_candidate`: 긴 유성 구간을 다시 쪼개 전사할 후보
- `split_candidate`: 긴 정상 자막을 나눌 후보

`--tail-retranscribe-long-subtitles`는 `drop_candidate` 또는 `resegment_candidate` 중 10~30초 길이의 자막만 대상으로 삼습니다. 뒤쪽 5초 재전사 결과가 기존 자막과 충분히 비슷할 때만 보정하며, 시도 결과는 `tail_refine_attempts` 항목으로 리포트에 남깁니다. 기본값은 재분할 후보를 먼저 보고, 그 안에서 긴 후보부터 최대 20개입니다.

품질 태그를 화면에서 직접 확인하려면:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --report-subtitle-quality --annotate-subtitle-quality
```

`--annotate-subtitle-quality`는 SRT 텍스트에 `[quality: ...]` 줄을 실제로 추가하므로 Subtitle Edit이나 동영상 플레이어에서 검토할 때만 사용하세요. 번역까지 함께 실행하면 태그도 번역 입력에 들어갈 수 있습니다.

### Qwen3-ASR 전사 실험

Qwen3-ASR은 실험 백엔드입니다. Kotoba 기본값을 대체하지 않고, 같은 영상에서 전사 품질과 타임스탬프 품질을 비교할 때 사용합니다. `install-qwen3.bat` 설치 후 GUI의 `전사 엔진`에서 `Qwen3-ASR 1.7B (실험)`을 선택할 수 있습니다.

먼저 `install-qwen3.bat`를 실행하거나, Qwen 의존성을 별도 `.venv-qwen` 환경에 직접 추가합니다.

```powershell
$env:UV_PROJECT_ENVIRONMENT=".venv-qwen"
uv sync --group cuda --group pyannote --group qwen
Remove-Item Env:\UV_PROJECT_ENVIRONMENT
```

`qwen-asr`가 요구하는 `transformers` 버전은 Kotoba 기본 전사용 `transcribe` 그룹과 다르므로, Qwen은 `.venv-qwen`에 분리하고 Qwen 실험 환경에서는 `--group transcribe`를 함께 지정하지 않습니다.

Qwen3-ASR 1.7B와 Qwen3-ForcedAligner를 사용해 전사하려면:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --asr-backend qwen3 --model-dtype bfloat16
```

`--asr-backend qwen3`를 사용하면 현재 환경에 `qwen-asr`가 없을 때 CLI가 자동으로 `.venv-qwen`의 Python으로 같은 명령을 다시 실행합니다. Qwen Python 경로를 직접 지정하려면 `KOTOBA_QWEN_PYTHON` 환경 변수를 사용합니다.

기본 Qwen 설정은 다음과 같습니다.

- ASR 모델: `Qwen/Qwen3-ASR-1.7B`
- 강제정렬 모델: `Qwen/Qwen3-ForcedAligner-0.6B`
- 결과 메타데이터: `*.process.json`의 `asr_backend`, `asr_model`, `qwen_aligner_model`

VRAM이 부족하거나 속도 비교를 하고 싶다면 `--qwen-model-name Qwen/Qwen3-ASR-0.6B`를 지정해 볼 수 있습니다. 이 기능은 아직 일본어 AV 영상 기준 품질 검증 전이므로, 어려운 샘플에서 Kotoba 결과와 나란히 비교하는 용도입니다.

### WhisperX 싱크 보정 실험(CLI 전용)

WhisperX alignment는 이미 만들어진 일본어 자막의 시작 시간을 오디오에 다시 맞춰 보는 실험 기능입니다. GUI에는 표시하지 않습니다.

현재 standalone의 기본 Torch/pyannote 조합과 최신 WhisperX 의존성이 서로 맞지 않으므로, lock 파일에는 포함하지 않았습니다. 기존 standalone 환경에서 실험할 때만 다음처럼 최소 설치합니다.

```powershell
uv pip install --no-deps whisperx==3.4.5
uv pip install "nltk>=3.9.1"
```

이미 만든 SRT와 WAV를 보정하려면:

```powershell
uv run --no-sync kotoba align "D:\Videos\sample.ja.srt" "D:\Videos\sample.standalone.wav" --output "D:\Videos\sample.whisperx.ja.srt"
```

전사 과정 뒤에 바로 보정본을 함께 만들려면:

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --whisperx-align
```

결과는 원본 `*.ja.srt`를 덮어쓰지 않고 `*.whisperx.ja.srt`와 `*.whisperx-align.json`으로 따로 저장합니다. 일본어 짧은 감탄사가 너무 짧아지는 문제를 피하기 위해 기본적으로 WhisperX의 시작 시간은 반영하되 기존 자막의 종료 시간은 보존하고, 최소 0.8초 표시 시간을 보장합니다.

## Ollama 설치

한국어 번역을 하려면 Ollama가 필요합니다.

공식 다운로드:

https://ollama.com/download

설치 후 PowerShell에서 확인합니다.

```powershell
ollama list
```

## 번역 모델 선택

추천 모델은 두 가지입니다.

| 용도 | 모델 | Ollama 크기 | VRAM 시작점 |
| --- | --- | ---: | --- |
| 품질 우선 | `hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M` | 18GB | 24GB급 GPU 권장 |
| 최소 권장 | `hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M` | 4.6GB | 8GB급 최소, 12GB 이상 권장 |

Ollama 크기는 다운로드되는 모델 파일 크기입니다. 실제 VRAM 사용량은 context, KV cache, GPU offload, 동시에 실행 중인 프로그램에 따라 달라집니다.

24GB급 GPU가 있다면 품질 우선 모델을 받습니다.

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M
```

8GB 또는 12GB급 GPU라면 7B 모델부터 시도하세요.

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

`Hy-MT2-1.8B-GGUF:Q4_K_M`은 테스트에서 한국어 출력이 깨져서 현재 추천하지 않습니다.

## 이미 만든 일본어 자막 번역

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

`--output`에는 SRT 파일 경로 또는 결과 폴더를 지정할 수 있습니다. 폴더를 지정하면 원본 이름을 기준으로 `*.ko.srt` 파일을 그 안에 만듭니다.

Ollama에 설치된 모델을 번호로 고르려면:

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model-choice
```

번역이 성공하면 사용한 모델이 아래 파일에 저장됩니다.

```text
config\translation-defaults.json
```

다음부터는 `--model`을 생략해도 저장된 모델을 다시 사용합니다.

## 자막 추출과 번역을 한 번에 실행

```powershell
uv run --no-sync kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice
```

번역까지 성공하면 한국어 자막은 두 곳에 저장됩니다.

- 작업 결과 폴더: `tmp-output\영상이름.ko.srt`
- 원본 영상 폴더: `영상이름.srt`

원본 영상 폴더에 이미 `영상이름.srt`가 있으면 덮어쓰지 않고 `영상이름.ko.srt`로 저장합니다.

## 긴 자막 번역 전 모델 테스트

긴 자막을 번역하기 전에 짧은 문장으로 모델이 제대로 한국어를 출력하는지 확인할 수 있습니다.

```powershell
python .\tools\compare-ollama-models.py --timeout-seconds 30
```

현재 테스트 결과는 다음과 같습니다.

- 30B Q4: 한국어 품질이 가장 좋음
- 7B Q4: 중간 사양용으로 사용 가능
- 1.8B Q4: 출력이 깨져서 비추천

## 문제가 생겼을 때

### `nvidia-smi`가 안 됩니다

NVIDIA 드라이버가 설치되지 않았거나 PATH가 잡히지 않은 상태일 수 있습니다. 드라이버를 설치한 뒤 PowerShell을 새로 열어 다시 확인하세요.

### `uv` 명령을 찾을 수 없습니다

PowerShell을 새로 열어 보세요. 그래도 안 되면 `winget install --id astral-sh.uv -e`를 다시 실행하세요.

### Ollama가 꺼져 있다고 나옵니다

Ollama 앱을 실행한 뒤 다시 시도하세요. 확인 명령은 다음과 같습니다.

```powershell
ollama list
```

### FFmpeg 오디오 추출에 실패합니다

기본 사용에서는 FFmpeg를 따로 설치하지 않아도 됩니다. 다만 일부 MP4/MKV 파일은 AAC 오디오 스트림에 손상된 패킷이 있어 번들 FFmpeg가 중간에 멈출 수 있습니다. 이때는 외부 FFmpeg를 설치해 사용하면 처리되는 경우가 있습니다.

1. [FFmpeg 다운로드 페이지](https://ffmpeg.org/download.html)에서 Windows용 FFmpeg를 내려받습니다.
2. 압축을 풀고 `ffmpeg.exe`가 들어 있는 `bin` 폴더를 Windows PATH에 추가합니다.
3. PowerShell을 새로 열고 아래 명령으로 확인합니다.

```powershell
ffmpeg -version
```

PATH 설정이 어렵거나 특정 FFmpeg만 쓰고 싶다면 `run-gui.bat`의 아래 예시 줄에서 `REM `을 지우고 실제 경로로 바꾸세요.

```bat
REM set KOTOBA_FFMPEG_PATH=C:\Python\Faster-Whisper-XXL\ffmpeg.exe
```

### 모델이 너무 느리거나 로딩에 실패합니다

30B 모델 대신 7B 모델을 사용하세요.

```powershell
ollama pull hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M
```

## 테스트 실행

개발자가 기능 확인을 할 때는:

```powershell
uv run --group dev pytest
```
