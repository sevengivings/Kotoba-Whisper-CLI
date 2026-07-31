# Kotoba Standalone 설치와 사용 방법

언어: **한국어** | [English](README.en.md)

이 문서는 컴퓨터에 익숙하지 않은 분도 따라 할 수 있도록, Docker를 쓰지 않는 standalone 버전의 설치와 사용 방법을 설명합니다.

standalone 버전은 동영상 파일을 직접 지정해서 일본어 자막을 만들고, 필요하면 Ollama로 한국어 자막까지 번역합니다. `ffmpeg`는 따로 설치하지 않아도 됩니다.

## 무엇을 설치하나요?

필요한 것은 네 가지입니다.

1. NVIDIA 그래픽카드 드라이버
2. Python 실행 환경을 관리하는 `uv`
3. 자막 번역에 사용할 Ollama
4. 이 프로젝트의 standalone Python 환경

자막 추출은 현재 NVIDIA CUDA GPU가 있는 Windows 11 또는 Linux를 기준으로 합니다. GPU가 없거나 CUDA가 잡히지 않으면 오디오 추출까지만 되고 전사는 실행되지 않습니다.

## 1. NVIDIA GPU 확인

PowerShell을 열고 아래 명령을 실행합니다.

```powershell
nvidia-smi
```

그래픽카드 이름과 메모리 정보가 보이면 준비가 된 것입니다. 명령을 찾을 수 없거나 오류가 나면 NVIDIA 드라이버를 먼저 설치하거나 업데이트하세요.

## 2. uv 설치

Windows에서는 PowerShell에서 아래 명령을 실행합니다.

```powershell
winget install --id Astral.UV -e
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
uv sync --group transcribe --group cuda
```

다운로드가 오래 걸릴 수 있습니다. PyTorch와 CUDA 관련 파일이 크기 때문입니다.

설치가 끝나면 도움말이 보이는지 확인합니다.

```powershell
uv run --group transcribe --group cuda kotoba --help
```

## 5. 짧은 샘플로 자막 추출 테스트

```powershell
uv run --group transcribe --group cuda kotoba process "..\sample\ja_short_test.mp4" --output-dir ".\tmp-output" --auto-silence-threshold
```

성공하면 `tmp-output` 폴더에 아래와 같은 파일이 생깁니다.

- `ja_short_test.ja.srt`: 일본어 자막
- `ja_short_test.ja.txt`: 일본어 텍스트
- `ja_short_test.raw.json`: 원본 전사 결과
- `ja_short_test.process.json`: 처리 시간과 설정 정보

## 실제 동영상 처리

파일 하나를 처리하려면:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos\sample.mp4" --output-dir ".\tmp-output" --auto-silence-threshold
```

폴더 안의 동영상을 한 번에 처리하려면:

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos" --output-dir ".\tmp-output" --auto-silence-threshold
```

폴더 입력은 해당 폴더 바로 아래의 미디어 파일만 처리합니다. 하위 폴더까지 재귀적으로 들어가지는 않습니다.

## 자막 타이밍 옵션

기본값으로 VAD pre-split이 켜져 있습니다. 프로그램은 먼저 무음 구간을 찾고, 말을 하는 구간만 잘라서 전사한 뒤 원래 영상 시간으로 다시 맞춥니다.

자주 쓰는 옵션은 아래 정도입니다.

- `--auto-silence-threshold`: 파일 전체 음량을 보고 무음 기준을 자동으로 잡습니다.
- `--silence-threshold-db -42dB`: 무음 기준을 직접 지정합니다.
- `--no-vad-pre-split`: VAD 분할 없이 전체 오디오를 한 번에 전사합니다.
- `--vad-padding-s 0.4`: 말 앞뒤로 약간의 여유 시간을 붙입니다.

처음에는 `--auto-silence-threshold`를 붙여 쓰는 것을 권장합니다.

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

Ollama에 설치된 모델을 번호로 고르려면:

```powershell
uv run kotoba translate ".\tmp-output\sample.ja.srt" --model-choice
```

번역이 성공하면 사용한 모델이 아래 파일에 저장됩니다.

```text
standalone\config\translation-defaults.json
```

다음부터는 `--model`을 생략해도 저장된 모델을 다시 사용합니다.

## 자막 추출과 번역을 한 번에 실행

```powershell
uv run --group transcribe --group cuda kotoba process "D:\Videos\sample.mp4" --translate --translation-model-choice --auto-silence-threshold
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

PowerShell을 새로 열어 보세요. 그래도 안 되면 `winget install --id Astral.UV -e`를 다시 실행하세요.

### Ollama가 꺼져 있다고 나옵니다

Ollama 앱을 실행한 뒤 다시 시도하세요. 확인 명령은 다음과 같습니다.

```powershell
ollama list
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

