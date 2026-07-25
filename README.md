# Kotoba-Whisper V2.2 Windows 폴더 감시 자막 생성기

Windows 11에서 `input` 폴더에 영상 또는 음성 파일을 넣으면 Docker 컨테이너 안에서 `kotoba-tech/kotoba-whisper-v2.2`로 일본어 음성을 전사하고 SRT/TXT/JSON 결과를 생성합니다. 호스트의 Python, PyTorch 환경은 사용하거나 변경하지 않습니다.

## 필요 환경

- Windows 11
- Docker Desktop, WSL2 백엔드
- NVIDIA GeForce RTX 3090 또는 CUDA 컨테이너를 실행할 수 있는 NVIDIA GPU
- 최신 NVIDIA 드라이버
- 권장 설치 위치: `C:\AI\kotoba-folder-watcher` 또는 `C:\Python\Kotoba-Whisper-CLI`

경로에 공백이나 한글이 있어도 배치 파일은 가능한 범위에서 처리하지만, Docker/WSL 볼륨 문제를 줄이려면 짧은 영문 경로를 권장합니다.

## 고정 버전

- CUDA 이미지: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- 컨테이너 Python: `3.11`
- PyTorch: `torch==2.5.1`, `torchaudio==2.5.1`, CUDA 12.1 wheel
- Transformers: `4.46.3`
- Accelerate: `1.1.1`
- Punctuators: `0.0.5`
- Watchdog: `6.0.0`
- PyYAML: `6.0.2`

선택 이유: Kotoba-Whisper V2.2 모델 카드는 Transformers 4.39 이상과 `punctuators==0.0.5`를 안내합니다. PyTorch 2.5.1은 공식 설치표에 CUDA 12.1 wheel 조합이 명시되어 있고, RTX 3090의 FP16 및 SDPA 추론에 적합합니다. NVIDIA CUDA 문서는 `latest` 태그 사용을 피하고 명시 태그를 쓰는 방향이므로 Docker 이미지도 고정했습니다. V2.2의 원격 커스텀 pipeline은 pyannote 화자 분리를 포함하므로, 이 프로그램은 화자 분리 제외 요구사항에 맞춰 표준 Transformers ASR pipeline으로 모델을 직접 로드하고 문장부호만 별도 후처리합니다.

## GPU 확인

명령 프롬프트에서 프로젝트 폴더로 이동한 뒤:

```bat
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

GPU 이름과 드라이버 정보가 출력되어야 합니다. 실패하면 Docker Desktop의 WSL2 백엔드, NVIDIA 드라이버, GPU 컨테이너 지원을 먼저 확인하세요.

## 최초 실행

```bat
cd /d C:\Python\Kotoba-Whisper-CLI
start.bat
```

PowerShell을 선호하면:

```powershell
cd C:\Python\Kotoba-Whisper-CLI
.\start.ps1
```

첫 실행에서는 Docker 이미지 빌드와 Hugging Face 모델 다운로드 때문에 시간이 걸립니다. 모델 캐시는 `models` 폴더에 유지되므로 컨테이너를 다시 빌드해도 다시 다운로드하지 않습니다.

## 사용 방법

1. `input` 폴더에 `.mp4`, `.mkv`, `.mp3`, `.wav`, `.m4a` 같은 파일을 복사합니다.
2. 파일 크기가 약 15초 이상 안정화되면 처리를 시작합니다.
3. 처리 중 원본은 `processing`으로 이동합니다.
4. 성공하면 원본은 `archive`, 실패하면 `failed`로 이동합니다.

네트워크 드라이브나 다른 폴더의 파일을 직접 지정하려면 PowerShell에서 다음을 사용합니다.

```powershell
.\process-file.ps1 "Z:\Japanese\sample.mp4"
```

완료까지 기다리려면:

```powershell
.\process-file.ps1 "\\NAS\Videos\Japanese\sample.mp4" -Wait
```

이 스크립트는 원본 파일을 삭제하거나 이동하지 않습니다. 처리 안정성을 위해 `.part` 파일로 로컬 `input`에 staging한 뒤 감시기가 처리하게 합니다.

생성 파일:

- `output/sample.ja.srt`: 일본어 SRT
- `output/sample.ja.txt`: 일본어 텍스트
- `output/sample.raw.json`: 모델 원시 결과와 정리된 chunk
- `output/sample.process.json`: 처리 시간, GPU, 배치 크기, 완주 검증 결과

자막이 영상보다 빨리 뜨는 파일을 줄이기 위해 오디오 추출 시 `aresample=async=1:first_pts=0`을 적용합니다. 이 옵션은 MP4/MKV 안의 오디오 타임라인 공백을 WAV에 무음으로 보존해서, 대사 사이 긴 공백이 SRT 시간에도 반영되도록 돕습니다.

## 로그와 중지

```bat
logs.bat
stop.bat
```

PowerShell:

```powershell
.\logs.ps1
.\status.ps1
.\stop.ps1
```

로그 파일은 `logs/kotoba-folder-watcher.log`에 날짜별로 회전 저장됩니다.

## 설정 변경

`config/config.yaml`에서 다음을 바꿀 수 있습니다.

- `inference.batch_size`: 기본 8. GPU 메모리 부족 시 4, 2, 1로 자동 재시도합니다.
- `watcher.*`: 파일 안정화 시간과 폴더 스캔 주기
- `validation.*`: 마지막 자막 시간이 미디어 길이에 비해 너무 짧을 때 `suspicious_incomplete`로 처리하는 기준
- `output.utf8_bom`: SRT/TXT에 UTF-8 BOM을 넣을지 여부

설정을 바꾼 뒤:

```bat
docker compose restart
```

## 실패 파일 처리

실패한 원본은 `failed`로 이동하고, 실패 원인은 `failed/*.failure.json`에 저장됩니다. 실패 원인을 확인한 뒤 원본 파일을 다시 `input`에 넣으면 재처리됩니다.

컨테이너가 처리 중 종료되면 다음 시작 때 `processing`에 남은 지원 미디어 파일을 다시 큐에 넣습니다.

## 모델 캐시 삭제

모델을 다시 다운로드하고 싶으면 컨테이너를 중지한 뒤 `models` 폴더 내용을 삭제하세요.

```bat
stop.bat
```

그 다음 `models` 폴더 내부 파일을 삭제하고 `start.bat`을 다시 실행합니다.

## 수동 단일 파일 전사

컨테이너 실행 중 다음처럼 한 파일만 처리할 수도 있습니다.

```bat
docker compose run --rm kotoba-folder-watcher python3.11 -m app.main --config /workspace/config/config.yaml transcribe /workspace/input/sample.mp4
```

## 테스트

호스트 Python을 변경하지 않으려면 Docker 안에서 실행하세요.

```bat
docker compose run --rm kotoba-folder-watcher pytest
```

PowerShell:

```powershell
.\test.ps1
```

로컬에 pytest/PyYAML만 있는 개발 환경이라면 다음도 가능합니다.

```bat
python -m pytest
```

## 문제 해결

- `CUDA is not available`: Docker GPU 확인 명령이 성공하는지 먼저 확인하세요.
- `GPU memory exhausted`: `config/config.yaml`의 `inference.batch_size`를 4 또는 2로 낮추세요.
- 모델 다운로드 실패: 네트워크와 Hugging Face 접근 가능 여부를 확인하세요.
- `suspicious_incomplete`: 결과 파일은 보존되지만 원본은 기본적으로 `failed`로 이동합니다. 긴 무음 엔딩이 있는 파일이면 `validation.maximum_uncovered_tail_seconds`를 늘리거나 `suspicious_result_destination`을 `archive`로 바꿀 수 있습니다.

## 제한사항

- 일본어 전사만 수행합니다.
- 한국어 번역, DeepL, 로컬 LLM, 화자 분리, GUI, 웹 UI는 포함하지 않습니다.
- GPU 전체 추론 검증은 실제 RTX 3090 Docker 환경에서 수행해야 합니다.
