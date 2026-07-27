# Kotoba-Whisper CLI

Windows + Docker 환경에서 `input` 폴더에 넣은 영상/음성 파일을 `kotoba-tech/kotoba-whisper-v2.2`로 일본어 전사하고 SRT/TXT/JSON 결과를 생성하는 폴더 감시형 CLI입니다.

## 주요 기능

- Docker 컨테이너 안에서 CUDA/GPU 기반 전사 실행
- `input` 폴더 감시 후 안정화된 미디어 파일 자동 처리
- 결과 파일 생성:
  - `output/<name>.ja.srt`
  - `output/<name>.ja.txt`
  - `output/<name>.raw.json`
  - `output/<name>.process.json`
- 처리 완료 원본은 `archive`, 실패 원본은 `failed`로 이동
- VAD/무음 기반 선분할로 긴 자막 뭉침 완화
- 짧은 발화 보존을 위한 VAD padding/merge 설정
- `ごめん。`, `ありがとうございました。` 같은 짧은 단독 hallucination 문구 필터링
- `ごめん。`처럼 자주 오탐되는 단독 문구는 길이와 무관하게 제거
- `.`, `。`, `、`, `??` 같은 구두점만 있는 자막 조각 제거
- word timestamp 시도 후 실패 시 segment timestamp로 자동 fallback
- word timestamp fallback 결과의 과도하게 긴 자막 표시 시간 자동 축소

## 실행 환경

- Windows 11
- Docker Desktop + WSL2
- NVIDIA GPU 및 최신 NVIDIA 드라이버
- 권장 위치: `C:\Python\Kotoba-Whisper-CLI`

고정 버전:

- CUDA 이미지: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Python: `3.11`
- PyTorch: `2.5.1`
- Transformers: `4.46.3`

## 시작

```powershell
cd C:\Python\Kotoba-Whisper-CLI
docker compose up -d
```

또는 제공 스크립트:

```powershell
.\start.ps1
```

상태 확인:

```powershell
docker compose ps
```

로그 확인:

```powershell
docker logs --tail 50 kotoba-folder-watcher
```

## 사용 방법

1. `input` 폴더에 `.mp4`, `.mkv`, `.mp3`, `.wav`, `.m4a` 같은 파일을 넣습니다.
2. 파일 크기와 수정 시간이 안정화되면 자동 처리됩니다.
3. 처리 중 원본은 `processing`으로 이동합니다.
4. 성공 시 원본은 `archive`, 실패 시 원본과 failure JSON은 `failed`로 이동합니다.

네트워크 드라이브나 다른 위치의 단일 파일을 처리하려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4"
```

완료까지 기다리려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -Wait
```

## 자막 분할 및 동기화 설정

주요 설정은 [config/config.yaml](config/config.yaml)에 있습니다.

현재 기본 방향은 대사를 놓치지 않되, fallback 결과에서 자막이 너무 오래 표시되지 않도록 조정하는 쪽입니다.

```yaml
inference:
  return_timestamps: true
  word_timestamps: true
  silence_split: true
  silence_threshold_db: -32dB
  min_silence_duration_s: 0.5
  min_subtitle_duration_s: 0.8

  vad_pre_split: true
  vad_max_segment_duration_s: 30
  vad_min_speech_duration_s: 0.25
  vad_padding_s: 0.4
  vad_merge_gap_s: 0.0

  subtitle_merge_gap_s: 0.5
  subtitle_max_merged_duration_s: 6.0
  subtitle_max_merged_chars: 42

  fallback_subtitle_max_duration_s: 5.0
  fallback_subtitle_chars_per_second: 5.0
  fallback_subtitle_padding_s: 0.4
  fallback_subtitle_start_delay_s: 0.5
```

설정 의미:

- `silence_threshold_db`: 무음 판정 기준입니다. 값이 낮을수록 작은 소리도 발화로 잡습니다.
  - `-30dB`: 배경음이 큰 영상에서 더 촘촘하게 나눔
  - `-32dB`: 현재 권장값
  - `-35dB`: 명확한 발화 위주, 누락 가능성 증가
  - `-40dB` 이하: 작은 소리 보존에 유리하지만 긴 segment와 오탐 가능성 증가
- `vad_pre_split`: 전사 전에 발화 구간을 먼저 나누어 Whisper에 넣습니다.
- `vad_min_speech_duration_s`: 짧은 발화도 버리지 않기 위한 최소 발화 길이입니다.
- `vad_padding_s`: VAD 구간 앞뒤에 붙이는 여유 시간입니다.
- `vad_merge_gap_s`: 가까운 발화 구간을 하나의 전사 조각으로 병합하는 간격입니다.
- `subtitle_merge_gap_s`: 전사 후 가까운 자막 조각을 합치는 간격입니다.
- `subtitle_max_merged_duration_s`: 병합된 자막의 최대 길이입니다.
- `fallback_subtitle_max_duration_s`: word timestamp fallback 시 자막 한 줄의 최대 표시 시간입니다.
- `fallback_subtitle_chars_per_second`: fallback 자막의 예상 읽기 속도입니다. 값이 낮을수록 더 오래 표시됩니다.
- `fallback_subtitle_padding_s`: fallback 자막 표시 시간에 더하는 여유 시간입니다.
- `fallback_subtitle_start_delay_s`: fallback 자막 전체를 뒤로 미는 시간입니다. VAD segment가 실제 발화보다 조금 일찍 시작할 때 보정합니다.

word timestamp가 성공하면 단어 단위 시간으로 자막을 묶습니다. 실패하면 segment timestamp로 자동 fallback하며, 이때 짧은 문장이 10초 이상 떠 있는 문제를 줄이기 위해 문장 길이 기반으로 표시 시간을 줄이고 필요하면 시작 시간을 약간 늦춥니다.

## 짧은 오탐 문구 필터

VAD 임계값을 낮추면 작은 잡음이나 숨소리가 `ごめん。` 같은 문구로 잘못 전사되는 경우가 있습니다. 이를 줄이기 위해 짧은 단독 문구 필터가 있습니다.

```yaml
filter_short_repeated_phrases: true
filtered_short_phrases:
  - すみません。
  - ありがとうございました。
filtered_short_phrase_max_duration_s: 1.6
filtered_always_phrases:
  - ごめん。
```

이 필터는 문구가 단독 자막이고 지정된 시간 이하일 때만 제거합니다. 예를 들어 `ごめん、待って。`처럼 문장 안에 포함된 경우는 유지합니다.
`filtered_always_phrases`에 있는 문구는 표시 시간과 무관하게 단독 자막이면 제거합니다.

또한 `.`, `。`, `、`, `??`, `?？`처럼 구두점만 있는 자막 조각은 최종 출력 전에 자동으로 제거됩니다.

## Docker 적용 방법

`config/config.yaml`만 수정한 경우에는 이미지 재빌드가 필요 없습니다. 컨테이너 재시작만 하면 됩니다.

```powershell
docker compose restart kotoba-folder-watcher
```

코드 파일(`app/*.py`)이나 `requirements.txt`, `Dockerfile`을 수정한 경우에는 이미지를 다시 빌드해야 합니다.

```powershell
docker compose build
docker compose up -d
```

상태 확인:

```powershell
docker compose ps
```

로그 확인:

```powershell
docker logs --tail 80 kotoba-folder-watcher
```

## 실패 파일과 재시도 카운터

실패 시 원본은 `failed`로 이동하고 실패 원인은 `failed/*.failure.json`에 저장됩니다.

같은 파일명이 여러 번 실패하면 `processing/.attempts.json`에 재시도 횟수가 쌓이며, `recovery.maximum_attempts`를 넘으면 즉시 실패 처리됩니다. 이 경우 다시 처리하려면:

1. `processing/.attempts.json`에서 해당 파일명 항목을 제거합니다.
2. `failed`에 있는 원본 파일을 `input`으로 다시 옮깁니다.

예:

```powershell
Move-Item -LiteralPath "failed\ROYVR-023 [8K] - A.mp4" -Destination "input\ROYVR-023 [8K] - A.mp4"
```

파일명을 바꿔 다시 넣어도 새 작업으로 처리됩니다.

## 테스트

Docker 환경에서 전체 테스트:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace kotoba-folder-watcher:2.2 python3.11 -m pytest -q
```

또는:

```powershell
.\test.ps1
```

## 자주 보는 문제

- `CUDA is not available`: Docker GPU 설정, NVIDIA 드라이버, WSL2 GPU 지원을 확인하세요.
- `GPU memory exhausted`: `inference.batch_size`를 낮추세요.
- `Maximum attempts exceeded`: `processing/.attempts.json`의 해당 파일 항목을 초기화하세요.
- 작은 소리 누락: `silence_threshold_db`를 더 낮추세요. 예: `-32dB`에서 `-35dB`.
- 자막이 너무 오래 표시됨: `fallback_subtitle_max_duration_s`를 낮추거나 `fallback_subtitle_chars_per_second`를 높이세요.
- 자막이 너무 빨리 사라짐: `fallback_subtitle_chars_per_second`를 낮추거나 `fallback_subtitle_padding_s`를 높이세요.
- 자막이 전반적으로 빠르게 표시됨: `fallback_subtitle_start_delay_s`를 높이세요. 예: `0.5`에서 `0.8`.
- 자막이 전반적으로 늦게 표시됨: `fallback_subtitle_start_delay_s`를 낮추세요. 예: `0.5`에서 `0.2`.
- 짧은 오탐 증가: `filtered_short_phrases`에 문구를 추가하거나 `filtered_short_phrase_max_duration_s`를 조정하세요.
- 길이에 상관없이 빼고 싶은 단독 문구가 있음: `filtered_always_phrases`에 문구를 추가하세요.
