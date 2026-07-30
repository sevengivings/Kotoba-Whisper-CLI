# Kotoba-Whisper CLI

Windows 또는 Linux + Docker 환경에서 `input` 폴더에 넣은 영상/음성 파일을 `kotoba-tech/kotoba-whisper-v2.2`로 일본어 전사하고 SRT/TXT/JSON 결과를 생성하는 폴더 감시형 CLI입니다.

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
- `ごめん。`, `ありがとうございました。`처럼 자주 오탐되는 단독 문구는 길이와 무관하게 제거
- `.`, `。`, `、`, `??` 같은 구두점만 있는 자막 조각 제거
- word timestamp 시도 후 실패 시 segment timestamp로 자동 fallback
- word timestamp fallback 결과의 과도하게 긴 자막 표시 시간 자동 축소

## 실행 환경

- Windows 11 + Docker Desktop + WSL2 또는 Ubuntu + Docker Engine
- NVIDIA GPU 및 최신 NVIDIA 드라이버
- Windows 권장 위치: `C:\Python\Kotoba-Whisper-CLI`
- Ubuntu 예시 위치: `~/Kotoba-Whisper-CLI`

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

Ubuntu에서는:

```bash
cd ~/Kotoba-Whisper-CLI
chmod +x process-common.sh process-file.sh process-dir.sh
docker compose up -d
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

```bash
./process-file.sh "/mnt/best/sample.mp4"
```

제출만 하고 바로 돌아오려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -NoWait
```

```bash
./process-file.sh "/mnt/best/sample.mp4" --no-wait
```

특정 파일만 무음 감지 기준을 바꾸려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -SilenceThresholdDb -35dB
```

```bash
./process-file.sh "/mnt/best/sample.mp4" --silence-threshold-db -35dB
```

파일의 전체 음량 분포를 보고 무음 감지 기준을 자동으로 잡으려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -AutoSilenceThreshold
```

```bash
./process-file.sh "/mnt/best/sample.mp4" --auto-silence-threshold
```

전사 완료 후 Ollama로 한국어 자막까지 번역하려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -Translate
```

```bash
./process-file.sh "/mnt/best/sample.mp4" --translate
```

처음 사용할 모델을 목록에서 고르려면:

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -Translate -TranslateModelChoice
```

```bash
./process-file.sh "/mnt/best/sample.mp4" --translate --translate-model-choice
```

디렉터리 안의 지원 미디어 파일을 한꺼번에 처리 큐에 넣으려면:

```powershell
.\process-dir.ps1 "Y:\Best"
```

```bash
./process-dir.sh "/mnt/best"
```

하위 폴더까지 포함하려면:

```powershell
.\process-dir.ps1 "Y:\Best" -Recurse
```

```bash
./process-dir.sh "/mnt/best" --recurse
```

모든 제출 파일의 완료까지 기다리려면:

```powershell
.\process-dir.ps1 "Y:\Best"
```

제출만 하고 바로 돌아오려면:

```powershell
.\process-dir.ps1 "Y:\Best" -NoWait
```

```bash
./process-dir.sh "/mnt/best" --no-wait
```

디렉터리 안의 파일 전체에 같은 무음 감지 기준을 적용하려면:

```powershell
.\process-dir.ps1 "Y:\Best" -SilenceThresholdDb -35dB
```

```bash
./process-dir.sh "/mnt/best" --silence-threshold-db -35dB
```

디렉터리 안의 각 파일마다 무음 감지 기준을 자동으로 잡으려면:

```powershell
.\process-dir.ps1 "Y:\Best" -AutoSilenceThreshold
```

```bash
./process-dir.sh "/mnt/best" --auto-silence-threshold
```

디렉터리 전체를 전사 후 번역하려면:

```powershell
.\process-dir.ps1 "Y:\Best" -Translate
```

```bash
./process-dir.sh "/mnt/best" --translate
```

`process-file.ps1`/`process-dir.ps1`과 `process-file.sh`/`process-dir.sh`는 기본적으로 완료까지 기다립니다. `-NoWait` 또는 `--no-wait`를 붙이면 파일을 `input` 폴더에 제출한 뒤 바로 종료합니다. Docker 컨테이너가 떠 있지 않으면 제출 전에 `docker compose up -d --build`로 먼저 시작합니다. Docker 데몬이 응답하지 않는 경우에는 명확한 오류를 보여줍니다. `-SilenceThresholdDb` 또는 `--silence-threshold-db`를 붙이면 해당 제출 파일에만 `config.yaml`의 `silence_threshold_db`를 override합니다. `-AutoSilenceThreshold` 또는 `--auto-silence-threshold`를 붙이면 컨테이너 안에서 추출된 WAV의 frame RMS 분포를 분석해 파일별 `silence_threshold_db`를 자동으로 선택합니다. 두 옵션은 동시에 쓸 수 없습니다. 필요하면 `-MinSilenceDurationSeconds 0.4` 또는 `--min-silence-duration-seconds 0.4`처럼 최소 무음 길이도 함께 override할 수 있습니다. `process-dir` 스크립트는 기본적으로 `.mp4`, `.mkv`, `.mp3`, `.wav`, `.m4a` 등 지원 확장자만 복사합니다. 파일은 `.part`로 먼저 복사한 뒤 이름을 바꾸므로, watcher가 복사 중인 파일을 먼저 처리하지 않습니다.

`-Translate`는 완료를 기다린 뒤 `output\<name>.ja.srt`를 `output\<name>.ko.srt`로 번역합니다. 번역 전에는 `http://localhost:11434/api/tags`로 Ollama 서버와 모델 목록을 확인합니다. `-TranslateModelChoice`를 붙이면 Ollama 서버에 등록된 모델을 번호로 보여줍니다. 번역이 성공하면 사용한 모델을 `config/translation-defaults.json`에 저장하고, 다음부터는 `-Translate`만 붙여도 저장된 모델을 재사용합니다. 모델을 직접 지정하려면 `-TranslationModel "gemma3:4b"`처럼 넘기면 됩니다. 다른 PC나 컨테이너의 Ollama를 쓰려면 `-OllamaHost`와 `-OllamaPort`를 지정하세요.

번역은 기본적으로 자막 50개씩 묶어 처리합니다. 묶음 단위로 진행 상황이 표시되며, 더 크거나 작게 나누려면 `-BatchSize 100` 또는 `--batch-size 100`처럼 지정하세요. 한 줄씩 번역하려면 PowerShell에서는 `-NoBatchTranslate`, bash/직접 실행에서는 `--no-batch-translate`를 사용합니다. `--batch-translate`와 `-BatchTranslate`는 기존 명령 호환용으로 남아 있습니다. `--text-split-size`는 긴 프롬프트를 더 잘게 자르고 싶을 때만 추가로 쓰는 글자 수 제한입니다.

한국어 번역은 기본적으로 반말을 피하고 존댓말 자막체를 사용하도록 프롬프트를 강제합니다. 편한 말투가 필요하면 PowerShell에서는 `-KoreanStyle banmal`, bash/직접 실행에서는 `--korean-style banmal`을 사용하세요. 일본어 원문이 존대 표현이어도 한국어 출력을 비존대 informal 스타일로 강하게 고정하려면 `-KoreanStyle strict-banmal` 또는 `--korean-style strict-banmal`을 사용합니다.

스크립트로 제출한 파일은 원본을 복사한 staged copy이므로, 성공 후 기본적으로 삭제됩니다. 예전처럼 처리된 복사본을 `archive`에 남기려면 `-KeepStagedCopy`를 붙이세요.

```powershell
.\process-file.ps1 "Y:\Best\sample.mp4" -KeepStagedCopy
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
  - `-30dB`: 무음 판정이 강해져 작은 발화를 놓칠 수 있음
  - `-32dB`: 현재 권장값
  - `-35dB`: 작은 발화 보존에 조금 더 유리
  - `-40dB` 이하: 작은 소리 보존에 유리하지만 긴 segment와 오탐 가능성 증가
- `-AutoSilenceThreshold`: 파일 전체 음량 분포의 하위 20%를 배경 소음, 상위 85%를 음성 후보로 보고 `silence_threshold_db`를 자동 선택합니다. 선택된 값과 분석 정보는 `output\<name>.process.json`의 `auto_silence_threshold`에 기록됩니다.
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

자동 기준을 사용하면 처리 결과 JSON에 다음처럼 분석값이 남습니다.

```json
{
  "silence_threshold_db": "-42dB",
  "auto_silence_threshold": {
    "threshold_db": "-42dB",
    "noise_floor_db": -64.79,
    "speech_level_db": -45.3,
    "analyzed_frame_count": 73843
  }
}
```

`threshold_db`가 `-42dB`에 자주 붙으면 해당 영상들이 전체적으로 작은 음성 위주라는 뜻입니다. 이 경우 자동 기준을 쓰는 편이 기본 `-32dB`보다 작은 발화를 보존하는 데 유리합니다.

word timestamp가 성공하면 단어 단위 시간으로 자막을 묶습니다. 실패하면 segment timestamp로 자동 fallback하며, 이때 짧은 문장이 10초 이상 떠 있는 문제를 줄이기 위해 문장 길이 기반으로 표시 시간을 줄이고 필요하면 시작 시간을 약간 늦춥니다.

## 짧은 오탐 문구 필터

VAD 임계값을 낮추면 작은 잡음이나 숨소리가 `ごめん。` 같은 문구로 잘못 전사되는 경우가 있습니다. 이를 줄이기 위해 짧은 단독 문구 필터가 있습니다.

```yaml
filter_short_repeated_phrases: true
filtered_short_phrases:
  - すみません。
filtered_short_phrase_max_duration_s: 1.6
filtered_always_phrases:
  - ごめん。
  - ありがとうございました。
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

## 시간 예측

전사 결과의 `*.process.json`에는 `processing_seconds`와 `realtime_factor`가 기록되고, 번역 결과의 `*.translation.json`에는 번역 `processing_seconds`와 `subtitle_count`가 기록됩니다. 기존 `output` 기록을 바탕으로 다음 작업 시간을 대략 예측할 수 있습니다.

Windows:

```powershell
.\estimate-time.ps1
.\estimate-time.ps1 -MediaDurationMinutes 120 -SubtitleCount 900
.\estimate-time.ps1 -Recent 5
```

Ubuntu:

```bash
./estimate-time.sh
./estimate-time.sh --media-duration-minutes 120 --subtitle-count 900
./estimate-time.sh --recent 5
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
- 특정 파일에서만 음성 누락이 많음: `process-file.ps1` 또는 `process-dir.ps1`에 `-SilenceThresholdDb -35dB`처럼 override를 붙이세요.
