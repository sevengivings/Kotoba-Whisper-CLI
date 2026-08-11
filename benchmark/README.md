# 실제 일본어 벤치마크

언어: **한국어** | [English](README.en.md)

이 디렉터리는 Kotoba-Whisper-CLI의 일본어 자막 추출 성능을 회귀 테스트하기 위한 실제 음성 benchmark 도구 모음입니다.

목표는 다음 상황을 작은 테스트 세트로 반복 확인하는 것입니다.

- 실제 일본어 음성에서의 ASR 동작
- 조용한 발화, 잡음, 대화, 긴 발화에서의 VAD 동작
- 비음성/무음 구간에서의 자막 hallucination
- 현장 인터뷰, 빠른 대화, 배경 소음, 공식 발언 같은 어려운 음향 조건

## 중요한 제한

현재 생성되는 실제 음성 클립에는 아직 검수된 정답 transcript와 timestamp가 없습니다.

따라서 이 단계의 도구는 CER/WER, VAD precision/recall, alignment score를 계산하지 않습니다. `benchmark/references/`에는 나중에 사람이 검수한 일본어 reference를 추가할 예정입니다.

다만 `20260119kaiken02.webm`의 경우 일본 총리관저 공식 transcript가 공개되어 있으므로, 향후 공식 텍스트를 기준으로 SRT를 정렬·검수하는 좋은 reference 후보입니다.

## 클립 생성

FFmpeg가 필요합니다. `ffmpeg`와 `ffprobe`가 `PATH`에 있거나, `KOTOBA_FFMPEG_PATH`가 정상 FFmpeg 실행 파일을 가리켜야 합니다.

```bash
python benchmark/build_clips.py
```

일부만 생성할 수도 있습니다.

```bash
python benchmark/build_clips.py --source japanese_macaques
python benchmark/build_clips.py --clip S01_silence_30s
python benchmark/build_clips.py --force-download --source job_interview
```

생성 결과는 `benchmark/output/` 아래에 저장됩니다.

- `audio/*.wav`: 16 kHz mono PCM signed 16-bit WAV
- `manifest.json`: 클립 원본, timestamp, tag, expected-speech, license, attribution
- `LICENSES.md`: 소스별 license와 attribution 요약

다운로드된 원본 미디어는 `benchmark/cache/`에 저장되며 git에 포함하지 않습니다.

## 클립 ID 체계

- `R01`-style: 실제 음성 클립
- `N01`-style: 실제 비음성 negative-control 클립
- `S01`-style: synthetic silence 클립

현재 v1은 `R01`-`R07`, `N01`, `S01`-`S02`로 구성됩니다.

## ASR 출력 비교

전사는 사용자가 직접 수행하고, 각 모델 결과를 별도 작업 폴더에 저장합니다. 예를 들면:

```bash
cd standalone-silicon

uv run --no-sync kotoba process ../benchmark/output/audio \
  --output-dir ../benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  --asr-backend kotoba-mlx

uv run --no-sync kotoba process ../benchmark/output/audio \
  --output-dir ../benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --asr-backend qwen3-mlx \
  --model-dtype float16 \
  --qwen-mlx-model-name Qwen/Qwen3-ASR-1.7B
```

그 다음 기존 출력 폴더 2개를 비교합니다.

```bash
python benchmark/compare_asr.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen
```

특정 클립만 빠르게 비교할 수도 있습니다.

```bash
python benchmark/compare_asr.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen \
  --clip R07_short_utterance \
  --clip N01_environment_no_speech
```

`compare_asr.py`는 ASR을 실행하지 않습니다. 두 폴더의 `.ja.txt`, `.process.json`, 선택적 `.subtitle-quality.json`을 읽고 공통 부모 폴더에 `summary.json`과 `summary.md`를 작성합니다.

## Disagreement Review Queue 생성

두 ASR 결과가 크게 다른 시간 구간만 사람이 검수하도록 짧은 WAV를 자동 생성할 수 있습니다.

```bash
python benchmark/disagreement_review.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen
```

이 도구는 각 모델의 `.raw.json` timestamp를 우선 사용하고, 없으면 `.ja.srt`를 사용합니다. 텍스트 유사도가 낮은 겹침 구간을 찾아 5-15초 WAV로 잘라냅니다.

출력:

- `benchmark/output/asr-compare/review-queue/review_queue.json`
- `benchmark/output/asr-compare/review-queue/review_queue.md`
- `benchmark/output/asr-compare/review-queue/review_clips/*.wav`

사람 검수자는 `review_queue.md`를 보면서 짧은 WAV를 듣고 `reference_text`에 정답을 채울 수 있습니다. 이 방식은 전체 파일을 처음부터 받아쓰지 않고, 모델 간 불일치가 큰 부분부터 reference를 축적하기 위한 것입니다.

## 라이선스

원본 영상은 저장소에 커밋하지 않고 원래 공개 위치에서 다운로드합니다.

생성된 클립은 `benchmark/sources.json`과 `benchmark/output/LICENSES.md`에 적힌 각 원본의 license를 따릅니다. Synthetic silence에는 제3자 콘텐츠가 없습니다.

선택 후보였던 HoneyWorks 음악/노래 클립은 Wikimedia Commons 페이지에 license review 필요 표시가 있어 v1에서는 제외했습니다. 나중에 라이선스가 명확해지면 다시 검토할 수 있습니다.
