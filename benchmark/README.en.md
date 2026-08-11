# Real-World Japanese Benchmark

Language: [한국어](README.md) | **English**

This directory contains real-world Japanese benchmark utilities for regression-testing subtitle extraction with Kotoba-Whisper-CLI.

The benchmark is intended to exercise:

- ASR behavior on real Japanese audio
- VAD behavior around quiet speech, noise, dialogue, and long utterances
- Subtitle hallucination on non-speech and silence clips
- Difficult acoustic conditions such as field interviews, rapid turn-taking, background noise, and formal speeches

## Important Limitation

The generated real-world clips do not yet include reviewed ground-truth transcripts and timestamps.

These tools therefore do not calculate CER/WER, VAD precision/recall, or alignment scores yet. Reviewed Japanese references can be added later under `benchmark/references/`.

One useful exception is `20260119kaiken02.webm`: the Japanese Prime Minister's Office publishes an official transcript for that press conference. It is a strong future reference candidate once aligned and reviewed against the video.

## Build Clips

FFmpeg is required. Both `ffmpeg` and `ffprobe` must be available on `PATH`, or `KOTOBA_FFMPEG_PATH` must point to a known-good FFmpeg executable.

```bash
python benchmark/build_clips.py
```

Focused builds:

```bash
python benchmark/build_clips.py --source japanese_macaques
python benchmark/build_clips.py --clip S01_silence_30s
python benchmark/build_clips.py --force-download --source job_interview
```

Generated files are written under `benchmark/output/`:

- `audio/*.wav`: 16 kHz mono PCM signed 16-bit WAV clips
- `manifest.json`: clip origin, timestamps, tags, expected-speech status, license, and attribution
- `LICENSES.md`: source-level license and attribution summary

Downloaded source media is cached under `benchmark/cache/` and is not committed to git.

## Clip IDs

- `R01`-style names: real speech clips
- `N01`-style names: real non-speech negative controls
- `S01`-style names: synthetic silence clips

The current v1 set contains `R01`-`R07`, `N01`, and `S01`-`S02`.

## Compare ASR Outputs

Run transcription yourself first and save each backend into a separate work folder. For example:

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

Then compare the two existing output folders:

```bash
python benchmark/compare_asr.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen
```

For a focused comparison:

```bash
python benchmark/compare_asr.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen \
  --clip R07_short_utterance \
  --clip N01_environment_no_speech
```

`compare_asr.py` does not run ASR. It reads `.ja.txt`, `.process.json`, and optional `.subtitle-quality.json` files from both folders, then writes `summary.json` and `summary.md` to the common parent folder.

## Build A Disagreement Review Queue

You can generate short WAV clips only for time ranges where two ASR outputs disagree strongly:

```bash
python benchmark/disagreement_review.py \
  benchmark/output/asr-compare/kotoba-whisper-v2.2-mlx \
  benchmark/output/asr-compare/qwen3-asr-1.7b-mlx \
  --left-label kotoba \
  --right-label qwen
```

The tool reads each model's `.raw.json` timestamps first and falls back to `.ja.srt` if needed. It finds overlapping low-similarity regions and extracts 5-15 second WAV snippets.

Output:

- `benchmark/output/asr-compare/review-queue/review_queue.json`
- `benchmark/output/asr-compare/review-queue/review_queue.md`
- `benchmark/output/asr-compare/review-queue/review_clips/*.wav`

A human reviewer can listen to each short WAV and fill in `reference_text`. This lets the benchmark accumulate reviewed references from high-value disagreement regions first instead of transcribing entire files from scratch.

## Licensing

Source videos are not committed to this repository. They are downloaded from their original public locations when needed.

Generated clips inherit the license of the source material listed in `benchmark/sources.json` and `benchmark/output/LICENSES.md`. Synthetic silence clips contain no third-party content.

The optional HoneyWorks music/singing clip is intentionally omitted from v1 because the Wikimedia Commons page indicates license review is needed. It can be revisited later once licensing is clear.
