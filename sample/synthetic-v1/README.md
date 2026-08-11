# Synthetic Benchmark v1

This directory is reserved for generated Japanese synthetic benchmark media.

Generated WAV/SRT/TXT/reference files are intentionally not committed by default. Recreate them locally:

```bash
cd sample/synthetic-v1
python3 -m tools.generate
```

The first version focuses on a short smoke benchmark for regression checks:

- clean monologue
- long silence followed by short utterances
- short interjections
- low-volume speech
- background music with non-overlapping speech
- overlapping dialogue

Run Kotoba against the generated WAV files, then compare outputs:

```bash
cd sample/synthetic-v1
python3 -m tools.evaluate \
  generated/05_bgm_only.reference.json \
  --srt ../../standalone/tmp-output/05_bgm_only.ja.srt \
  --vad-json ../../standalone/tmp-output/05_bgm_only.vad.json
```

The synthetic set is useful for VAD and timing regression tests. It should not replace a real-world Japanese media benchmark because TTS speech is cleaner and more regular than natural recordings.

The generator currently uses local system TTS: macOS `say` or Windows SAPI. Japanese voices should be installed on the machine that generates the benchmark.
