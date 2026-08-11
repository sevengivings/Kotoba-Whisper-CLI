from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.common import ReferenceSegment, reference_to_srt, write_reference


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Utterance:
    start: float
    speaker: str
    text: str
    tags: list[str]
    gain_db: float = 0.0
    tempo: float = 1.0


@dataclass(frozen=True)
class NoiseBed:
    start: float
    end: float
    kind: str
    gain_db: float


@dataclass(frozen=True)
class Scenario:
    name: str
    duration: float
    utterances: list[Utterance]
    noise_beds: list[NoiseBed]


SCENARIOS = [
    Scenario(
        name="01_clean_female",
        duration=18.0,
        utterances=[
            Utterance(2.0, "female_01", "今日は新しいカメラについて説明します。", ["clean", "monologue"]),
            Utterance(8.0, "female_01", "まず明るい場所で短い動画を撮影します。", ["clean", "monologue"]),
        ],
        noise_beds=[],
    ),
    Scenario(
        name="02_long_silence_short_voice",
        duration=32.0,
        utterances=[
            Utterance(15.0, "female_01", "えっ？", ["short_utterance", "after_silence"]),
            Utterance(24.0, "male_01", "はい、今から始めます。", ["short_utterance", "after_silence"]),
        ],
        noise_beds=[],
    ),
    Scenario(
        name="03_interjections",
        duration=22.0,
        utterances=[
            Utterance(2.0, "female_01", "えっ？", ["interjection"]),
            Utterance(3.0, "male_01", "あ、うん。", ["interjection"]),
            Utterance(4.2, "female_01", "え？そうそう。", ["interjection", "rapid"]),
            Utterance(7.0, "male_01", "ちょっと待ってください。", ["short_utterance"]),
            Utterance(11.0, "female_01", "なるほど、わかりました。", ["short_utterance"]),
        ],
        noise_beds=[],
    ),
    Scenario(
        name="04_low_volume",
        duration=24.0,
        utterances=[
            Utterance(3.0, "male_01", "これは小さな声で話しているテストです。", ["low_volume"], gain_db=-18.0),
            Utterance(12.0, "female_01", "音量が低い場合でも聞き取れるか確認します。", ["low_volume"], gain_db=-22.0),
        ],
        noise_beds=[],
    ),
    Scenario(
        name="05_bgm_only",
        duration=30.0,
        utterances=[
            Utterance(4.0, "female_01", "後ろで音楽が流れています。", ["bgm"]),
            Utterance(11.0, "male_01", "音楽がある状態で説明を続けます。", ["bgm"]),
            Utterance(19.0, "female_01", "声と音楽のバランスを確認します。", ["bgm"]),
        ],
        noise_beds=[
            NoiseBed(0.0, 30.0, "music", -18.0),
        ],
    ),
    Scenario(
        name="06_overlap_dialogue",
        duration=24.0,
        utterances=[
            Utterance(3.0, "male_01", "少し速い会話を試します。", ["rapid_dialogue"], tempo=1.25),
            Utterance(4.0, "female_01", "はい、短く答えます。", ["rapid_dialogue"], tempo=1.25),
            Utterance(10.0, "male_01", "ここでは声が重なります。", ["overlap"]),
            Utterance(10.4, "female_01", "同時に話しています。", ["overlap"]),
            Utterance(16.0, "male_01", "最後は普通に話します。", ["clean"]),
        ],
        noise_beds=[
            NoiseBed(18.0, 22.0, "cafe_noise", -26.0),
        ],
    ),
]


VOICE_BY_SPEAKER = {
    "female_01": {"darwin": "Kyoko", "windows": "Haruka"},
    "male_01": {"darwin": "Otoya", "windows": "Ichiro"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Japanese benchmark audio.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample/synthetic-v1/generated"),
        help="Directory for generated wav/srt/txt/reference files.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary TTS files for debugging.")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be available on PATH.")
    backend = tts_backend()
    if backend is None:
        raise SystemExit("No supported local TTS backend found. macOS 'say' and Windows SAPI are supported.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "synthetic-v1",
        "sample_rate": SAMPLE_RATE,
        "generator": "sample/synthetic-v1/tools/generate.py",
        "tts_backend": backend,
        "scenarios": [],
    }
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        for scenario in SCENARIOS:
            generated = generate_scenario(scenario, args.output_dir, temp_dir, backend)
            manifest["scenarios"].append(generated)
        if args.keep_temp:
            kept = args.output_dir / "_temp"
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(temp_dir, kept)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(SCENARIOS)} synthetic benchmark file(s) in {args.output_dir}")
    return 0


def generate_scenario(
    scenario: Scenario,
    output_dir: Path,
    temp_dir: Path,
    backend: str,
) -> dict[str, object]:
    scenario_temp = temp_dir / scenario.name
    scenario_temp.mkdir(parents=True, exist_ok=True)
    inputs: list[Path] = []
    references: list[ReferenceSegment] = []

    base = scenario_temp / "base.wav"
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t",
        f"{scenario.duration:.3f}",
        str(base),
    ])
    inputs.append(base)

    for index, utterance in enumerate(scenario.utterances, 1):
        speech_path = scenario_temp / f"speech_{index:02d}.wav"
        synthesize_text(utterance.text, utterance.speaker, speech_path, backend)
        processed_path = scenario_temp / f"speech_{index:02d}_processed.wav"
        process_speech(speech_path, processed_path, utterance.gain_db, utterance.tempo)
        delayed_path = scenario_temp / f"speech_{index:02d}_delayed.wav"
        delay_audio(processed_path, delayed_path, utterance.start, scenario.duration)
        inputs.append(delayed_path)
        duration = probe_duration(processed_path)
        references.append(
            ReferenceSegment(
                start=round(utterance.start, 3),
                end=round(min(scenario.duration, utterance.start + duration), 3),
                speaker=utterance.speaker,
                text=utterance.text,
                tags=utterance.tags,
            )
        )

    for index, noise in enumerate(scenario.noise_beds, 1):
        noise_path = scenario_temp / f"noise_{index:02d}.wav"
        synthesize_noise(noise, noise_path, scenario.duration)
        inputs.append(noise_path)

    wav_path = output_dir / f"{scenario.name}.wav"
    mix_audio(inputs, wav_path)
    references.sort(key=lambda segment: (segment.start, segment.end))
    write_reference(
        output_dir / f"{scenario.name}.reference.json",
        name=scenario.name,
        duration=scenario.duration,
        segments=references,
    )
    (output_dir / f"{scenario.name}.srt").write_text(reference_to_srt(references), encoding="utf-8")
    (output_dir / f"{scenario.name}.txt").write_text(
        "\n".join(segment.text for segment in references if segment.text.strip()) + "\n",
        encoding="utf-8",
    )
    return {
        "name": scenario.name,
        "duration": scenario.duration,
        "wav": f"{scenario.name}.wav",
        "reference": f"{scenario.name}.reference.json",
        "srt": f"{scenario.name}.srt",
        "txt": f"{scenario.name}.txt",
    }


def tts_backend() -> str | None:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("say"):
        return "macos-say"
    if system == "windows" and shutil.which("powershell"):
        return "windows-sapi"
    return None


def synthesize_text(text: str, speaker: str, output_path: Path, backend: str) -> None:
    if backend == "macos-say":
        voice = VOICE_BY_SPEAKER.get(speaker, {}).get("darwin", "Kyoko")
        aiff_path = output_path.with_suffix(".aiff")
        run(["say", "-v", voice, "-o", str(aiff_path), text])
        convert_to_wav(aiff_path, output_path)
        return
    if backend == "windows-sapi":
        voice_hint = VOICE_BY_SPEAKER.get(speaker, {}).get("windows", "")
        script = windows_sapi_script(text, output_path, voice_hint)
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
        convert_to_wav(output_path, output_path)
        return
    raise RuntimeError(f"Unsupported TTS backend: {backend}")


def windows_sapi_script(text: str, output_path: Path, voice_hint: str) -> str:
    escaped_text = text.replace("'", "''")
    escaped_path = str(output_path).replace("'", "''")
    escaped_hint = voice_hint.replace("'", "''")
    return (
        "$voice = New-Object -ComObject SAPI.SpVoice; "
        f"$hint = '{escaped_hint}'; "
        "if ($hint) { "
        "  foreach ($candidate in $voice.GetVoices()) { "
        "    if ($candidate.GetDescription() -like \"*$hint*\") { $voice.Voice = $candidate; break } "
        "  } "
        "} "
        "$stream = New-Object -ComObject SAPI.SpFileStream; "
        "$format = New-Object -ComObject SAPI.SpAudioFormat; "
        "$format.Type = 22; "
        "$stream.Format = $format; "
        f"$stream.Open('{escaped_path}', 3, $false); "
        "$voice.AudioOutputStream = $stream; "
        f"$voice.Speak('{escaped_text}') | Out-Null; "
        "$stream.Close();"
    )


def convert_to_wav(source: Path, output_path: Path) -> None:
    temp_output = output_path.with_suffix(".converted.wav")
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(temp_output),
    ])
    temp_output.replace(output_path)


def process_speech(source: Path, output_path: Path, gain_db: float, tempo: float) -> None:
    filters = []
    if gain_db:
        filters.append(f"volume={gain_db}dB")
    if tempo != 1.0:
        filters.append(f"atempo={tempo}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(output_path)])
    run(command)


def delay_audio(source: Path, output_path: Path, start: float, duration: float) -> None:
    delay_ms = int(round(start * 1000))
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        f"adelay={delay_ms}:all=1,apad,atrim=0:{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])


def synthesize_noise(noise: NoiseBed, output_path: Path, total_duration: float) -> None:
    noise_duration = max(0.001, noise.end - noise.start)
    raw_path = output_path.with_name(output_path.stem + "_raw.wav")
    if noise.kind == "music":
        synthesize_music_bed(raw_path, noise_duration, noise.gain_db)
        delay_audio(raw_path, output_path, noise.start, total_duration)
        return
    source = f"anoisesrc=color=pink:sample_rate={SAMPLE_RATE}:duration={noise_duration:.3f}"
    filters = "highpass=f=200,lowpass=f=3500"
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        source,
        "-af",
        f"{filters},volume={noise.gain_db}dB",
        str(raw_path),
    ])
    delay_audio(raw_path, output_path, noise.start, total_duration)


def synthesize_music_bed(output_path: Path, duration: float, gain_db: float) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for frequency in (220, 277, 330):
        command.extend([
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate={SAMPLE_RATE}:duration={duration:.3f}",
        ])
    command.extend([
        "-filter_complex",
        f"amix=inputs=3:duration=first:dropout_transition=0:normalize=0,tremolo=f=3:d=0.25,volume={gain_db}dB",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])
    run(command)


def mix_audio(inputs: list[Path], output_path: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend([
        "-filter_complex",
        f"amix=inputs={len(inputs)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])
    run(command)


def probe_duration(path: Path) -> float:
    result = run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], capture=True)
    return float(result.stdout.strip())


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


if __name__ == "__main__":
    raise SystemExit(main())
