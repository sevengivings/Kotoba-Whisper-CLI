from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptionSample:
    path: Path
    media_duration_seconds: float
    processing_seconds: float
    realtime_factor: float


@dataclass(frozen=True)
class TranslationSample:
    path: Path
    subtitle_count: int
    processing_seconds: float
    seconds_per_subtitle: float
    model: str
    batch_translate: bool
    batch_size: int | None
    korean_style: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate transcription and translation time from output history.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--media-duration-seconds", type=float)
    parser.add_argument("--media-duration-minutes", type=float)
    parser.add_argument("--subtitle-count", type=int)
    parser.add_argument("--recent", type=int, default=0, help="Use only the most recent N samples per category.")
    args = parser.parse_args()

    output_dir = args.output_dir
    transcription_samples = load_transcription_samples(output_dir)
    translation_samples = load_translation_samples(output_dir)
    if args.recent > 0:
        transcription_samples = transcription_samples[: args.recent]
        translation_samples = translation_samples[: args.recent]

    media_duration_seconds = args.media_duration_seconds
    if media_duration_seconds is None and args.media_duration_minutes is not None:
        media_duration_seconds = args.media_duration_minutes * 60.0

    print("Time Estimate")
    print(f"  Output dir: {output_dir}")
    print()
    print_transcription_summary(transcription_samples, media_duration_seconds)
    print()
    print_translation_summary(translation_samples, args.subtitle_count)
    return 0


def load_transcription_samples(output_dir: Path) -> list[TranscriptionSample]:
    samples: list[TranscriptionSample] = []
    for path in sorted(output_dir.glob("*.process.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(path)
        if payload.get("status") not in {"success", "suspicious_incomplete"}:
            continue
        media_duration = as_positive_float(payload.get("media_duration_seconds"))
        processing_seconds = as_positive_float(payload.get("processing_seconds"))
        realtime_factor = as_positive_float(payload.get("realtime_factor"))
        if realtime_factor is None and media_duration and processing_seconds:
            realtime_factor = processing_seconds / media_duration
        if media_duration and processing_seconds and realtime_factor:
            samples.append(TranscriptionSample(path, media_duration, processing_seconds, realtime_factor))
    return samples


def load_translation_samples(output_dir: Path) -> list[TranslationSample]:
    samples: list[TranslationSample] = []
    for path in sorted(output_dir.glob("*.translation.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(path)
        if payload.get("status") != "success":
            continue
        subtitle_count = as_positive_int(payload.get("subtitle_count"))
        processing_seconds = as_positive_float(payload.get("processing_seconds"))
        if not subtitle_count or not processing_seconds:
            continue
        samples.append(
            TranslationSample(
                path=path,
                subtitle_count=subtitle_count,
                processing_seconds=processing_seconds,
                seconds_per_subtitle=processing_seconds / subtitle_count,
                model=str(payload.get("model", "")),
                batch_translate=bool(payload.get("batch_translate", False)),
                batch_size=as_positive_int(payload.get("batch_size")),
                korean_style=str(payload.get("korean_style", "")) or None,
            )
        )
    return samples


def print_transcription_summary(samples: list[TranscriptionSample], media_duration_seconds: float | None) -> None:
    print("Transcription")
    if not samples:
        print("  No process history found.")
        return
    realtime_factors = [sample.realtime_factor for sample in samples]
    print(f"  Samples: {len(samples)}")
    print(f"  Avg realtime factor:    {mean(realtime_factors):.4f}x")
    print(f"  Median realtime factor: {statistics.median(realtime_factors):.4f}x")
    print(f"  P75 realtime factor:    {percentile(realtime_factors, 75):.4f}x")
    print(f"  Fastest/slowest:        {min(realtime_factors):.4f}x / {max(realtime_factors):.4f}x")
    latest = samples[0]
    print(f"  Latest: {format_seconds(latest.processing_seconds)} for {format_seconds(latest.media_duration_seconds)} media")
    if media_duration_seconds:
        avg_seconds = media_duration_seconds * mean(realtime_factors)
        median_seconds = media_duration_seconds * statistics.median(realtime_factors)
        p75_seconds = media_duration_seconds * percentile(realtime_factors, 75)
        slowest_seconds = media_duration_seconds * max(realtime_factors)
        print(f"  Estimate for {format_seconds(media_duration_seconds)} media:")
        print(f"    Avg-based:      {format_seconds(avg_seconds)}")
        print(f"    Median-based:   {format_seconds(median_seconds)}")
        print(f"    P75-based:      {format_seconds(p75_seconds)}")
        print(f"    Slowest-based:  {format_seconds(slowest_seconds)}")


def print_translation_summary(samples: list[TranslationSample], subtitle_count: int | None) -> None:
    print("Translation")
    if not samples:
        print("  No translation history found.")
        return
    seconds_per_subtitle = [sample.seconds_per_subtitle for sample in samples]
    print(f"  Samples: {len(samples)}")
    print(f"  Avg seconds/subtitle:    {mean(seconds_per_subtitle):.3f}s")
    print(f"  Median seconds/subtitle: {statistics.median(seconds_per_subtitle):.3f}s")
    print(f"  P75 seconds/subtitle:    {percentile(seconds_per_subtitle, 75):.3f}s")
    print(f"  Slowest seconds/subtitle:{max(seconds_per_subtitle):.3f}s")
    latest = samples[0]
    print(
        "  Latest: "
        f"{format_seconds(latest.processing_seconds)} for {latest.subtitle_count} subtitles "
        f"({latest.seconds_per_subtitle:.3f}s/subtitle)"
    )
    if latest.model:
        print(f"  Latest model: {latest.model}")
    if latest.batch_translate:
        print(f"  Latest batch: enabled, size={latest.batch_size or 'unknown'}")
    else:
        print("  Latest batch: disabled")
    if latest.korean_style:
        print(f"  Latest Korean style: {latest.korean_style}")
    if subtitle_count:
        avg_seconds = subtitle_count * mean(seconds_per_subtitle)
        median_seconds = subtitle_count * statistics.median(seconds_per_subtitle)
        p75_seconds = subtitle_count * percentile(seconds_per_subtitle, 75)
        latest_seconds = subtitle_count * latest.seconds_per_subtitle
        slowest_seconds = subtitle_count * max(seconds_per_subtitle)
        print(f"  Estimate for {subtitle_count} subtitles:")
        print(f"    Avg-based:      {format_seconds(avg_seconds)}")
        print(f"    Median-based:   {format_seconds(median_seconds)}")
        print(f"    P75-based:      {format_seconds(p75_seconds)}")
        print(f"    Latest-based:   {format_seconds(latest_seconds)}")
        print(f"    Slowest-based:  {format_seconds(slowest_seconds)}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def as_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def as_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
