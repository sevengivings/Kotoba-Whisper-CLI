from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tools.common import (
    ReferenceSegment,
    cer,
    interval_duration,
    interval_overlap,
    merge_intervals,
    parse_srt,
    read_reference,
    speech_intervals,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Kotoba output against a synthetic reference.")
    parser.add_argument("reference", type=Path, help="Path to *.reference.json")
    parser.add_argument("--srt", type=Path, help="Generated Kotoba *.ja.srt file")
    parser.add_argument("--vad-json", type=Path, help="Generated Kotoba *.vad.json file")
    parser.add_argument("--json-output", type=Path, help="Optional path to write evaluation JSON")
    args = parser.parse_args()

    result = evaluate(args.reference, srt_path=args.srt, vad_json_path=args.vad_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def evaluate(
    reference_path: Path,
    *,
    srt_path: Path | None = None,
    vad_json_path: Path | None = None,
) -> dict[str, Any]:
    duration, reference_segments = read_reference(reference_path)
    reference_speech = speech_intervals(reference_segments)
    result: dict[str, Any] = {
        "reference": str(reference_path),
        "duration": duration,
        "reference_speech_seconds": round(interval_duration(reference_speech), 3),
    }
    if srt_path is not None:
        predicted_segments = parse_srt(srt_path)
        result["srt"] = evaluate_srt(reference_segments, predicted_segments)
    if vad_json_path is not None:
        vad_payload = json.loads(vad_json_path.read_text(encoding="utf-8"))
        predicted_intervals = vad_intervals(vad_payload)
        result["vad"] = evaluate_intervals(reference_speech, predicted_intervals)
    return result


def evaluate_srt(
    reference_segments: list[ReferenceSegment],
    predicted_segments: list[ReferenceSegment],
) -> dict[str, Any]:
    reference_text = "".join(segment.text for segment in reference_segments if segment.text.strip())
    predicted_text = "".join(segment.text for segment in predicted_segments if segment.text.strip())
    reference_intervals = speech_intervals(reference_segments)
    predicted_intervals = speech_intervals(predicted_segments)
    interval_metrics = evaluate_intervals(reference_intervals, predicted_intervals)
    return {
        "subtitle_count": len(predicted_segments),
        "cer": round(cer(reference_text, predicted_text), 4),
        "text_reference_chars": len(reference_text),
        "text_predicted_chars": len(predicted_text),
        **interval_metrics,
    }


def evaluate_intervals(
    reference_intervals: list[tuple[float, float]],
    predicted_intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    reference_intervals = merge_intervals(reference_intervals)
    predicted_intervals = merge_intervals(predicted_intervals)
    overlap = interval_overlap(reference_intervals, predicted_intervals)
    reference_seconds = interval_duration(reference_intervals)
    predicted_seconds = interval_duration(predicted_intervals)
    matched_errors = boundary_errors(reference_intervals, predicted_intervals)
    return {
        "reference_speech_seconds": round(reference_seconds, 3),
        "predicted_speech_seconds": round(predicted_seconds, 3),
        "overlap_seconds": round(overlap, 3),
        "recall": round(overlap / reference_seconds, 4) if reference_seconds else None,
        "precision": round(overlap / predicted_seconds, 4) if predicted_seconds else None,
        "false_positive_seconds": round(max(0.0, predicted_seconds - overlap), 3),
        "missed_speech_seconds": round(max(0.0, reference_seconds - overlap), 3),
        "mean_boundary_error_s": round(mean(matched_errors), 3) if matched_errors else None,
        "matched_interval_count": len(matched_errors),
    }


def boundary_errors(
    reference_intervals: list[tuple[float, float]],
    predicted_intervals: list[tuple[float, float]],
) -> list[float]:
    errors: list[float] = []
    for ref_start, ref_end in reference_intervals:
        candidates = [
            (max(0.0, min(ref_end, pred_end) - max(ref_start, pred_start)), pred_start, pred_end)
            for pred_start, pred_end in predicted_intervals
        ]
        best = max(candidates, default=None)
        if best is None or best[0] <= 0:
            continue
        _, pred_start, pred_end = best
        errors.append((abs(ref_start - pred_start) + abs(ref_end - pred_end)) / 2)
    return errors


def vad_intervals(payload: dict[str, Any]) -> list[tuple[float, float]]:
    spans = payload.get("transcription_spans") or payload.get("raw_speech_spans") or []
    intervals = []
    for span in spans:
        try:
            intervals.append((float(span["start"]), float(span["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return intervals


if __name__ == "__main__":
    raise SystemExit(main())
