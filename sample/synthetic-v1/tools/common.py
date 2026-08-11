from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceSegment:
    start: float
    end: float
    speaker: str
    text: str
    tags: list[str]


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def reference_to_srt(segments: list[ReferenceSegment]) -> str:
    blocks = []
    for index, segment in enumerate([s for s in segments if s.text.strip()], 1):
        blocks.append(
            f"{index}\n{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n{segment.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_reference(path: Path, *, name: str, duration: float, segments: list[ReferenceSegment]) -> None:
    payload = {
        "name": name,
        "duration": round(duration, 3),
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_reference(path: Path) -> tuple[float, list[ReferenceSegment]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["duration"]), [
        ReferenceSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            speaker=str(segment.get("speaker") or ""),
            text=str(segment.get("text") or ""),
            tags=[str(tag) for tag in segment.get("tags", [])],
        )
        for segment in payload.get("segments", [])
    ]


def parse_srt(path: Path) -> list[ReferenceSegment]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[ReferenceSegment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_line = next((line for line in lines if "-->" in line), "")
        if not timing_line:
            continue
        start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
        timing_index = lines.index(timing_line)
        subtitle_text = " ".join(lines[timing_index + 1 :]).strip()
        segments.append(
            ReferenceSegment(
                start=parse_srt_time(start_text),
                end=parse_srt_time(end_text),
                speaker="predicted",
                text=subtitle_text,
                tags=[],
            )
        )
    return segments


def parse_srt_time(value: str) -> float:
    hours_text, minutes_text, rest = value.split(":")
    seconds_text, millis_text = rest.split(",")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + int(seconds_text)
        + int(millis_text) / 1000
    )


def speech_intervals(segments: list[ReferenceSegment]) -> list[tuple[float, float]]:
    return [(segment.start, segment.end) for segment in segments if segment.text.strip()]


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_overlap(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    overlap = 0.0
    for left_start, left_end in merge_intervals(left):
        for right_start, right_end in merge_intervals(right):
            overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return overlap


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, 1):
        current = [i]
        for j, hyp_char in enumerate(hyp, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1] / len(ref)
