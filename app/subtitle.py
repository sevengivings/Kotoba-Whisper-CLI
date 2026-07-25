from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubtitleChunk:
    start: float
    end: float
    text: str


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+([。、！？,.!?])", r"\1", text)
    text = re.sub(r"([「『（])\s+", r"\1", text)
    text = re.sub(r"\s+([」』）])", r"\1", text)
    return text


def normalize_chunks(raw_chunks: list[dict[str, Any]]) -> list[SubtitleChunk]:
    chunks: list[SubtitleChunk] = []
    previous_end = 0.0
    previous_text = ""

    for item in raw_chunks:
        timestamp = item.get("timestamp") or item.get("timestamps")
        if timestamp is None:
            continue
        start, end = _parse_timestamp(timestamp)
        text = clean_text(str(item.get("text", "")))
        if not text:
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        if end <= start:
            end = start + 0.01
        if start < previous_end:
            start = previous_end
            if end <= start:
                end = start + 0.01
        if text == previous_text:
            continue
        chunks.append(SubtitleChunk(start=start, end=end, text=text))
        previous_end = end
        previous_text = text

    return chunks


def _parse_timestamp(timestamp: Any) -> tuple[float, float]:
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = 0.0 if timestamp[0] is None else float(timestamp[0])
        end = start if timestamp[1] is None else float(timestamp[1])
        return start, end
    if isinstance(timestamp, dict):
        start = 0.0 if timestamp.get("start") is None else float(timestamp["start"])
        end = start if timestamp.get("end") is None else float(timestamp["end"])
        return start, end
    raise ValueError(f"Unsupported timestamp format: {timestamp!r}")


def chunks_to_srt(chunks: list[SubtitleChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        blocks.append(
            f"{index}\n{format_srt_time(chunk.start)} --> {format_srt_time(chunk.end)}\n{chunk.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def chunks_to_txt(chunks: list[SubtitleChunk]) -> str:
    return "\n".join(chunk.text for chunk in chunks) + ("\n" if chunks else "")


def chunks_to_json(chunks: list[SubtitleChunk]) -> list[dict[str, Any]]:
    return [{"start": c.start, "end": c.end, "text": c.text} for c in chunks]

