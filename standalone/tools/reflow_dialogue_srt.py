"""Make an existing translated SRT easier to follow in rapid dialogue.

The input has subtitle-level timestamps only. Times within a subtitle are
estimated from the relative text length; this tool does not identify speakers.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$")
SENTENCE_END_RE = re.compile(r"[.!?。！？]+")


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def parse_time(parts: tuple[str, ...]) -> int:
    hour, minute, second, millis = map(int, parts)
    return ((hour * 60 + minute) * 60 + second) * 1000 + millis


def format_time(ms: int) -> str:
    hour, rem = divmod(ms, 3_600_000)
    minute, rem = divmod(rem, 60_000)
    second, millis = divmod(rem, 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d},{millis:03d}"


def read_srt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid SRT block: {block[:80]!r}")
        match = TIME_RE.fullmatch(lines[1].strip())
        if match is None:
            raise ValueError(f"Invalid SRT timing: {lines[1]!r}")
        start_ms = parse_time(match.groups()[:4])
        end_ms = parse_time(match.groups()[4:])
        if end_ms <= start_ms:
            raise ValueError(f"Invalid cue duration: {lines[1]!r}")
        cues.append(Cue(start_ms, end_ms, " ".join(line.strip() for line in lines[2:] if line.strip())))
    return cues


def sentence_units(text: str) -> list[str]:
    units: list[str] = []
    cursor = 0
    for match in SENTENCE_END_RE.finditer(text):
        # Dots in an ellipsis belong to the same sentence.
        end = match.end()
        unit = text[cursor:end].strip()
        if unit:
            units.append(unit)
        cursor = end
    tail = text[cursor:].strip()
    if tail:
        units.append(tail)
    return units or [text.strip()]


def split_long_unit(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        candidates = [m.end() for m in re.finditer(r"[,，、]\s*|\s+", window)]
        candidates = [pos for pos in candidates if pos >= max_chars // 2]
        cut = max(candidates) if candidates else max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def weight(text: str) -> int:
    return max(1, len(re.sub(r"\s|[.!?。！？，,、…]", "", text)))


def reflow_cue(cue: Cue, *, max_chars: int, max_duration_ms: int, min_duration_ms: int) -> list[Cue]:
    units = [part for sentence in sentence_units(cue.text) for part in split_long_unit(sentence, max_chars)]
    duration = cue.end_ms - cue.start_ms
    if len(units) > 1:
        # A cue cannot hold more independently readable parts than its available time.
        while len(units) > 1 and duration < len(units) * min_duration_ms:
            index = min(range(len(units) - 1), key=lambda i: weight(units[i]) + weight(units[i + 1]))
            units[index : index + 2] = [units[index] + " " + units[index + 1]]
    if len(units) == 1:
        return [Cue(cue.start_ms, min(cue.end_ms, cue.start_ms + max_duration_ms), units[0])]

    weights = [weight(unit) for unit in units]
    total_weight = sum(weights)
    ends: list[int] = []
    used_weight = 0
    for index, part_weight in enumerate(weights[:-1]):
        used_weight += part_weight
        ideal = cue.start_ms + round(duration * used_weight / total_weight)
        minimum = (ends[-1] if ends else cue.start_ms) + min_duration_ms
        maximum = cue.end_ms - min_duration_ms * (len(units) - index - 1)
        ends.append(max(minimum, min(ideal, maximum)))
    ends.append(cue.end_ms)
    starts = [cue.start_ms, *ends[:-1]]
    return [Cue(start, min(end, start + max_duration_ms), text) for start, end, text in zip(starts, ends, units, strict=True)]


def write_srt(path: Path, cues: list[Cue]) -> None:
    blocks = [f"{i}\n{format_time(c.start_ms)} --> {format_time(c.end_ms)}\n{c.text}" for i, c in enumerate(cues, 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-duration", type=float, default=6.0, help="Maximum displayed seconds per cue")
    parser.add_argument("--max-chars", type=int, default=40, help="Maximum characters per text unit")
    parser.add_argument("--min-duration", type=float, default=0.65, help="Minimum allocated seconds per text unit")
    args = parser.parse_args()
    if args.max_duration <= 0 or args.max_chars <= 0 or args.min_duration <= 0:
        parser.error("duration and character limits must be positive")
    original = read_srt(args.input)
    result = [part for cue in original for part in reflow_cue(
        cue,
        max_chars=args.max_chars,
        max_duration_ms=round(args.max_duration * 1000),
        min_duration_ms=round(args.min_duration * 1000),
    )]
    write_srt(args.output, result)
    print(f"Wrote {len(result)} cues from {len(original)} source cues to {args.output}")


if __name__ == "__main__":
    main()
