from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "output" / "manifest.json"
FFMPEG_PATH_ENV = "KOTOBA_FFMPEG_PATH"


@dataclass(frozen=True)
class Clip:
    id: str
    audio_path: Path
    duration: float
    expected_speech: bool
    tags: list[str]


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class WorkDir:
    label: str
    path: Path


@dataclass(frozen=True)
class Candidate:
    clip: Clip
    start: float
    end: float
    left_text: str
    right_text: str
    similarity: float
    reason: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create short review WAVs for ASR disagreement regions."
    )
    parser.add_argument("left_dir", type=Path, help="First ASR output folder.")
    parser.add_argument("right_dir", type=Path, help="Second ASR output folder.")
    parser.add_argument("--left-label", help="Display label for the first folder. Defaults to folder name.")
    parser.add_argument("--right-label", help="Display label for the second folder. Defaults to folder name.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write review_queue files. Defaults to benchmark/output/asr-compare/review-queue.",
    )
    parser.add_argument("--clip", action="append", help="Clip id to inspect. Repeatable. Defaults to all clips.")
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument("--min-overlap-s", type=float, default=0.2)
    parser.add_argument("--padding-s", type=float, default=0.6)
    parser.add_argument("--min-window-s", type=float, default=5.0)
    parser.add_argument("--max-window-s", type=float, default=15.0)
    parser.add_argument("--max-items", type=int, default=80)
    parser.add_argument("--no-audio", action="store_true", help="Write queue only; do not extract review WAVs.")
    args = parser.parse_args(argv)

    try:
        build_review_queue(
            left=WorkDir(args.left_label or args.left_dir.name, args.left_dir),
            right=WorkDir(args.right_label or args.right_dir.name, args.right_dir),
            manifest_path=args.manifest,
            output_dir=args.output_dir or default_output_dir(args.left_dir, args.right_dir),
            clip_filters=args.clip,
            similarity_threshold=args.similarity_threshold,
            min_overlap_s=args.min_overlap_s,
            padding_s=args.padding_s,
            min_window_s=args.min_window_s,
            max_window_s=args.max_window_s,
            max_items=args.max_items,
            extract_audio=not args.no_audio,
        )
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


class ReviewError(RuntimeError):
    pass


def build_review_queue(
    *,
    left: WorkDir,
    right: WorkDir,
    manifest_path: Path,
    output_dir: Path,
    clip_filters: list[str] | None,
    similarity_threshold: float,
    min_overlap_s: float,
    padding_s: float,
    min_window_s: float,
    max_window_s: float,
    max_items: int,
    extract_audio: bool,
) -> None:
    validate_workdir(left)
    validate_workdir(right)
    clips = select_clips(load_clips(manifest_path), clip_filters)
    candidates: list[Candidate] = []
    for clip in clips:
        candidates.extend(
            disagreement_candidates(
                clip,
                load_chunks(left, clip.id),
                load_chunks(right, clip.id),
                similarity_threshold=similarity_threshold,
                min_overlap_s=min_overlap_s,
                padding_s=padding_s,
                min_window_s=min_window_s,
                max_window_s=max_window_s,
            )
        )
    candidates = sorted(candidates, key=candidate_priority)[:max_items]
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "review_clips"
    if extract_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = ffmpeg_exe()
    else:
        ffmpeg = ""

    items = []
    for index, candidate in enumerate(candidates, 1):
        item_id = f"{candidate.clip.id}_{index:03d}"
        wav_path = audio_dir / f"{item_id}.wav"
        if extract_audio:
            extract_review_audio(ffmpeg, candidate.clip.audio_path, wav_path, candidate.start, candidate.end)
            audio_value = wav_path.relative_to(output_dir).as_posix()
        else:
            audio_value = None
        items.append(
            {
                "id": item_id,
                "clip_id": candidate.clip.id,
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "duration": round(candidate.end - candidate.start, 3),
                "audio": audio_value,
                "left_label": left.label,
                "right_label": right.label,
                "left_text": candidate.left_text,
                "right_text": candidate.right_text,
                "similarity": candidate.similarity,
                "reason": candidate.reason,
                "expected_speech": candidate.clip.expected_speech,
                "tags": candidate.clip.tags,
                "reference_text": "",
                "review_status": "pending",
            }
        )
    payload = {
        "left": {"label": left.label, "path": str(left.path)},
        "right": {"label": right.label, "path": str(right.path)},
        "settings": {
            "similarity_threshold": similarity_threshold,
            "min_overlap_s": min_overlap_s,
            "padding_s": padding_s,
            "min_window_s": min_window_s,
            "max_window_s": max_window_s,
            "max_items": max_items,
        },
        "items": items,
    }
    (output_dir / "review_queue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review_queue.md").write_text(review_markdown(payload), encoding="utf-8")
    print(f"wrote {output_dir / 'review_queue.json'}")
    print(f"wrote {output_dir / 'review_queue.md'}")
    if extract_audio:
        print(f"wrote {len(items)} review wav(s) under {audio_dir}")


def validate_workdir(workdir: WorkDir) -> None:
    if not workdir.path.exists() or not workdir.path.is_dir():
        raise ReviewError(f"{workdir.label} folder does not exist: {workdir.path}")


def default_output_dir(left: Path, right: Path) -> Path:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if left.parent == right.parent:
        return left.parent / "review-queue"
    return DEFAULT_MANIFEST.parent / "asr-compare" / "review-queue"


def load_clips(manifest_path: Path) -> list[Clip]:
    if not manifest_path.exists():
        raise ReviewError(f"benchmark manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    clips = []
    for item in data.get("clips", []):
        audio_path = base_dir / str(item["path"])
        clips.append(
            Clip(
                id=str(item["id"]),
                audio_path=audio_path,
                duration=float(item.get("duration") or 0.0),
                expected_speech=bool(item.get("expected_speech", True)),
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )
    if not clips:
        raise ReviewError(f"no clips found in {manifest_path}")
    return clips


def select_clips(clips: list[Clip], clip_filters: list[str] | None) -> list[Clip]:
    if not clip_filters:
        return clips
    requested = set(clip_filters)
    selected = [clip for clip in clips if clip.id in requested]
    missing = sorted(requested - {clip.id for clip in selected})
    if missing:
        raise ReviewError(f"unknown clip id(s): {', '.join(missing)}")
    return selected


def load_chunks(workdir: WorkDir, clip_id: str) -> list[Chunk]:
    raw_path = find_clip_file(workdir.path, clip_id, ".raw.json")
    if raw_path:
        chunks = chunks_from_raw_json(raw_path)
        if chunks:
            return chunks
    srt_path = find_clip_file(workdir.path, clip_id, ".ja.srt")
    if srt_path:
        return chunks_from_srt(srt_path)
    return []


def find_clip_file(root: Path, clip_id: str, suffix: str) -> Path | None:
    direct = root / f"{clip_id}{suffix}"
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{clip_id}{suffix}"))
    return matches[0] if matches else None


def chunks_from_raw_json(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_chunks = data.get("chunks", []) if isinstance(data, dict) else []
    chunks = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp") or item.get("timestamps")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            continue
        text = clean_text(str(item.get("text") or ""))
        if not text:
            continue
        try:
            start = float(timestamp[0] or 0.0)
            end = float(timestamp[1] or start)
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.01
        chunks.append(Chunk(start, end, text))
    return sorted(chunks, key=lambda chunk: (chunk.start, chunk.end))


def chunks_from_srt(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8-sig")
    chunks = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing = next((line for line in lines if "-->" in line), "")
        if not timing:
            continue
        index = lines.index(timing)
        body = clean_text("\n".join(lines[index + 1 :]))
        if not body:
            continue
        start_text, end_text = [part.strip() for part in timing.split("-->", 1)]
        chunks.append(Chunk(parse_srt_time(start_text), parse_srt_time(end_text), body))
    return sorted(chunks, key=lambda chunk: (chunk.start, chunk.end))


def disagreement_candidates(
    clip: Clip,
    left_chunks: list[Chunk],
    right_chunks: list[Chunk],
    *,
    similarity_threshold: float,
    min_overlap_s: float,
    padding_s: float,
    min_window_s: float,
    max_window_s: float,
) -> list[Candidate]:
    candidates = []
    matched_right: set[int] = set()
    for left in left_chunks:
        right_index, right = best_overlap(left, right_chunks, min_overlap_s)
        if right is None:
            candidates.append(
                make_candidate(clip, left.start, left.end, left.text, "", 0.0, "left_only", padding_s, min_window_s, max_window_s)
            )
            continue
        matched_right.add(right_index)
        similarity = text_similarity(left.text, right.text)
        if similarity < similarity_threshold:
            candidates.append(
                make_candidate(
                    clip,
                    min(left.start, right.start),
                    max(left.end, right.end),
                    left.text,
                    right.text,
                    similarity,
                    "low_similarity",
                    padding_s,
                    min_window_s,
                    max_window_s,
                )
            )
    for index, right in enumerate(right_chunks):
        if index not in matched_right:
            candidates.append(
                make_candidate(clip, right.start, right.end, "", right.text, 0.0, "right_only", padding_s, min_window_s, max_window_s)
            )
    return merge_similar_candidates(candidates)


def best_overlap(left: Chunk, right_chunks: list[Chunk], min_overlap_s: float) -> tuple[int, Chunk | None]:
    best_index = -1
    best_chunk = None
    best_score = 0.0
    for index, right in enumerate(right_chunks):
        overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        if overlap < min_overlap_s:
            continue
        denominator = max(0.001, min(left.end - left.start, right.end - right.start))
        score = overlap / denominator
        if score > best_score:
            best_index = index
            best_chunk = right
            best_score = score
    return best_index, best_chunk


def make_candidate(
    clip: Clip,
    start: float,
    end: float,
    left_text: str,
    right_text: str,
    similarity: float,
    reason: str,
    padding_s: float,
    min_window_s: float,
    max_window_s: float,
) -> Candidate:
    start = max(0.0, start - padding_s)
    end = min(clip.duration, end + padding_s)
    if end - start < min_window_s:
        center = (start + end) / 2
        start = max(0.0, center - min_window_s / 2)
        end = min(clip.duration, start + min_window_s)
        start = max(0.0, end - min_window_s)
    if end - start > max_window_s:
        center = (start + end) / 2
        start = max(0.0, center - max_window_s / 2)
        end = min(clip.duration, start + max_window_s)
        start = max(0.0, end - max_window_s)
    return Candidate(clip, start, end, left_text, right_text, round(similarity, 3), reason)


def merge_similar_candidates(candidates: list[Candidate], gap_s: float = 0.5) -> list[Candidate]:
    if not candidates:
        return []
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.clip.id, item.start, item.end)):
        if (
            merged
            and merged[-1].clip.id == candidate.clip.id
            and candidate.start - merged[-1].end <= gap_s
            and max(merged[-1].end, candidate.end) - merged[-1].start <= 15.0
        ):
            previous = merged.pop()
            merged.append(
                Candidate(
                    previous.clip,
                    previous.start,
                    max(previous.end, candidate.end),
                    join_unique(previous.left_text, candidate.left_text),
                    join_unique(previous.right_text, candidate.right_text),
                    min(previous.similarity, candidate.similarity),
                    join_unique(previous.reason, candidate.reason),
                )
            )
        else:
            merged.append(candidate)
    return merged


def candidate_priority(candidate: Candidate) -> tuple[int, float, str, float]:
    non_speech_priority = 0 if not candidate.clip.expected_speech else 1
    return (non_speech_priority, candidate.similarity, candidate.clip.id, candidate.start)


def extract_review_audio(ffmpeg: str, source: Path, output: Path, start: float, end: float) -> None:
    if not source.exists():
        raise ReviewError(f"benchmark audio is missing: {source}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ReviewError(f"failed to extract {output}: {result.stderr.strip()}")


def ffmpeg_exe() -> str:
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured and Path(configured).expanduser().exists():
        return str(Path(configured).expanduser())
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise ReviewError(f"ffmpeg not found. Install FFmpeg or set {FFMPEG_PATH_ENV}.")


def review_markdown(payload: dict[str, Any]) -> str:
    left = payload["left"]["label"]
    right = payload["right"]["label"]
    lines = [
        "# ASR Disagreement Review Queue",
        "",
        f"- Left: `{payload['left']['path']}`",
        f"- Right: `{payload['right']['path']}`",
        "",
        "| ID | Clip | Time | Similarity | Reason | Audio |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in payload["items"]:
        audio = item["audio"] or ""
        lines.append(
            f"| {item['id']} | {item['clip_id']} | {item['start']:.3f}-{item['end']:.3f} | {item['similarity']:.3f} | {item['reason']} | {audio} |"
        )
    lines.append("")
    for item in payload["items"]:
        lines.extend(
            [
                f"## {item['id']} ({item['clip_id']} {item['start']:.3f}-{item['end']:.3f})",
                "",
                f"- Audio: `{item['audio'] or ''}`",
                f"- {left}: {markdown_inline(item['left_text'])}",
                f"- {right}: {markdown_inline(item['right_text'])}",
                "- Reference: ",
                "",
            ]
        )
    return "\n".join(lines)


def parse_srt_time(value: str) -> float:
    match = re.match(r"^(\d+):(\d{2}):(\d{2})[,.](\d{3})$", value.strip())
    if not match:
        raise ReviewError(f"unsupported SRT timestamp: {value}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def text_similarity(left: str, right: str) -> float:
    return round(difflib.SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio(), 3)


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[\u3002\u3001,.!?！？、。]+", "", text)
    return text


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_unique(left: str, right: str) -> str:
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left} / {right}"


def markdown_inline(value: str) -> str:
    return clean_text(value).replace("|", "\\|") or "(empty)"


if __name__ == "__main__":
    raise SystemExit(main())
