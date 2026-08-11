from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "output" / "manifest.json"


@dataclass(frozen=True)
class Clip:
    id: str
    duration: float
    expected_speech: bool
    tags: list[str]


@dataclass(frozen=True)
class WorkDir:
    label: str
    path: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two existing ASR result folders without running transcription."
    )
    parser.add_argument("left_dir", type=Path, help="First ASR output folder, e.g. benchmark/output/asr-compare/kotoba-mlx")
    parser.add_argument("right_dir", type=Path, help="Second ASR output folder, e.g. benchmark/output/asr-compare/qwen3-mlx")
    parser.add_argument("--left-label", help="Display label for the first folder. Defaults to folder name.")
    parser.add_argument("--right-label", help="Display label for the second folder. Defaults to folder name.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, help="Where to write summary files. Defaults to the common parent.")
    parser.add_argument("--clip", action="append", help="Clip id to compare. Repeatable. Defaults to all manifest clips.")
    args = parser.parse_args(argv)

    try:
        compare_folders(
            left=WorkDir(args.left_label or args.left_dir.name, args.left_dir),
            right=WorkDir(args.right_label or args.right_dir.name, args.right_dir),
            manifest_path=args.manifest,
            output_dir=args.output_dir or default_output_dir(args.left_dir, args.right_dir),
            clip_filters=args.clip,
        )
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


class CompareError(RuntimeError):
    pass


def compare_folders(
    *,
    left: WorkDir,
    right: WorkDir,
    manifest_path: Path,
    output_dir: Path,
    clip_filters: list[str] | None,
) -> None:
    validate_workdir(left)
    validate_workdir(right)
    clips = select_clips(load_clips(manifest_path), clip_filters)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [compare_clip(clip, left, right) for clip in clips]
    payload = {
        "left": {"label": left.label, "path": str(left.path)},
        "right": {"label": right.label, "path": str(right.path)},
        "results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_markdown(payload), encoding="utf-8")
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'summary.md'}")


def validate_workdir(workdir: WorkDir) -> None:
    if not workdir.path.exists():
        raise CompareError(f"{workdir.label} folder does not exist: {workdir.path}")
    if not workdir.path.is_dir():
        raise CompareError(f"{workdir.label} path is not a folder: {workdir.path}")


def default_output_dir(left: Path, right: Path) -> Path:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if left.parent == right.parent:
        return left.parent
    return DEFAULT_MANIFEST.parent / "asr-compare"


def load_clips(manifest_path: Path) -> list[Clip]:
    if not manifest_path.exists():
        raise CompareError(
            f"benchmark manifest not found: {manifest_path}. Run 'python3 benchmark/build_clips.py' first."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [
        Clip(
            id=str(item["id"]),
            duration=float(item.get("duration") or 0.0),
            expected_speech=bool(item.get("expected_speech", True)),
            tags=[str(tag) for tag in item.get("tags", [])],
        )
        for item in data.get("clips", [])
    ]
    if not clips:
        raise CompareError(f"no clips found in {manifest_path}")
    return clips


def select_clips(clips: list[Clip], clip_filters: list[str] | None) -> list[Clip]:
    if not clip_filters:
        return clips
    requested = set(clip_filters)
    selected = [clip for clip in clips if clip.id in requested]
    missing = sorted(requested - {clip.id for clip in selected})
    if missing:
        raise CompareError(f"unknown clip id(s): {', '.join(missing)}")
    return selected


def compare_clip(clip: Clip, left: WorkDir, right: WorkDir) -> dict[str, object]:
    left_result = read_result(left, clip)
    right_result = read_result(right, clip)
    left_text = str(left_result.get("text") or "")
    right_text = str(right_result.get("text") or "")
    left_norm = normalize_text(left_text)
    right_norm = normalize_text(right_text)
    return {
        "clip_id": clip.id,
        "duration": clip.duration,
        "expected_speech": clip.expected_speech,
        "tags": clip.tags,
        left.label: left_result,
        right.label: right_result,
        "text_equal": left_norm == right_norm,
        "text_similarity": round(difflib.SequenceMatcher(None, left_norm, right_norm).ratio(), 3)
        if left_norm or right_norm
        else 1.0,
        "char_delta": len(left_norm) - len(right_norm),
        "hallucination_candidate": {
            left.label: (not clip.expected_speech and bool(left_norm)),
            right.label: (not clip.expected_speech and bool(right_norm)),
        },
    }


def read_result(workdir: WorkDir, clip: Clip) -> dict[str, object]:
    txt_path = find_clip_file(workdir.path, clip.id, ".ja.txt")
    process_path = find_clip_file(workdir.path, clip.id, ".process.json")
    quality_path = find_clip_file(workdir.path, clip.id, ".subtitle-quality.json")
    text = txt_path.read_text(encoding="utf-8").strip() if txt_path else ""
    process = read_json(process_path) if process_path else {}
    quality = read_json(quality_path) if quality_path else {}
    normalized = normalize_text(text)
    processing_seconds = optional_float(process.get("processing_seconds"))
    return {
        "status": process.get("status", "success" if txt_path else "missing_output"),
        "processing_seconds": processing_seconds,
        "real_time_factor": round(processing_seconds / clip.duration, 3)
        if processing_seconds is not None and clip.duration > 0
        else None,
        "subtitle_count": optional_int(process.get("subtitle_count")),
        "quality_issue_count": optional_int(process.get("subtitle_quality_issue_count") or quality.get("issue_count")),
        "text": text,
        "text_chars": len(normalized),
        "text_preview": text[:120],
        "txt": str(txt_path) if txt_path else None,
        "process_json": str(process_path) if process_path else None,
        "quality_json": str(quality_path) if quality_path else None,
    }


def find_clip_file(root: Path, clip_id: str, suffix: str) -> Path | None:
    direct = root / f"{clip_id}{suffix}"
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{clip_id}{suffix}"))
    return matches[0] if matches else None


def summary_markdown(payload: dict[str, object]) -> str:
    left = str(payload["left"]["label"])  # type: ignore[index]
    right = str(payload["right"]["label"])  # type: ignore[index]
    lines = [
        "# ASR Compare Summary",
        "",
        "This compares existing ASR output folders. It does not run transcription and does not calculate CER/WER because reviewed references are not available yet.",
        "",
        f"- Left: `{payload['left']['path']}`",  # type: ignore[index]
        f"- Right: `{payload['right']['path']}`",  # type: ignore[index]
        "",
        f"| Clip | Duration | {left} status | {right} status | {left} sec | {right} sec | {left} chars | {right} chars | Similarity | Hallucination? |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["results"]:  # type: ignore[index]
        left_result = row[left]
        right_result = row[right]
        hallucination = row["hallucination_candidate"]
        hallucination_text = ", ".join(label for label, yes in hallucination.items() if yes)
        lines.append(
            "| {clip} | {duration:.1f} | {left_status} | {right_status} | {left_sec} | {right_sec} | {left_chars} | {right_chars} | {similarity:.3f} | {hallucination} |".format(
                clip=row["clip_id"],
                duration=float(row.get("duration") or 0.0),
                left_status=left_result.get("status", ""),
                right_status=right_result.get("status", ""),
                left_sec=format_optional_float(left_result.get("processing_seconds")),
                right_sec=format_optional_float(right_result.get("processing_seconds")),
                left_chars=left_result.get("text_chars", ""),
                right_chars=right_result.get("text_chars", ""),
                similarity=float(row.get("text_similarity") or 0.0),
                hallucination=hallucination_text,
            )
        )
    lines.extend(["", "## Text Preview", ""])
    for row in payload["results"]:  # type: ignore[index]
        lines.append(f"### {row['clip_id']}")
        lines.append(f"- {left}: {markdown_inline(str(row[left].get('text_preview') or ''))}")
        lines.append(f"- {right}: {markdown_inline(str(row[right].get('text_preview') or ''))}")
        lines.append("")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_optional_float(value: object) -> str:
    number = optional_float(value)
    return f"{number:.3f}" if number is not None else ""


def markdown_inline(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("|", "\\|") or "(empty)"


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"could not read {path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
