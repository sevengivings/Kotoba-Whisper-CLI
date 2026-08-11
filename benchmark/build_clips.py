from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_JSON = REPO_ROOT / "benchmark" / "sources.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "benchmark" / "cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "output"
FFMPEG_PATH_ENV = "KOTOBA_FFMPEG_PATH"
SAMPLE_RATE = "16000"
USER_AGENT = "Kotoba-Whisper-CLI benchmark builder/1.0"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 5.0
SILENCE_CLIPS = [
    {
        "id": "S01_silence_30s",
        "duration": 30.0,
        "tags": ["silence", "non_speech", "hallucination_test"],
        "expected_speech": False,
    },
    {
        "id": "S02_silence_60s",
        "duration": 60.0,
        "tags": ["silence", "non_speech", "hallucination_test"],
        "expected_speech": False,
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build real-world Japanese benchmark clips.")
    parser.add_argument("--sources-json", type=Path, default=DEFAULT_SOURCES_JSON)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--source", help="Build only one source id. Use 'synthetic' for silence clips.")
    parser.add_argument("--clip", help="Build only one clip id.")
    args = parser.parse_args(argv)

    try:
        build_benchmark(
            sources_json=args.sources_json,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            force_download=args.force_download,
            source_filter=args.source,
            clip_filter=args.clip,
        )
    except BenchmarkBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


class BenchmarkBuildError(RuntimeError):
    pass


def build_benchmark(
    *,
    sources_json: Path,
    cache_dir: Path,
    output_dir: Path,
    force_download: bool,
    source_filter: str | None,
    clip_filter: str | None,
) -> None:
    ffmpeg = ffmpeg_exe()
    ffprobe = ffprobe_exe(ffmpeg)
    config = load_sources(sources_json)
    audio_dir = output_dir / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_clips: list[dict[str, object]] = []
    used_sources: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    matched = False

    for source in config["sources"]:
        selected_clips = select_source_clips(source, source_filter, clip_filter)
        if not selected_clips:
            continue
        matched = True
        source_path = download_source(source, cache_dir, force_download, ffprobe)
        generated_for_source: list[dict[str, object]] = []
        for clip in selected_clips:
            wav_path = audio_dir / f"{clip['id']}.wav"
            extract_clip(ffmpeg, source_path, wav_path, clip["start"], clip["end"])
            info = validate_wav(ffprobe, wav_path)
            entry = clip_manifest_entry(source, clip, wav_path, output_dir, info)
            manifest_clips.append(entry)
            generated_for_source.append(entry)
            print(f"built {clip['id']} -> {entry['path']}")
        used_sources.append((source, generated_for_source))

    silence_clips = select_silence_clips(source_filter, clip_filter)
    if silence_clips:
        matched = True
    for clip in silence_clips:
        wav_path = audio_dir / f"{clip['id']}.wav"
        generate_silence(ffmpeg, wav_path, clip["duration"])
        info = validate_wav(ffprobe, wav_path)
        if abs(float(info["duration"]) - float(clip["duration"])) > 0.25:
            raise BenchmarkBuildError(
                f"{clip['id']} duration mismatch: expected about {clip['duration']}s, got {info['duration']}s"
            )
        entry = silence_manifest_entry(clip, wav_path, output_dir, info)
        manifest_clips.append(entry)
        print(f"built {clip['id']} -> {entry['path']}")

    if not matched:
        describe_missing_filter(config, source_filter, clip_filter)

    write_manifest(output_dir / "manifest.json", config["version"], manifest_clips)
    write_licenses(output_dir / "LICENSES.md", used_sources, silence_clips)
    print(f"wrote {output_dir / 'manifest.json'}")
    print(f"wrote {output_dir / 'LICENSES.md'}")


def load_sources(path: Path) -> dict[str, object]:
    if not path.exists():
        raise BenchmarkBuildError(f"sources.json not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkBuildError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data.get("sources"), list):
        raise BenchmarkBuildError(f"{path} must contain a 'sources' list")
    return data


def select_source_clips(
    source: dict[str, object],
    source_filter: str | None,
    clip_filter: str | None,
) -> list[dict[str, object]]:
    if source_filter and source_filter != source.get("id"):
        return []
    selected = []
    for clip in source.get("clips", []):
        if not isinstance(clip, dict):
            continue
        if clip_filter and clip_filter != clip.get("id"):
            continue
        selected.append(clip)
    return selected


def select_silence_clips(source_filter: str | None, clip_filter: str | None) -> list[dict[str, object]]:
    if source_filter and source_filter != "synthetic":
        return []
    return [clip for clip in SILENCE_CLIPS if not clip_filter or clip_filter == clip["id"]]


def download_source(
    source: dict[str, object],
    cache_dir: Path,
    force_download: bool,
    ffprobe: str,
) -> Path:
    filename = str(source.get("filename") or "")
    url = str(source.get("download_url") or "")
    if not filename or not url:
        raise BenchmarkBuildError(f"source {source.get('id')} must define filename and download_url")
    target = cache_dir / filename
    if target.exists() and target.stat().st_size > 0 and not force_download and source_media_is_valid(ffprobe, target):
        print(f"using cached {source['id']} -> {target}")
        return target
    print(f"downloading {source['id']} -> {target}")
    download_file(url, target)
    if not source_media_is_valid(ffprobe, target):
        raise BenchmarkBuildError(f"downloaded source media is not readable by ffprobe: {target}")
    return target


def download_file(url: str, target: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_file_once(url, target)
            return
        except BenchmarkBuildError as exc:
            last_error = exc
            if attempt >= DOWNLOAD_ATTEMPTS or not is_retryable_download_error(exc):
                break
            delay = DOWNLOAD_RETRY_DELAY_SECONDS * attempt
            print(f"download retry {attempt}/{DOWNLOAD_ATTEMPTS - 1} after {delay:.0f}s: {url}")
            time.sleep(delay)
    raise BenchmarkBuildError(f"failed to download {url}: {last_error}")


def download_file_once(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temp_path.stat().st_size <= 0:
            raise BenchmarkBuildError(f"downloaded file is empty: {url}")
        temp_path.replace(target)
    except (BenchmarkBuildError, HTTPError, URLError, TimeoutError, OSError) as exc:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise BenchmarkBuildError(str(exc)) from exc


def is_retryable_download_error(exc: BenchmarkBuildError) -> bool:
    message = str(exc)
    return "HTTP Error 429" in message or "HTTP Error 500" in message or "HTTP Error 503" in message


def ffmpeg_exe() -> str:
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
        raise BenchmarkBuildError(f"{FFMPEG_PATH_ENV} points to a missing ffmpeg executable: {configured}")
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise BenchmarkBuildError(
        "ffmpeg was not found. Install FFmpeg and add it to PATH, or set "
        f"{FFMPEG_PATH_ENV} to a known-good ffmpeg executable."
    )


def ffprobe_exe(ffmpeg: str) -> str:
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured:
        ffmpeg_path = Path(configured).expanduser()
        sibling = ffmpeg_path.with_name("ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe")
        if sibling.exists():
            return str(sibling)
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg_path = Path(ffmpeg)
    sibling = ffmpeg_path.with_name("ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe")
    if sibling.exists():
        return str(sibling)
    raise BenchmarkBuildError(
        "ffprobe was not found. Install FFmpeg with ffprobe and add it to PATH, "
        f"or place ffprobe next to {ffmpeg}."
    )


def source_media_is_valid(ffprobe: str, path: Path) -> bool:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False
    try:
        return float(result.stdout.strip()) > 0
    except ValueError:
        return False


def extract_clip(ffmpeg: str, source_path: Path, output_path: Path, start: str, end: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ss",
        start,
        "-to",
        end,
        "-vn",
        "-ac",
        "1",
        "-ar",
        SAMPLE_RATE,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run(command, f"failed to extract {output_path.name} from {source_path}")


def generate_silence(ffmpeg: str, output_path: Path, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run(command, f"failed to generate silence clip {output_path.name}")


def validate_wav(ffprobe: str, wav_path: Path) -> dict[str, object]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_type,sample_rate,channels,bits_per_sample:format=duration",
        "-of",
        "json",
        str(wav_path),
    ]
    result = run(command, f"failed to probe {wav_path}", capture=True)
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(data["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkBuildError(f"ffprobe returned unexpected metadata for {wav_path}") from exc

    expected = {
        "codec_name": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "bits_per_sample": 16,
    }
    actual = {
        "codec_name": stream.get("codec_name"),
        "sample_rate": str(stream.get("sample_rate")),
        "channels": int(stream.get("channels", 0)),
        "bits_per_sample": int(stream.get("bits_per_sample", 0)),
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise BenchmarkBuildError(f"{wav_path} has invalid {key}: {actual[key]} != {expected_value}")
    if duration <= 0:
        raise BenchmarkBuildError(f"{wav_path} has invalid duration: {duration}")
    return {**actual, "duration": round(duration, 3)}


def clip_manifest_entry(
    source: dict[str, object],
    clip: dict[str, object],
    wav_path: Path,
    output_dir: Path,
    wav_info: dict[str, object],
) -> dict[str, object]:
    return {
        "id": clip["id"],
        "path": relative_posix(wav_path, output_dir),
        "source_id": source["id"],
        "source_page": source["source_page"],
        "source_start": clip["start"],
        "source_end": clip["end"],
        "tags": list(clip.get("tags", [])),
        "expected_speech": bool(clip.get("expected_speech", True)),
        "selection_status": clip.get("selection_status", "selected"),
        "license": source["license"],
        "attribution": source["attribution"],
        "duration": wav_info["duration"],
    }


def silence_manifest_entry(
    clip: dict[str, object],
    wav_path: Path,
    output_dir: Path,
    wav_info: dict[str, object],
) -> dict[str, object]:
    return {
        "id": clip["id"],
        "path": relative_posix(wav_path, output_dir),
        "source_id": "synthetic",
        "source_start": "00:00:00",
        "source_end": seconds_to_time(float(clip["duration"])),
        "tags": list(clip["tags"]),
        "expected_speech": False,
        "selection_status": "selected",
        "license": "None",
        "attribution": "Generated digital silence",
        "duration": wav_info["duration"],
    }


def write_manifest(path: Path, version: str, clips: list[dict[str, object]]) -> None:
    payload = {
        "version": version,
        "audio_format": {
            "container": "wav",
            "codec": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        },
        "clips": sorted(clips, key=lambda item: str(item["id"])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_licenses(
    path: Path,
    used_sources: list[tuple[dict[str, object], list[dict[str, object]]]],
    silence_clips: list[dict[str, object]],
) -> None:
    lines = [
        "# Benchmark Licenses",
        "",
        "Generated clips inherit the license of their source material. Source media is not committed to this repository.",
        "",
    ]
    for source, clips in used_sources:
        lines.extend(
            [
                f"## {source['id']}",
                "",
                f"- Source page: {source['source_page']}",
                f"- License: {source['license']}",
                f"- Attribution: {source['attribution']}",
                f"- Clips: {', '.join(str(clip['id']) for clip in clips)}",
                "",
            ]
        )
    if silence_clips:
        lines.extend(
            [
                "## synthetic",
                "",
                "- Source page: generated locally",
                "- License: no third-party content",
                "- Attribution: Generated digital silence",
                f"- Clips: {', '.join(str(clip['id']) for clip in silence_clips)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def describe_missing_filter(config: dict[str, object], source_filter: str | None, clip_filter: str | None) -> None:
    source_ids = [str(source.get("id")) for source in config["sources"]]
    source_ids.append("synthetic")
    clip_ids = [
        str(clip.get("id"))
        for source in config["sources"]
        for clip in source.get("clips", [])
        if isinstance(clip, dict)
    ]
    clip_ids.extend(str(clip["id"]) for clip in SILENCE_CLIPS)
    if source_filter and source_filter not in source_ids:
        raise BenchmarkBuildError(f"unknown source id '{source_filter}'. Known sources: {', '.join(source_ids)}")
    if clip_filter and clip_filter not in clip_ids:
        raise BenchmarkBuildError(f"unknown clip id '{clip_filter}'. Known clips: {', '.join(clip_ids)}")
    raise BenchmarkBuildError("filters did not match any benchmark clips")


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def seconds_to_time(seconds: float) -> str:
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def run(command: list[str], context: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "command failed without stderr"
        raise BenchmarkBuildError(f"{context}: {stderr}")
    if capture:
        return result
    return result


if __name__ == "__main__":
    raise SystemExit(main())
