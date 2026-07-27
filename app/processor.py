from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.media import (
    detect_silences,
    extract_audio,
    extract_audio_segment,
    is_supported_media,
    probe_duration_seconds,
    speech_spans_from_silences,
)
from app.recovery import increment_attempt
from app.subtitle import (
    chunks_to_json,
    chunks_to_srt,
    chunks_to_txt,
    filter_short_repeated_phrases,
    filter_punctuation_only_chunks,
    group_chunks_by_timing,
    normalize_chunks,
    split_chunks_on_silence,
    tighten_fallback_subtitle_durations,
)
from app.transcriber import KotobaTranscriber

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessOutcome:
    status: str
    source: Path
    destination: Path | None
    process_json: Path | None


class MediaProcessor:
    def __init__(self, config: AppConfig, transcriber: KotobaTranscriber) -> None:
        self.config = config
        self.transcriber = transcriber
        self.attempts_file = config.paths.processing / ".attempts.json"

    def process_input_file(self, input_file: Path) -> ProcessOutcome:
        if not is_supported_media(input_file):
            LOGGER.info("Unsupported file ignored: %s", input_file)
            return ProcessOutcome("ignored", input_file, None, None)
        processing_file = unique_destination(self.config.paths.processing / input_file.name)
        shutil.move(str(input_file), str(processing_file))
        return self.process_processing_file(processing_file)

    def process_processing_file(self, processing_file: Path) -> ProcessOutcome:
        job_id = uuid.uuid4().hex[:12]
        attempt = increment_attempt(self.attempts_file, processing_file.name)
        started = now_iso()
        start_time = time.perf_counter()
        wav_path = self.config.paths.temp / f"{processing_file.stem}.{job_id}.wav"
        process_json_path = self.config.paths.output / f"{processing_file.stem}.process.json"

        LOGGER.info("Job %s started: %s attempt=%s", job_id, processing_file.name, attempt)
        if attempt > self.config.recovery.maximum_attempts:
            failure = {
                "status": "failed",
                "source_file": processing_file.name,
                "started_at": started,
                "finished_at": now_iso(),
                "error": f"Maximum attempts exceeded: {attempt}",
                "job_id": job_id,
            }
            failure_path = unique_destination(self.config.paths.failed / f"{processing_file.stem}.failure.json")
            write_json_atomic(failure_path, failure)
            destination = unique_destination(self.config.paths.failed / processing_file.name)
            shutil.move(str(processing_file), str(destination))
            LOGGER.error("Maximum attempts exceeded for %s", processing_file.name)
            return ProcessOutcome("failed", processing_file, destination, failure_path)
        try:
            cleanup_existing_parts(self.config.paths.output, processing_file.stem)
            media_duration = probe_duration_seconds(processing_file)
            LOGGER.info("Audio extraction started: %s", processing_file.name)
            extract_audio(processing_file, wav_path)
            silences = []
            if self.config.inference.silence_split or self.config.inference.vad_pre_split:
                try:
                    silences = detect_silences(
                        wav_path,
                        self.config.inference.silence_threshold_db,
                        self.config.inference.min_silence_duration_s,
                    )
                except RuntimeError as exc:
                    LOGGER.warning("Silence detection skipped: %s", exc)
            silence_count = len(silences)
            transcription, raw_chunks, segment_count = self._transcribe_audio(
                wav_path,
                media_duration,
                silences,
                job_id,
            )
            chunks = normalize_chunks(raw_chunks)
            if transcription.word_timestamps_used:
                chunks = group_chunks_by_timing(
                    chunks,
                    self.config.inference.word_max_gap_s,
                    self.config.inference.word_max_subtitle_duration_s,
                    self.config.inference.word_max_subtitle_chars,
                )
            elif self.config.inference.silence_split and silences:
                chunks = split_chunks_on_silence(
                    chunks,
                    silences,
                    self.config.inference.min_subtitle_duration_s,
                )
            chunks = group_chunks_by_timing(
                chunks,
                self.config.inference.subtitle_merge_gap_s,
                self.config.inference.subtitle_max_merged_duration_s,
                self.config.inference.subtitle_max_merged_chars,
            )
            if not transcription.word_timestamps_used:
                chunks = tighten_fallback_subtitle_durations(
                    chunks,
                    self.config.inference.min_subtitle_duration_s,
                    self.config.inference.fallback_subtitle_max_duration_s,
                    self.config.inference.fallback_subtitle_chars_per_second,
                    self.config.inference.fallback_subtitle_padding_s,
                )
            if self.config.inference.filter_short_repeated_phrases:
                chunks = filter_short_repeated_phrases(
                    chunks,
                    self.config.inference.filtered_short_phrases,
                    self.config.inference.filtered_short_phrase_max_duration_s,
                )
            chunks = filter_punctuation_only_chunks(chunks)
            last_end = chunks[-1].end if chunks else 0.0
            validation_status = validate_completion(media_duration, last_end, self.config)

            raw_payload = {
                "source_file": processing_file.name,
                "model": self.config.model.name,
                "language": "ja",
                "text": transcription.raw.get("text", ""),
                "chunks": chunks_to_json(chunks),
                "raw_result": transcription.raw,
            }
            process_payload: dict[str, Any] = {
                "status": validation_status,
                "source_file": processing_file.name,
                "started_at": started,
                "finished_at": now_iso(),
                "media_duration_seconds": media_duration,
                "last_subtitle_end_seconds": last_end,
                "processing_seconds": round(time.perf_counter() - start_time, 3),
                "realtime_factor": _realtime_factor(time.perf_counter() - start_time, media_duration),
                "device": transcription.device_name,
                "torch_version": transcription.torch_version,
                "torch_cuda_version": transcription.torch_cuda_version,
                "batch_size_used": transcription.batch_size_used,
                "word_timestamps_requested": self.config.inference.word_timestamps,
                "word_timestamps_used": transcription.word_timestamps_used,
                "silence_split": self.config.inference.silence_split,
                "detected_silence_count": silence_count,
                "vad_pre_split": self.config.inference.vad_pre_split,
                "transcription_segment_count": segment_count,
                "job_id": job_id,
            }

            self._write_outputs(processing_file.stem, chunks, raw_payload, process_payload)
            destination_root = self.config.paths.archive
            if validation_status == "suspicious_incomplete":
                destination_root = (
                    self.config.paths.failed
                    if self.config.validation.suspicious_result_destination == "failed"
                    else self.config.paths.archive
                )
            destination = unique_destination(destination_root / processing_file.name)
            shutil.move(str(processing_file), str(destination))
            LOGGER.info("Source moved to %s: %s", destination_root.name, destination.name)
            return ProcessOutcome(validation_status, processing_file, destination, process_json_path)
        except Exception as exc:
            LOGGER.exception("Job %s failed for %s: %s", job_id, processing_file, exc)
            failure = {
                "status": "failed",
                "source_file": processing_file.name,
                "started_at": started,
                "finished_at": now_iso(),
                "error": str(exc),
                "job_id": job_id,
            }
            failure_path = self.config.paths.failed / f"{processing_file.stem}.failure.json"
            write_json_atomic(unique_destination(failure_path), failure)
            destination = unique_destination(self.config.paths.failed / processing_file.name)
            if processing_file.exists():
                shutil.move(str(processing_file), str(destination))
            return ProcessOutcome("failed", processing_file, destination, failure_path)
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove temporary wav: %s", wav_path)

    def _transcribe_audio(
        self,
        wav_path: Path,
        media_duration: float | None,
        silences: list[Any],
        job_id: str,
    ) -> tuple[Any, list[dict[str, Any]], int]:
        if not self.config.inference.vad_pre_split or media_duration is None or not silences:
            transcription = self.transcriber.transcribe(str(wav_path))
            return transcription, extract_raw_chunks(transcription.raw), 1

        spans = speech_spans_from_silences(
            media_duration,
            silences,
            self.config.inference.vad_min_speech_duration_s,
            self.config.inference.vad_max_segment_duration_s,
            self.config.inference.vad_padding_s,
            self.config.inference.vad_merge_gap_s,
        )
        if not spans:
            transcription = self.transcriber.transcribe(str(wav_path))
            return transcription, extract_raw_chunks(transcription.raw), 1

        LOGGER.info("VAD pre-split transcription: segments=%s", len(spans))
        all_chunks: list[dict[str, Any]] = []
        last_transcription = None
        for index, span in enumerate(spans, 1):
            segment_path = self.config.paths.temp / f"{wav_path.stem}.{job_id}.{index:04d}.wav"
            try:
                extract_audio_segment(wav_path, segment_path, span.start, span.end)
                last_transcription = self.transcriber.transcribe(str(segment_path))
                all_chunks.extend(_offset_raw_chunks(extract_raw_chunks(last_transcription.raw), span.start))
            finally:
                try:
                    segment_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove temporary segment wav: %s", segment_path)

        if last_transcription is None:
            last_transcription = self.transcriber.transcribe(str(wav_path))
            return last_transcription, extract_raw_chunks(last_transcription.raw), 1
        return last_transcription, all_chunks, len(spans)

    def _write_outputs(
        self,
        basename: str,
        chunks: list[Any],
        raw_payload: dict[str, Any],
        process_payload: dict[str, Any],
    ) -> None:
        encoding = "utf-8-sig" if self.config.output.utf8_bom else "utf-8"
        if self.config.output.write_srt:
            write_text_atomic(self._output_path(basename, ".ja.srt"), chunks_to_srt(chunks), encoding)
        if self.config.output.write_txt:
            write_text_atomic(self._output_path(basename, ".ja.txt"), chunks_to_txt(chunks), encoding)
        if self.config.output.write_raw_json:
            write_json_atomic(self._output_path(basename, ".raw.json"), raw_payload)
        if self.config.output.write_process_json:
            write_json_atomic(self._output_path(basename, ".process.json"), process_payload)

    def _output_path(self, basename: str, suffix: str) -> Path:
        candidate = self.config.paths.output / f"{basename}{suffix}"
        if self.config.output.overwrite_existing:
            return candidate
        return unique_destination(candidate)


def extract_raw_chunks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("chunks"), list):
        return raw["chunks"]
    collected: list[dict[str, Any]] = []
    for key, value in raw.items():
        if str(key).startswith("chunks") and isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))
    return collected


def _offset_raw_chunks(raw_chunks: list[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    offset_chunks: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if timestamp is None:
            continue
        try:
            start, end = _offset_timestamp(timestamp, offset_s)
        except (TypeError, ValueError):
            continue
        copied = dict(chunk)
        copied["timestamp"] = [start, end]
        copied.pop("timestamps", None)
        offset_chunks.append(copied)
    return offset_chunks


def _offset_timestamp(timestamp: Any, offset_s: float) -> tuple[float, float]:
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = 0.0 if timestamp[0] is None else float(timestamp[0])
        end = start if timestamp[1] is None else float(timestamp[1])
        return start + offset_s, end + offset_s
    if isinstance(timestamp, dict):
        start = 0.0 if timestamp.get("start") is None else float(timestamp["start"])
        end = start if timestamp.get("end") is None else float(timestamp["end"])
        return start + offset_s, end + offset_s
    raise ValueError(f"Unsupported timestamp format: {timestamp!r}")


def validate_completion(media_duration: float | None, last_end: float, config: AppConfig) -> str:
    if not config.validation.enabled or media_duration is None or media_duration <= 0:
        return "success"
    coverage = last_end / media_duration
    uncovered_tail = media_duration - last_end
    if (
        coverage < config.validation.minimum_coverage_ratio
        and uncovered_tail > config.validation.maximum_uncovered_tail_seconds
    ):
        return "suspicious_incomplete"
    return "success"


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{timestamp}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(content, encoding=encoding)
    temp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def cleanup_existing_parts(output_dir: Path, basename: str) -> None:
    for part in output_dir.glob(f"{basename}*.part"):
        part.unlink(missing_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _realtime_factor(processing_seconds: float, media_duration: float | None) -> float | None:
    if not media_duration or media_duration <= 0:
        return None
    return round(processing_seconds / media_duration, 6)
