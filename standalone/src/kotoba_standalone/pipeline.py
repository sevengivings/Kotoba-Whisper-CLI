from __future__ import annotations

import re
import time
import json
import shutil
from typing import Any
from pathlib import Path

from kotoba_standalone.alignment import (
    WhisperXAlignmentError,
    align_chunks_with_whisperx,
    write_alignment_metadata,
)
from kotoba_standalone.media import (
    detect_silences,
    estimate_silence_threshold,
    extract_audio,
    extract_audio_segment,
    is_supported_media,
    normalize_speech_spans,
    probe_duration_seconds,
    speech_spans_from_silences,
    wav_duration_seconds,
)
from kotoba_standalone.pyannote_vad import PyannoteVadError, detect_speech_spans_pyannote
from kotoba_standalone.qwen_transcriber import Qwen3Transcriber
from kotoba_standalone.progress import ProgressCallback
from kotoba_standalone.subtitle import (
    analyze_subtitle_quality,
    annotate_chunks_with_quality,
    apply_tail_refinements,
    chunks_to_srt,
    chunks_to_txt,
    filter_likely_hallucinations,
    group_chunks_by_timing,
    normalize_chunks,
    normalize_phrase,
    quality_issues_to_json,
    split_long_subtitle_candidates,
    tail_retranscribe_candidate_indexes,
    text_similarity,
    SubtitleChunk,
    SubtitleQualityIssue,
)
from kotoba_standalone.transcriber import KotobaTranscriber, TranscriptionDependencyError, extract_raw_chunks, offset_raw_chunks
from kotoba_standalone.translate.ollama import translate_srt
from kotoba_standalone.types import ProcessOptions, ProcessResult, ProgressEvent, TranslationOptions


def process_video(
    input_path: Path,
    options: ProcessOptions,
    progress: ProgressCallback | None = None,
) -> ProcessResult:
    started = time.time()
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not is_supported_media(input_path):
        raise ValueError(f"Unsupported media file: {input_path}")

    output_dir = (options.output_dir or input_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ja_srt_path = output_dir / f"{input_path.stem}.ja.srt"
    ja_txt_path = output_dir / f"{input_path.stem}.ja.txt"
    raw_json_path = output_dir / f"{input_path.stem}.raw.json"
    process_json_path = output_dir / f"{input_path.stem}.process.json"
    vad_json_path = output_dir / f"{input_path.stem}.vad.json"
    quality_json_path = output_dir / f"{input_path.stem}.subtitle-quality.json"
    whisperx_srt_path = output_dir / f"{input_path.stem}.whisperx.ja.srt"
    whisperx_json_path = output_dir / f"{input_path.stem}.whisperx-align.json"
    wav_path = output_dir / f"{input_path.stem}.standalone.wav"

    progress_total = 12 if options.translate and options.alignment_engine == "whisperx" else 11 if options.translate or options.alignment_engine == "whisperx" else 10
    _emit(progress, started, "prepare", "Preparing input", 1, progress_total)
    media_duration = probe_duration_seconds(input_path)
    _emit(progress, started, "extract_audio", "Extracting audio with ffmpeg", 2, progress_total)
    extract_audio(input_path, wav_path)
    audio_duration = wav_duration_seconds(wav_path)
    _emit(progress, started, "probe_audio", "Audio extracted", 3, progress_total)
    silence_threshold_db = options.silence_threshold_db
    auto_silence_threshold = None
    silences = []
    spans = []
    pyannote_vad = None
    pyannote_no_speech = False
    if options.vad_pre_split and options.vad_engine == "ffmpeg":
        if options.auto_silence_threshold:
            _emit(progress, started, "auto_silence_threshold", "Analyzing audio levels", 4, progress_total)
            auto_silence_threshold = estimate_silence_threshold(wav_path)
            silence_threshold_db = auto_silence_threshold.threshold_db
        _emit(progress, started, "detect_silence", "Detecting silence spans", 5, progress_total)
        silences = detect_silences(wav_path, silence_threshold_db, options.min_silence_duration_s)
        if silences:
            spans = speech_spans_from_silences(
                audio_duration,
                silences,
                options.vad_min_speech_duration_s,
                options.vad_max_segment_duration_s,
                options.vad_padding_s,
                options.vad_merge_gap_s,
            )
    elif options.vad_pre_split and options.vad_engine == "pyannote":
        _emit(progress, started, "load_vad", "Loading pyannote VAD model", 4, progress_total)
        try:
            pyannote_vad = detect_speech_spans_pyannote(
                wav_path,
                model_name=options.pyannote_model,
                device=options.model_device,
                min_speech_duration_s=options.vad_min_speech_duration_s,
                min_silence_duration_s=options.min_silence_duration_s,
                progress=lambda current, total: _emit(
                    progress,
                    started,
                    "pyannote_vad",
                    "Detecting speech with pyannote",
                    current,
                    total,
                ),
            )
        except PyannoteVadError as exc:
            _emit(progress, started, "vad_error", "pyannote VAD could not start", 1, 1)
            return ProcessResult(
                input_path=input_path,
                output_dir=output_dir,
                wav_path=wav_path,
                ja_srt_path=None,
                ko_srt_path=None,
                copied_ko_srt_path=None,
                status="vad_error",
                message=str(exc),
            )
        spans = normalize_speech_spans(
            audio_duration,
            pyannote_vad.speech_spans,
            options.vad_min_speech_duration_s,
            options.vad_max_segment_duration_s,
            options.vad_padding_s,
            options.vad_merge_gap_s,
        )
        vad_json_path.write_text(
            json.dumps(
                {
                    "engine": "pyannote",
                    "model": pyannote_vad.model_name,
                    "model_source": pyannote_vad.model_source,
                    "model_revision": pyannote_vad.model_revision,
                    "device": pyannote_vad.device,
                    "pyannote_audio_version": pyannote_vad.pyannote_audio_version,
                    "processed_audio_duration_s": pyannote_vad.processed_audio_duration_s,
                    "chunk_duration_s": pyannote_vad.chunk_duration_s,
                    "min_speech_duration_s": options.vad_min_speech_duration_s,
                    "min_silence_duration_s": options.min_silence_duration_s,
                    "max_segment_duration_s": options.vad_max_segment_duration_s,
                    "padding_s": options.vad_padding_s,
                    "merge_gap_s": options.vad_merge_gap_s,
                    "raw_speech_spans": _spans_to_json(pyannote_vad.speech_spans),
                    "transcription_spans": _spans_to_json(spans),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pyannote_no_speech = not spans
        _emit(progress, started, "detect_speech", f"pyannote found {len(spans)} speech span(s)", 5, progress_total)
    elif options.vad_engine not in {"ffmpeg", "pyannote"}:
        raise ValueError(f"Unsupported VAD engine: {options.vad_engine}")
    if pyannote_no_speech:
        _emit(progress, started, "load_model", f"No speech found; skipping {options.asr_backend} model", 6, progress_total)
    else:
        _emit(progress, started, "load_model", f"Loading {options.asr_backend} ASR model", 6, progress_total)

    transcriber = None
    if not pyannote_no_speech:
        transcriber = create_transcriber(options)
        try:
            transcriber.load()
        except TranscriptionDependencyError as exc:
            message = (
                f"Standalone extracted audio to {wav_path}. "
                f"Audio duration: {audio_duration:.3f} seconds. "
                f"{exc}"
            )
            _emit(progress, started, "dependency", "Transcription dependency required", 1, 1)
            return ProcessResult(
                input_path=input_path,
                output_dir=output_dir,
                wav_path=wav_path,
                ja_srt_path=None,
                ko_srt_path=None,
                copied_ko_srt_path=None,
                status="missing_transcription_dependency",
                message=message,
            )

    segment_count = 1
    transcription = None
    if pyannote_no_speech:
        raw_chunks = []
        raw_output = {"chunks": []}
        segment_count = 0
    elif spans:
        assert transcriber is not None
        _emit(progress, started, "transcribe", "Transcribing VAD segments", 7, progress_total)
        raw_chunks = []
        segment_count = len(spans)
        for index, span in enumerate(spans, 1):
            _emit(progress, started, "transcribe_segments", f"Transcribing VAD segment {index}/{len(spans)}", index, len(spans))
            segment_path = output_dir / f"{input_path.stem}.segment.{index:04d}.wav"
            try:
                extract_audio_segment(wav_path, segment_path, span.start, span.end)
                transcription = transcriber.transcribe(str(segment_path))
                raw_chunks.extend(offset_raw_chunks(extract_raw_chunks(transcription.raw), span.start))
            finally:
                segment_path.unlink(missing_ok=True)
        raw_output = {"chunks": raw_chunks}
    else:
        assert transcriber is not None
        _emit(progress, started, "transcribe", "Transcribing audio", 7, progress_total)
        transcription = transcriber.transcribe(str(wav_path))
        raw_chunks = extract_raw_chunks(transcription.raw)
        raw_output = transcription.raw

    if transcription is None and not pyannote_no_speech:
        assert transcriber is not None
        _emit(progress, started, "transcribe", "Transcribing audio", 7, progress_total)
        transcription = transcriber.transcribe(str(wav_path))
        raw_chunks = extract_raw_chunks(transcription.raw)
        raw_output = transcription.raw
    _emit(progress, started, "postprocess", "Writing Japanese subtitles", 8, progress_total)
    chunks = group_chunks_by_timing(normalize_chunks(raw_chunks))
    quality_issues = []
    if options.report_subtitle_quality or options.drop_likely_hallucinations:
        quality_issues = analyze_subtitle_quality(chunks, wav_path)
    dropped_quality_issues = []
    if options.drop_likely_hallucinations:
        chunks, dropped_quality_issues = filter_likely_hallucinations(chunks)
    tail_refine_quality_issues = []
    tail_refine_attempts: list[dict[str, Any]] = []
    if options.tail_retranscribe_long_subtitles and chunks:
        assert transcriber is not None
        tail_source_issues = analyze_subtitle_quality(chunks, wav_path)
        chunks, tail_refine_quality_issues, tail_refine_attempts = tail_retranscribe_long_subtitles(
            chunks,
            tail_source_issues,
            wav_path,
            output_dir,
            input_path.stem,
            transcriber,
            max_candidates=options.tail_retranscribe_max_candidates,
        )
    split_quality_issues = []
    if options.split_long_subtitles:
        split_source_issues = analyze_subtitle_quality(chunks, wav_path)
        chunks, split_quality_issues = split_long_subtitle_candidates(chunks, split_source_issues)
    if (
        options.report_subtitle_quality
        or options.drop_likely_hallucinations
        or options.tail_retranscribe_long_subtitles
        or options.split_long_subtitles
        or options.annotate_subtitle_quality
    ):
        quality_issues = analyze_subtitle_quality(chunks, wav_path)
        quality_json_path.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "wav_path": str(wav_path),
                    "subtitle_count": len(chunks),
                    "dropped_count": len(dropped_quality_issues),
                    "tail_refine_count": len(tail_refine_quality_issues),
                    "split_count": len(split_quality_issues),
                    "issue_count": len(quality_issues),
                    "dropped": quality_issues_to_json(dropped_quality_issues),
                    "tail_refine": quality_issues_to_json(tail_refine_quality_issues),
                    "tail_refine_attempts": tail_refine_attempts,
                    "split": quality_issues_to_json(split_quality_issues),
                    "issues": quality_issues_to_json(quality_issues),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    srt_chunks = annotate_chunks_with_quality(chunks, quality_issues) if options.annotate_subtitle_quality else chunks
    ja_srt_path.write_text(chunks_to_srt(srt_chunks), encoding="utf-8")
    ja_txt_path.write_text(chunks_to_txt(chunks), encoding="utf-8")
    ja_aligned_srt_path = None
    whisperx_metadata = None
    if options.alignment_engine == "whisperx":
        _emit(progress, started, "whisperx_align", "Aligning subtitle timings with WhisperX", 9, progress_total)
        try:
            aligned = align_chunks_with_whisperx(
                chunks,
                wav_path,
                language_code=_whisperx_language_code(options.language),
                device=options.model_device,
                model_name=options.whisperx_align_model,
            )
        except WhisperXAlignmentError as exc:
            _emit(progress, started, "alignment_error", "WhisperX alignment failed", 1, 1)
            return ProcessResult(
                input_path=input_path,
                output_dir=output_dir,
                wav_path=wav_path,
                ja_srt_path=ja_srt_path,
                ko_srt_path=None,
                copied_ko_srt_path=None,
                status="alignment_error",
                message=str(exc),
            )
        whisperx_srt_path.write_text(chunks_to_srt(aligned.chunks), encoding="utf-8")
        write_alignment_metadata(whisperx_json_path, aligned.metadata)
        ja_aligned_srt_path = whisperx_srt_path
        whisperx_metadata = aligned.metadata
    elif options.alignment_engine != "none":
        raise ValueError(f"Unsupported alignment engine: {options.alignment_engine}")
    raw_json_path.write_text(json.dumps(raw_output, ensure_ascii=False, indent=2), encoding="utf-8")
    process_json_path.write_text(
        json.dumps(
            {
                "status": "success",
                "input_path": str(input_path),
                "wav_path": str(wav_path),
                "media_duration_seconds": media_duration,
                "audio_duration_seconds": audio_duration,
                "device": transcription.device_name if transcription is not None else None,
                "torch_version": transcription.torch_version if transcription is not None else None,
                "torch_cuda_version": transcription.torch_cuda_version if transcription is not None else None,
                "batch_size_used": transcription.batch_size_used if transcription is not None else None,
                "word_timestamps_used": transcription.word_timestamps_used if transcription is not None else False,
                "asr_backend": options.asr_backend,
                "asr_model": options.qwen_model_name if options.asr_backend == "qwen3" else options.model_name,
                "qwen_aligner_model": options.qwen_aligner_model if options.asr_backend == "qwen3" else None,
                "qwen_return_timestamps": options.qwen_return_timestamps if options.asr_backend == "qwen3" else None,
                "vad_engine": options.vad_engine,
                "silence_threshold_db": silence_threshold_db if options.vad_engine == "ffmpeg" else None,
                "min_silence_duration_s": options.min_silence_duration_s,
                "auto_silence_threshold": (
                    {
                        "threshold_db": auto_silence_threshold.threshold_db,
                        "noise_floor_db": auto_silence_threshold.noise_floor_db,
                        "speech_level_db": auto_silence_threshold.speech_level_db,
                        "analyzed_frame_count": auto_silence_threshold.analyzed_frame_count,
                    }
                    if auto_silence_threshold is not None
                    else None
                ),
                "detected_silence_count": len(silences) if options.vad_engine == "ffmpeg" else None,
                "detected_speech_count": len(spans) if options.vad_engine == "pyannote" else None,
                "pyannote_vad": (
                    {
                        "model": pyannote_vad.model_name,
                        "model_source": pyannote_vad.model_source,
                        "model_revision": pyannote_vad.model_revision,
                        "device": pyannote_vad.device,
                        "pyannote_audio_version": pyannote_vad.pyannote_audio_version,
                        "processed_audio_duration_s": pyannote_vad.processed_audio_duration_s,
                        "chunk_duration_s": pyannote_vad.chunk_duration_s,
                    }
                    if pyannote_vad is not None
                    else None
                ),
                "vad_report": str(vad_json_path) if pyannote_vad is not None else None,
                "vad_pre_split": options.vad_pre_split,
                "transcription_segment_count": segment_count,
                "subtitle_count": len(chunks),
                "alignment_engine": options.alignment_engine,
                "whisperx_aligned_srt": str(ja_aligned_srt_path) if ja_aligned_srt_path is not None else None,
                "whisperx_alignment_report": str(whisperx_json_path) if whisperx_metadata is not None else None,
                "whisperx_changed_subtitle_count": whisperx_metadata.get("changed_count") if whisperx_metadata else None,
                "subtitle_quality_report": str(quality_json_path)
                if (
                    options.report_subtitle_quality
                    or options.drop_likely_hallucinations
                    or options.tail_retranscribe_long_subtitles
                    or options.split_long_subtitles
                    or options.annotate_subtitle_quality
                )
                else None,
                "dropped_likely_hallucination_count": len(dropped_quality_issues),
                "tail_refined_subtitle_count": len(tail_refine_quality_issues),
                "split_long_subtitle_count": len(split_quality_issues),
                "annotated_subtitle_quality": options.annotate_subtitle_quality,
                "subtitle_quality_issue_count": len(quality_issues),
                "processing_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ko_srt_path = None
    copied_ko_srt_path = None
    if options.translate:
        _emit(progress, started, "translate_prepare", "Preparing Korean translation", 1, 2)
        translation = translate_srt(
            ja_srt_path,
            translation_options_from_process(options),
            progress=progress,
        )
        ko_srt_path = translation.output_srt
        copied_ko_srt_path = copy_korean_srt_to_media_dir(input_path, ko_srt_path)
        _emit(progress, started, "translate_done", "Korean translation completed", 2, 2)

    message = f"Standalone transcription completed: {ja_srt_path}"
    if ko_srt_path is not None:
        message = f"{message}; Korean translation completed: {ko_srt_path}"
    if copied_ko_srt_path is not None:
        message = f"{message}; copied Korean subtitle to: {copied_ko_srt_path}"
    if ja_aligned_srt_path is not None:
        message = f"{message}; WhisperX aligned Japanese subtitle: {ja_aligned_srt_path}"
    _emit(progress, started, "done", "Standalone transcription completed", progress_total, progress_total)
    return ProcessResult(
        input_path=input_path,
        output_dir=output_dir,
        wav_path=wav_path,
        ja_srt_path=ja_srt_path,
        ko_srt_path=ko_srt_path,
        copied_ko_srt_path=copied_ko_srt_path,
        status="success",
        message=message,
        ja_aligned_srt_path=ja_aligned_srt_path,
    )


def translated_srt_path(input_srt: Path) -> Path:
    if input_srt.name.endswith(".ja.srt"):
        return input_srt.with_name(input_srt.name.removesuffix(".ja.srt") + ".ko.srt")
    return input_srt.with_name(f"{input_srt.stem}.ko.srt")


def copy_korean_srt_to_media_dir(input_path: Path, ko_srt_path: Path) -> Path:
    primary_target = input_path.with_suffix(".srt")
    target = primary_target if not primary_target.exists() else input_path.with_suffix(".ko.srt")
    shutil.copy2(ko_srt_path, target)
    return target


def tail_retranscribe_long_subtitles(
    chunks: list[SubtitleChunk],
    issues: list[SubtitleQualityIssue],
    wav_path: Path,
    output_dir: Path,
    stem: str,
    transcriber: KotobaTranscriber,
    tail_window_s: float = 5.0,
    min_similarity: float = 0.45,
    min_tail_chars: int = 4,
    min_tail_ratio: float = 0.35,
    max_candidates: int = 20,
) -> tuple[list[SubtitleChunk], list[SubtitleQualityIssue], list[dict[str, Any]]]:
    candidate_indexes = tail_retranscribe_candidate_indexes(issues)
    if not candidate_indexes:
        return chunks, [], []
    candidate_order = [
        issue.index
        for issue in sorted(issues, key=_tail_retranscribe_priority)
        if issue.index in candidate_indexes
    ]
    if max_candidates > 0:
        candidate_order = candidate_order[:max_candidates]
    candidate_indexes = set(candidate_order)
    refinements: dict[int, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, 1):
        if index not in candidate_indexes:
            continue
        tail_start = max(chunk.start, chunk.end - tail_window_s)
        tail_path = output_dir / f"{stem}.tail-refine.{index:04d}.wav"
        try:
            extract_audio_segment(wav_path, tail_path, tail_start, chunk.end)
            tail_result = transcriber.transcribe(str(tail_path))
            tail_chunks = group_chunks_by_timing(normalize_chunks(extract_raw_chunks(tail_result.raw)))
        finally:
            tail_path.unlink(missing_ok=True)
        tail_text = "".join(tail_chunk.text for tail_chunk in tail_chunks)
        similarity = text_similarity(chunk.text, tail_text)
        source_len = len(normalize_phrase(chunk.text))
        tail_len = len(normalize_phrase(tail_text))
        tail_ratio = tail_len / source_len if source_len else 0.0
        accepted = bool(
            tail_chunks
            and tail_len >= min_tail_chars
            and tail_ratio >= min_tail_ratio
            and similarity >= min_similarity
        )
        new_start = tail_start + tail_chunks[0].start if accepted else None
        attempt = {
            "index": index,
            "start": chunk.start,
            "end": chunk.end,
            "tail_start": tail_start,
            "text": chunk.text,
            "tail_text": tail_text,
            "tail_text_length": tail_len,
            "tail_text_ratio": round(tail_ratio, 3),
            "similarity": round(similarity, 3),
            "accepted": accepted,
            "new_start": round(new_start, 3) if new_start is not None else None,
        }
        attempts.append(attempt)
        if accepted and new_start is not None:
            refinements[index] = {
                "new_start": new_start,
                "tail_start": tail_start,
                "similarity": similarity,
            }
    refined_chunks, applied = apply_tail_refinements(chunks, refinements)
    return refined_chunks, applied, attempts


def _tail_retranscribe_priority(issue: SubtitleQualityIssue) -> tuple[int, float, int]:
    action_priority = 0 if issue.recommended_action == "resegment_candidate" else 1
    return (action_priority, -issue.duration, issue.index)


def _spans_to_json(spans: list[Any]) -> list[dict[str, float]]:
    return [
        {
            "start": round(float(span.start), 3),
            "end": round(float(span.end), 3),
            "duration": round(float(span.end - span.start), 3),
        }
        for span in spans
    ]


def _whisperx_language_code(language: str) -> str:
    normalized = language.strip().lower()
    return {
        "ja": "ja",
        "japanese": "ja",
        "jp": "ja",
        "ko": "ko",
        "korean": "ko",
        "en": "en",
        "english": "en",
        "zh": "zh",
        "chinese": "zh",
    }.get(normalized, normalized)


def validate_silence_threshold(value: str) -> str:
    if not re.match(r"^-?\d+(\.\d+)?dB$", value):
        raise ValueError("silence threshold must look like -42dB")
    return value


def translation_options_from_process(options: ProcessOptions, output: Path | None = None) -> TranslationOptions:
    return TranslationOptions(
        model=options.translation_model,
        output=output,
        ollama_host=options.ollama_host,
        ollama_port=options.ollama_port,
        korean_style=options.korean_style,
    )


def create_transcriber(options: ProcessOptions) -> KotobaTranscriber | Qwen3Transcriber:
    if options.asr_backend == "kotoba":
        return KotobaTranscriber(options)
    if options.asr_backend == "qwen3":
        return Qwen3Transcriber(options)
    raise ValueError(f"Unsupported ASR backend: {options.asr_backend}")


def _emit(
    callback: ProgressCallback | None,
    started: float,
    stage: str,
    message: str,
    current: int,
    total: int,
) -> None:
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=(current / total) * 100 if total else None,
            elapsed_seconds=time.time() - started,
        )
    )
