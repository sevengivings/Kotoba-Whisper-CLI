from __future__ import annotations

import re
import time
import json
import shutil
from pathlib import Path

from kotoba_standalone.media import (
    detect_silences,
    estimate_silence_threshold,
    extract_audio,
    extract_audio_segment,
    is_supported_media,
    probe_duration_seconds,
    speech_spans_from_silences,
    wav_duration_seconds,
)
from kotoba_standalone.progress import ProgressCallback
from kotoba_standalone.subtitle import chunks_to_srt, chunks_to_txt, group_chunks_by_timing, normalize_chunks
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
    wav_path = output_dir / f"{input_path.stem}.standalone.wav"

    progress_total = 11 if options.translate else 10
    _emit(progress, started, "prepare", "Preparing input", 1, progress_total)
    media_duration = probe_duration_seconds(input_path)
    _emit(progress, started, "extract_audio", "Extracting audio with bundled ffmpeg", 2, progress_total)
    extract_audio(input_path, wav_path)
    audio_duration = wav_duration_seconds(wav_path)
    _emit(progress, started, "probe_audio", "Audio extracted", 3, progress_total)
    silence_threshold_db = options.silence_threshold_db
    auto_silence_threshold = None
    if options.auto_silence_threshold:
        _emit(progress, started, "auto_silence_threshold", "Analyzing audio levels", 4, progress_total)
        auto_silence_threshold = estimate_silence_threshold(wav_path)
        silence_threshold_db = auto_silence_threshold.threshold_db
    _emit(progress, started, "detect_silence", "Detecting silence spans", 5, progress_total)
    silences = detect_silences(wav_path, silence_threshold_db, options.min_silence_duration_s)
    spans = []
    if options.vad_pre_split and silences:
        spans = speech_spans_from_silences(
            audio_duration,
            silences,
            options.vad_min_speech_duration_s,
            options.vad_max_segment_duration_s,
            options.vad_padding_s,
            options.vad_merge_gap_s,
        )
    _emit(progress, started, "load_model", "Loading Kotoba model", 6, progress_total)

    transcriber = KotobaTranscriber(options)
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
    if spans:
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
        _emit(progress, started, "transcribe", "Transcribing audio", 7, progress_total)
        transcription = transcriber.transcribe(str(wav_path))
        raw_chunks = extract_raw_chunks(transcription.raw)
        raw_output = transcription.raw

    if transcription is None:
        _emit(progress, started, "transcribe", "Transcribing audio", 7, progress_total)
        transcription = transcriber.transcribe(str(wav_path))
        raw_chunks = extract_raw_chunks(transcription.raw)
        raw_output = transcription.raw
    _emit(progress, started, "postprocess", "Writing Japanese subtitles", 8, progress_total)
    chunks = group_chunks_by_timing(normalize_chunks(raw_chunks))
    ja_srt_path.write_text(chunks_to_srt(chunks), encoding="utf-8")
    ja_txt_path.write_text(chunks_to_txt(chunks), encoding="utf-8")
    raw_json_path.write_text(json.dumps(raw_output, ensure_ascii=False, indent=2), encoding="utf-8")
    process_json_path.write_text(
        json.dumps(
            {
                "status": "success",
                "input_path": str(input_path),
                "wav_path": str(wav_path),
                "media_duration_seconds": media_duration,
                "audio_duration_seconds": audio_duration,
                "device": transcription.device_name,
                "torch_version": transcription.torch_version,
                "torch_cuda_version": transcription.torch_cuda_version,
                "batch_size_used": transcription.batch_size_used,
                "word_timestamps_used": transcription.word_timestamps_used,
                "silence_threshold_db": silence_threshold_db,
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
                "detected_silence_count": len(silences),
                "vad_pre_split": options.vad_pre_split,
                "transcription_segment_count": segment_count,
                "subtitle_count": len(chunks),
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
