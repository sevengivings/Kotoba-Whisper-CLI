from __future__ import annotations

import re
import time
import json
from pathlib import Path

from kotoba_standalone.media import extract_audio, is_supported_media, probe_duration_seconds, wav_duration_seconds
from kotoba_standalone.progress import ProgressCallback
from kotoba_standalone.subtitle import chunks_to_json, chunks_to_srt, chunks_to_txt, group_chunks_by_timing, normalize_chunks
from kotoba_standalone.transcriber import KotobaTranscriber, TranscriptionDependencyError, extract_raw_chunks
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

    progress_total = 7
    _emit(progress, started, "prepare", "Preparing input", 1, progress_total)
    media_duration = probe_duration_seconds(input_path)
    _emit(progress, started, "extract_audio", "Extracting audio with bundled ffmpeg", 2, progress_total)
    extract_audio(input_path, wav_path)
    audio_duration = wav_duration_seconds(wav_path)
    _emit(progress, started, "probe_audio", "Audio extracted", 3, progress_total)
    _emit(progress, started, "load_model", "Loading Kotoba model", 4, progress_total)

    transcriber = KotobaTranscriber(options)
    try:
        transcriber.load()
    except TranscriptionDependencyError as exc:
        message = (
            f"Standalone extracted audio to {wav_path}. "
            f"Audio duration: {audio_duration:.3f} seconds. "
            f"{exc}"
        )
        _emit(progress, started, "dependency", "Transcription dependency required", progress_total, progress_total)
        return ProcessResult(
            input_path=input_path,
            output_dir=output_dir,
            wav_path=wav_path,
            ja_srt_path=None,
            ko_srt_path=None,
            status="missing_transcription_dependency",
            message=message,
        )

    _emit(progress, started, "transcribe", "Transcribing audio", 5, progress_total)
    transcription = transcriber.transcribe(str(wav_path))
    _emit(progress, started, "postprocess", "Writing Japanese subtitles", 6, progress_total)
    chunks = group_chunks_by_timing(normalize_chunks(extract_raw_chunks(transcription.raw)))
    ja_srt_path.write_text(chunks_to_srt(chunks), encoding="utf-8")
    ja_txt_path.write_text(chunks_to_txt(chunks), encoding="utf-8")
    raw_json_path.write_text(json.dumps(transcription.raw, ensure_ascii=False, indent=2), encoding="utf-8")
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
                "subtitle_count": len(chunks),
                "processing_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    message = (
        f"Standalone transcription completed: {ja_srt_path}"
    )
    _emit(progress, started, "done", "Standalone transcription completed", progress_total, progress_total)
    return ProcessResult(
        input_path=input_path,
        output_dir=output_dir,
        wav_path=wav_path,
        ja_srt_path=ja_srt_path,
        ko_srt_path=None,
        status="success",
        message=message,
    )


def translated_srt_path(input_srt: Path) -> Path:
    if input_srt.name.endswith(".ja.srt"):
        return input_srt.with_name(input_srt.name.removesuffix(".ja.srt") + ".ko.srt")
    return input_srt.with_name(f"{input_srt.stem}.ko.srt")


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
