from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from kotoba_standalone.alignment import (
    WhisperXAlignmentError,
    align_chunks_with_whisperx,
    write_alignment_metadata,
)
from kotoba_standalone.media import FFMPEG_PATH_ENV, FFmpegAudioExtractionError, is_supported_media
from kotoba_standalone.pipeline import process_video, validate_silence_threshold
from kotoba_standalone.progress import tqdm_progress
from kotoba_standalone.settings import DEFAULT_TRANSLATION_MODEL, load_saved_translation_model, save_translation_model
from kotoba_standalone.subtitle import chunks_to_srt, parse_srt_chunks
from kotoba_standalone.translate.ollama import (
    OllamaModelError,
    OllamaUnavailableError,
    format_ollama_model_choice,
    get_ollama_models,
    sort_ollama_models_for_translation,
    translate_srt,
)
from kotoba_standalone.types import ProcessOptions, ProcessResult, ProgressEvent, TranslationOptions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    old_ffmpeg_path = os.environ.get(FFMPEG_PATH_ENV)
    ffmpeg_path_was_applied = apply_ffmpeg_path_arg(args)
    try:
        if args.command == "process" and args.asr_backend == "qwen3":
            reexec_result = maybe_reexec_with_qwen_environment(argv)
            if reexec_result is not None:
                return reexec_result
        if args.command == "process":
            return run_process(args)
        if args.command == "translate":
            return run_translate(args)
        if args.command == "align":
            return run_align(args)
        parser.print_help()
        return 1
    finally:
        if ffmpeg_path_was_applied:
            restore_ffmpeg_path_env(old_ffmpeg_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kotoba", description="Kotoba standalone CLI.")
    subparsers = parser.add_subparsers(dest="command")

    process = subparsers.add_parser("process", help="Process one media file, or every media file in a folder.")
    process.add_argument("input", type=Path)
    process.add_argument("--output-dir", type=Path)
    process.add_argument("--language", default="japanese")
    process.add_argument("--silence-threshold-db", default="-42dB")
    process.add_argument("--min-silence-duration-s", type=float, default=0.5)
    process.add_argument("--auto-silence-threshold", action="store_true")
    process.add_argument("--no-vad-pre-split", dest="vad_pre_split", action="store_false")
    process.set_defaults(vad_pre_split=True)
    process.add_argument("--vad-engine", choices=("ffmpeg", "pyannote"), default="pyannote")
    process.add_argument("--pyannote-model", default="pyannote/segmentation-3.0")
    process.add_argument("--vad-max-segment-duration-s", type=float, default=30.0)
    process.add_argument("--vad-min-speech-duration-s", type=float, default=0.25)
    process.add_argument("--vad-padding-s", type=float, default=0.4)
    process.add_argument("--vad-merge-gap-s", type=float, default=0.0)
    process.add_argument("--ffmpeg-path", help=argparse.SUPPRESS)
    process.add_argument("--asr-backend", choices=("kotoba", "faster-kotoba", "qwen3"), default="kotoba", help=argparse.SUPPRESS)
    process.add_argument("--batch-size", type=int, default=8)
    process.add_argument("--chunk-length-s", type=int, default=15)
    process.add_argument("--model-name", default="kotoba-tech/kotoba-whisper-v2.2")
    process.add_argument("--model-device", default="cuda:0")
    process.add_argument("--model-dtype", default="float16")
    process.add_argument("--qwen-model-name", default="Qwen/Qwen3-ASR-1.7B", help=argparse.SUPPRESS)
    process.add_argument("--qwen-aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B", help=argparse.SUPPRESS)
    process.add_argument("--no-qwen-timestamps", dest="qwen_return_timestamps", action="store_false", help=argparse.SUPPRESS)
    process.add_argument("--report-subtitle-quality", action="store_true")
    process.add_argument("--drop-likely-hallucinations", action="store_true")
    process.add_argument("--split-long-subtitles", action="store_true")
    process.add_argument("--annotate-subtitle-quality", action="store_true")
    process.add_argument("--tail-retranscribe-long-subtitles", action="store_true")
    process.add_argument("--tail-retranscribe-max-candidates", type=int, default=20)
    process.add_argument("--whisperx-align", dest="alignment_engine", action="store_const", const="whisperx", default="none")
    process.add_argument("--whisperx-align-model")
    process.add_argument("--translate", action="store_true")
    process.add_argument("--translation-model")
    process.add_argument("--translation-model-choice", action="store_true")
    process.add_argument("--ollama-host", default="localhost")
    process.add_argument("--ollama-port", type=int, default=11434)
    process.add_argument("--korean-style", choices=("polite", "banmal", "strict-banmal"), default="polite")

    translate = subparsers.add_parser("translate", help="Translate one SRT file with Ollama.")
    translate.add_argument("input_srt", type=Path)
    translate.add_argument("--output", type=Path)
    translate.add_argument("--source", default="japanese")
    translate.add_argument("--target", default="korean")
    translate.add_argument("--model")
    translate.add_argument("--model-choice", action="store_true")
    translate.add_argument("--ollama-host", default="localhost")
    translate.add_argument("--ollama-port", type=int, default=11434)
    translate.set_defaults(batch_translate=True)
    translate.add_argument("--batch-translate", dest="batch_translate", action="store_true")
    translate.add_argument("--no-batch-translate", dest="batch_translate", action="store_false")
    translate.add_argument("--batch-size", type=int, default=50)
    translate.add_argument("--text-split-size", type=int, default=0)
    translate.add_argument("--timeout-seconds", type=int, default=600)
    translate.add_argument("--korean-style", choices=("polite", "banmal", "strict-banmal"), default="polite")

    align = subparsers.add_parser("align", help="Align an existing Japanese SRT with an existing WAV using WhisperX.")
    align.add_argument("input_srt", type=Path)
    align.add_argument("input_wav", type=Path)
    align.add_argument("--output", type=Path)
    align.add_argument("--language", default="japanese")
    align.add_argument("--model-device", default="cuda:0")
    align.add_argument("--whisperx-align-model")
    return parser


def apply_ffmpeg_path_arg(args: argparse.Namespace) -> bool:
    ffmpeg_path = getattr(args, "ffmpeg_path", None)
    if ffmpeg_path:
        os.environ[FFMPEG_PATH_ENV] = ffmpeg_path
        return True
    return False


def restore_ffmpeg_path_env(old_ffmpeg_path: str | None) -> None:
    if old_ffmpeg_path is None:
        os.environ.pop(FFMPEG_PATH_ENV, None)
    else:
        os.environ[FFMPEG_PATH_ENV] = old_ffmpeg_path


def maybe_reexec_with_qwen_environment(argv: list[str] | None) -> int | None:
    if os.environ.get("KOTOBA_QWEN_REEXEC") == "1" or qwen_dependencies_available():
        return None
    qwen_python = qwen_environment_python()
    if qwen_python is None:
        print_error(
            "Qwen3-ASR experimental environment was not found.\n"
            "  Install it first:\n"
            "    uv sync --group cuda --group pyannote --group qwen\n"
            "  Korean guide: Qwen3 실험 환경(.venv-qwen)을 먼저 설치해 주세요."
        )
        return 1
    command = [str(qwen_python), "-m", "kotoba_standalone.cli", *(argv if argv is not None else sys.argv[1:])]
    env = os.environ.copy()
    env["KOTOBA_QWEN_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(command, cwd=standalone_root(), env=env).returncode


def qwen_dependencies_available() -> bool:
    return importlib.util.find_spec("qwen_asr") is not None


def qwen_environment_python() -> Path | None:
    env_path = os.environ.get("KOTOBA_QWEN_PYTHON")
    candidates = [Path(env_path).expanduser()] if env_path else []
    root = standalone_root()
    if sys.platform == "win32":
        candidates.append(root / ".venv-qwen" / "Scripts" / "python.exe")
    else:
        candidates.append(root / ".venv-qwen" / "bin" / "python")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def standalone_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_process(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    try:
        if args.translate or args.translation_model or args.translation_model_choice:
            translation_model = resolve_translation_model(
                requested_model=args.translation_model,
                choose=args.translation_model_choice,
                ollama_host=args.ollama_host,
                ollama_port=args.ollama_port,
            )
        else:
            translation_model = DEFAULT_TRANSLATION_MODEL
    except (OllamaModelError, OllamaUnavailableError) as exc:
        print_error(str(exc))
        return 1
    options = ProcessOptions(
        output_dir=args.output_dir,
        language=args.language,
        silence_threshold_db=validate_silence_threshold(args.silence_threshold_db),
        min_silence_duration_s=args.min_silence_duration_s,
        auto_silence_threshold=args.auto_silence_threshold,
        vad_pre_split=args.vad_pre_split,
        vad_engine=args.vad_engine,
        pyannote_model=args.pyannote_model,
        vad_max_segment_duration_s=args.vad_max_segment_duration_s,
        vad_min_speech_duration_s=args.vad_min_speech_duration_s,
        vad_padding_s=args.vad_padding_s,
        vad_merge_gap_s=args.vad_merge_gap_s,
        asr_backend=args.asr_backend,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        model_name=args.model_name,
        model_device=args.model_device,
        model_dtype=args.model_dtype,
        qwen_model_name=args.qwen_model_name,
        qwen_aligner_model=args.qwen_aligner_model,
        qwen_return_timestamps=args.qwen_return_timestamps,
        report_subtitle_quality=args.report_subtitle_quality,
        drop_likely_hallucinations=args.drop_likely_hallucinations,
        split_long_subtitles=args.split_long_subtitles,
        annotate_subtitle_quality=args.annotate_subtitle_quality,
        tail_retranscribe_long_subtitles=args.tail_retranscribe_long_subtitles,
        tail_retranscribe_max_candidates=args.tail_retranscribe_max_candidates,
        alignment_engine=args.alignment_engine,
        whisperx_align_model=args.whisperx_align_model,
        translate=args.translate,
        translation_model=translation_model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        korean_style=args.korean_style,
    )

    if input_path.is_dir():
        media_files = list(iter_media_files(input_path))
        if not media_files:
            print_error(f"No supported media files found in {input_path}")
            return 1
        print(f"Found {len(media_files)} media file(s) in {input_path}")
        failed = 0
        for index, media_file in enumerate(media_files, 1):
            print(f"\nProcessing {index}/{len(media_files)}: {media_file.name}")
            try:
                with tqdm_progress() as progress:
                    result = process_video(media_file, options, progress=progress)
                if result.status != "success":
                    failed += 1
                print_process_result(result)
            except (FFmpegAudioExtractionError, OllamaModelError, OllamaUnavailableError, Exception) as exc:
                failed += 1
                print_error(f"Failed {media_file.name}: {exc}")
            else:
                if options.translate and result.ko_srt_path is not None:
                    save_translation_model(options.translation_model)
        print_directory_summary(len(media_files), failed)
        return 0 if failed == 0 else 1

    with tqdm_progress() as progress:
        try:
            result = process_video(input_path, options, progress=progress)
        except (FFmpegAudioExtractionError, OllamaModelError, OllamaUnavailableError) as exc:
            print_error(str(exc))
            return 1
    print_process_result(result)
    if options.translate and result.ko_srt_path is not None:
        save_translation_model(options.translation_model)
    return 0 if result.status == "success" else 1


def run_translate(args: argparse.Namespace) -> int:
    try:
        model = resolve_translation_model(
            requested_model=args.model,
            choose=args.model_choice,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            timeout_seconds=args.timeout_seconds,
        )
    except (OllamaModelError, OllamaUnavailableError) as exc:
        print_error(str(exc))
        return 1
    options = TranslationOptions(
        model=model,
        output=args.output,
        source=args.source,
        target=args.target,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        batch_translate=args.batch_translate,
        batch_size=args.batch_size,
        text_split_size=args.text_split_size,
        timeout_seconds=args.timeout_seconds,
        korean_style=args.korean_style,
    )
    print(f"Translating: {args.input_srt}")
    print(f"Model: {model}")
    try:
        with tqdm_progress() as progress:
            result = translate_srt(args.input_srt, options, progress=progress)
    except (OllamaModelError, OllamaUnavailableError) as exc:
        print_error(str(exc))
        return 1
    save_translation_model(model)
    print("Done.")
    print(f"  Korean SRT: {result.output_srt}")
    print(f"  Metadata: {result.metadata_path}")
    print(f"  Subtitles: {result.subtitle_count}")
    print(f"  Elapsed: {result.processing_seconds:.3f}s")
    return 0


def run_align(args: argparse.Namespace) -> int:
    input_srt = args.input_srt.expanduser().resolve()
    input_wav = args.input_wav.expanduser().resolve()
    output_srt = args.output or input_srt.with_name(input_srt.name.removesuffix(".srt") + ".whisperx.srt")
    metadata_path = output_srt.with_suffix(".json")
    try:
        chunks = parse_srt_chunks(input_srt.read_text(encoding="utf-8-sig"))
        with tqdm_progress() as progress:
            result = align_chunks_with_whisperx(
                chunks,
                input_wav,
                language_code=whisperx_language_code(args.language),
                device=args.model_device,
                model_name=args.whisperx_align_model,
            )
            progress(
                ProgressEvent(
                    stage="align",
                    message="WhisperX alignment completed",
                    current=len(result.chunks),
                    total=len(chunks),
                    percent=100.0,
                    elapsed_seconds=None,
                )
            )
    except (OSError, ValueError, WhisperXAlignmentError) as exc:
        print_error(str(exc))
        return 1
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text(chunks_to_srt(result.chunks), encoding="utf-8")
    write_alignment_metadata(metadata_path, result.metadata)
    print("Done.")
    print(f"  WhisperX Japanese SRT: {output_srt}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Subtitles: {len(result.chunks)}")
    print(f"  Changed timings: {result.metadata.get('changed_count')}")
    return 0


def whisperx_language_code(language: str) -> str:
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


def iter_media_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and is_supported_media(path))


def resolve_translation_model(
    requested_model: str | None,
    choose: bool,
    ollama_host: str,
    ollama_port: int,
    timeout_seconds: int = 600,
) -> str:
    if requested_model:
        return requested_model
    if choose:
        return choose_ollama_model(ollama_host, ollama_port, timeout_seconds)
    return load_saved_translation_model() or DEFAULT_TRANSLATION_MODEL


def choose_ollama_model(ollama_host: str, ollama_port: int, timeout_seconds: int = 600) -> str:
    options = TranslationOptions(
        model=DEFAULT_TRANSLATION_MODEL,
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        timeout_seconds=timeout_seconds,
    )
    models = sort_ollama_models_for_translation(get_ollama_models(options))
    print("Available Ollama models:")
    for index, model in enumerate(models, 1):
        print(f"  [{index}] {format_ollama_model_choice(model)}")
    choice = input("Choose translation model number: ").strip()
    if not choice.isdigit():
        raise OllamaModelError(f"Invalid model choice: {choice}")
    selected_index = int(choice)
    if selected_index < 1 or selected_index > len(models):
        raise OllamaModelError(f"Invalid model choice: {choice}")
    return models[selected_index - 1]


def print_process_result(result: ProcessResult) -> None:
    if result.status == "success":
        print("Done.")
    else:
        print(f"Finished with status: {result.status}")
    if result.ja_srt_path is not None:
        print(f"  Japanese SRT: {result.ja_srt_path}")
    if result.ja_aligned_srt_path is not None:
        print(f"  WhisperX Japanese SRT: {result.ja_aligned_srt_path}")
    if result.ko_srt_path is not None:
        print(f"  Korean SRT: {result.ko_srt_path}")
    if result.copied_ko_srt_path is not None:
        print(f"  Copied Korean SRT: {result.copied_ko_srt_path}")
    if result.wav_path is not None:
        print(f"  WAV: {result.wav_path}")
    print(f"  Output dir: {result.output_dir}")
    if result.status != "success":
        print(f"  Message: {result.message}")


def print_directory_summary(file_count: int, failed_count: int) -> None:
    succeeded = file_count - failed_count
    print("\nSummary")
    print(f"  Succeeded: {succeeded}/{file_count}")
    print(f"  Failed: {failed_count}")


def print_error(message: str) -> None:
    print("Failed.")
    print(f"  {message}")


if __name__ == "__main__":
    raise SystemExit(main())
