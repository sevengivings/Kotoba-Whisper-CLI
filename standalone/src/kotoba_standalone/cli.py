from __future__ import annotations

import argparse
from pathlib import Path

from kotoba_standalone.media import is_supported_media
from kotoba_standalone.pipeline import process_video, validate_silence_threshold
from kotoba_standalone.progress import tqdm_progress
from kotoba_standalone.settings import DEFAULT_TRANSLATION_MODEL, load_saved_translation_model, save_translation_model
from kotoba_standalone.translate.ollama import OllamaModelError, OllamaUnavailableError, get_ollama_models, translate_srt
from kotoba_standalone.types import ProcessOptions, ProcessResult, TranslationOptions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return run_process(args)
    if args.command == "translate":
        return run_translate(args)
    parser.print_help()
    return 1


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
    process.add_argument("--vad-max-segment-duration-s", type=float, default=30.0)
    process.add_argument("--vad-min-speech-duration-s", type=float, default=0.25)
    process.add_argument("--vad-padding-s", type=float, default=0.4)
    process.add_argument("--vad-merge-gap-s", type=float, default=0.0)
    process.add_argument("--batch-size", type=int, default=8)
    process.add_argument("--chunk-length-s", type=int, default=15)
    process.add_argument("--model-name", default="kotoba-tech/kotoba-whisper-v2.2")
    process.add_argument("--model-device", default="cuda:0")
    process.add_argument("--model-dtype", default="float16")
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
    return parser


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
        vad_max_segment_duration_s=args.vad_max_segment_duration_s,
        vad_min_speech_duration_s=args.vad_min_speech_duration_s,
        vad_padding_s=args.vad_padding_s,
        vad_merge_gap_s=args.vad_merge_gap_s,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        model_name=args.model_name,
        model_device=args.model_device,
        model_dtype=args.model_dtype,
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
                if result.status == "failed":
                    failed += 1
                print_process_result(result)
            except (OllamaModelError, OllamaUnavailableError, Exception) as exc:
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
        except (OllamaModelError, OllamaUnavailableError) as exc:
            print_error(str(exc))
            return 1
    print_process_result(result)
    if options.translate and result.ko_srt_path is not None:
        save_translation_model(options.translation_model)
    return 0 if result.status != "failed" else 1


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
    models = get_ollama_models(options)
    print("Available Ollama models:")
    for index, model in enumerate(models, 1):
        print(f"  [{index}] {model}")
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
