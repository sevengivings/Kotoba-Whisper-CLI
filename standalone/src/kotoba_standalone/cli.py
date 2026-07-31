from __future__ import annotations

import argparse
import json
from pathlib import Path

from kotoba_standalone.pipeline import process_video, validate_silence_threshold
from kotoba_standalone.progress import tqdm_progress
from kotoba_standalone.translate.ollama import translate_srt
from kotoba_standalone.types import ProcessOptions, TranslationOptions


DEFAULT_TRANSLATION_MODEL = "hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M"


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

    process = subparsers.add_parser("process", help="Process one media file directly.")
    process.add_argument("input", type=Path)
    process.add_argument("--output-dir", type=Path)
    process.add_argument("--language", default="japanese")
    process.add_argument("--silence-threshold-db", default="-42dB")
    process.add_argument("--auto-silence-threshold", action="store_true")
    process.add_argument("--batch-size", type=int, default=8)
    process.add_argument("--chunk-length-s", type=int, default=15)
    process.add_argument("--model-name", default="kotoba-tech/kotoba-whisper-v2.2")
    process.add_argument("--model-device", default="cuda:0")
    process.add_argument("--model-dtype", default="float16")
    process.add_argument("--translate", action="store_true")
    process.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    process.add_argument("--ollama-host", default="localhost")
    process.add_argument("--ollama-port", type=int, default=11434)
    process.add_argument("--korean-style", choices=("polite", "banmal", "strict-banmal"), default="polite")

    translate = subparsers.add_parser("translate", help="Translate one SRT file with Ollama.")
    translate.add_argument("input_srt", type=Path)
    translate.add_argument("--output", type=Path)
    translate.add_argument("--source", default="japanese")
    translate.add_argument("--target", default="korean")
    translate.add_argument("--model", default=DEFAULT_TRANSLATION_MODEL)
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
    options = ProcessOptions(
        output_dir=args.output_dir,
        language=args.language,
        silence_threshold_db=validate_silence_threshold(args.silence_threshold_db),
        auto_silence_threshold=args.auto_silence_threshold,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        model_name=args.model_name,
        model_device=args.model_device,
        model_dtype=args.model_dtype,
        translate=args.translate,
        translation_model=args.translation_model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        korean_style=args.korean_style,
    )
    with tqdm_progress() as progress:
        result = process_video(args.input, options, progress=progress)
    print(json.dumps(result.__dict__ | {
        "input_path": str(result.input_path),
        "output_dir": str(result.output_dir),
        "wav_path": str(result.wav_path) if result.wav_path else None,
        "ja_srt_path": str(result.ja_srt_path) if result.ja_srt_path else None,
        "ko_srt_path": str(result.ko_srt_path) if result.ko_srt_path else None,
    }, ensure_ascii=False, indent=2))
    return 0 if result.status != "failed" else 1


def run_translate(args: argparse.Namespace) -> int:
    options = TranslationOptions(
        model=args.model,
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
    print(f"[Info] Translating {args.input_srt} with Ollama model: {args.model}")
    with tqdm_progress() as progress:
        result = translate_srt(args.input_srt, options, progress=progress)
    print(f"[Info] Translation saved: {result.output_srt}")
    print(f"[Info] Translation metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
