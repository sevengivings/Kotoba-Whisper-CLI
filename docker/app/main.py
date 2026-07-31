from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import ensure_directories, load_config
from app.logging_config import setup_logging
from app.processor import MediaProcessor
from app.transcriber import KotobaTranscriber
from app.watcher import FolderWatcher, install_signal_handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kotoba-Whisper V2.2 folder watcher")
    parser.add_argument("--config", default="/workspace/config/config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("watch", help="Watch input folder")
    transcribe = subparsers.add_parser("transcribe", help="Transcribe one media file")
    transcribe.add_argument("file", help="Media file to transcribe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    ensure_directories(config)
    setup_logging(config.paths.logs, config.logging.level, config.logging.keep_days)

    transcriber = KotobaTranscriber(config)
    try:
        transcriber.load()
    except Exception:
        logging.exception("Global startup failure")
        return 2

    processor = MediaProcessor(config, transcriber)
    command = args.command or "watch"
    if command == "transcribe":
        outcome = processor.process_input_file(Path(args.file))
        logging.info("Transcribe finished: %s", outcome.status)
        return 0 if outcome.status in {"success", "suspicious_incomplete"} else 1

    watcher = FolderWatcher(config, processor)
    install_signal_handlers(watcher)
    watcher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
