from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, level: str, keep_days: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s")

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        log_dir / "kotoba-folder-watcher.log",
        when="midnight",
        backupCount=keep_days,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

