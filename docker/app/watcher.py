from __future__ import annotations

import json
import logging
import queue
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.config import AppConfig
from app.media import is_supported_media
from app.processor import MediaProcessor

LOGGER = logging.getLogger(__name__)


class MediaEventHandler(FileSystemEventHandler):
    def __init__(self, work_queue: "queue.Queue[Path]") -> None:
        self.work_queue = work_queue

    def on_created(self, event) -> None:  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.work_queue.put(Path(event.src_path))

    def on_moved(self, event) -> None:  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.work_queue.put(Path(event.dest_path))


class FolderWatcher:
    def __init__(self, config: AppConfig, processor: MediaProcessor) -> None:
        self.config = config
        self.processor = processor
        self.queue: "queue.Queue[Path]" = queue.Queue()
        self.in_progress: set[Path] = set()
        self.stop_event = threading.Event()
        self.health_lock = threading.Lock()
        self.health_file = config.paths.processing / ".health.json"

    def run(self) -> None:
        self._recover_processing_files()
        self._scan_input()

        observer = Observer()
        observer.schedule(MediaEventHandler(self.queue), str(self.config.paths.input), recursive=False)
        observer.start()
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()
        LOGGER.info("Watcher started: %s", self.config.paths.input)

        try:
            while not self.stop_event.is_set():
                self.write_health("ready")
                self._scan_input()
                self._drain_queue_once()
                time.sleep(self.config.watcher.scan_interval_seconds)
        finally:
            observer.stop()
            observer.join(timeout=10)
            self.write_health("stopped", watcher_running=False)

    def stop(self) -> None:
        self.stop_event.set()

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            self.write_health("ready")
            self.stop_event.wait(20)

    def _recover_processing_files(self) -> None:
        if not self.config.recovery.retry_processing_files_on_start:
            return
        for path in sorted(self.config.paths.processing.iterdir(), key=lambda p: p.stat().st_mtime):
            if path.is_file() and not should_ignore_path(path) and is_supported_media(path):
                LOGGER.info("Recovered processing file: %s", path.name)
                self.queue.put(path)

    def _scan_input(self) -> None:
        for path in sorted(self.config.paths.input.iterdir(), key=lambda p: p.stat().st_mtime):
            if path.is_file() and not should_ignore_path(path):
                self.queue.put(path)

    def _drain_queue_once(self) -> None:
        candidates: list[Path] = []
        while True:
            try:
                candidates.append(self.queue.get_nowait())
            except queue.Empty:
                break

        for path in unique_paths(candidates):
            if self.stop_event.is_set():
                break
            if path in self.in_progress or not path.exists():
                continue
            if path.parent == self.config.paths.input and not is_file_stable(path, self.config):
                continue
            self.in_progress.add(path)
            try:
                if path.parent == self.config.paths.processing:
                    self.processor.process_processing_file(path)
                else:
                    self.processor.process_input_file(path)
            finally:
                self.in_progress.discard(path)

    def write_health(
        self,
        status: str,
        *,
        model_loaded: bool = True,
        gpu_available: bool = True,
        watcher_running: bool = True,
    ) -> None:
        payload = {
            "status": status,
            "model_loaded": model_loaded,
            "gpu_available": gpu_available,
            "watcher_running": watcher_running,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        with self.health_lock:
            temp = self.health_file.with_suffix(".json.part")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.health_file)


def is_file_stable(path: Path, config: AppConfig) -> bool:
    if should_ignore_path(path):
        return False
    if not is_supported_media(path):
        LOGGER.info("Unsupported file ignored: %s", path)
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        age = time.time() - path.stat().st_mtime
        if age < config.watcher.minimum_file_age_seconds:
            return False
        previous_size = path.stat().st_size
        for _ in range(config.watcher.stable_required_checks):
            time.sleep(config.watcher.stable_check_interval_seconds)
            current_size = path.stat().st_size
            if current_size != previous_size or current_size <= 0:
                return False
            previous_size = current_size
        with path.open("rb"):
            pass
        LOGGER.info("Detected stable file: %s", path.name)
        return True
    except OSError as exc:
        LOGGER.info("File is not ready yet: %s (%s)", path, exc)
        return False


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def should_ignore_path(path: Path) -> bool:
    return path.name.startswith(".") or path.suffix.lower() == ".part"


def install_signal_handlers(watcher: FolderWatcher) -> None:
    def _stop(_signum, _frame) -> None:  # type: ignore[no-untyped-def]
        watcher.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
