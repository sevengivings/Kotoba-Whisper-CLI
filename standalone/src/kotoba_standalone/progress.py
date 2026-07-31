from __future__ import annotations

import time
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from tqdm import tqdm

from kotoba_standalone.types import ProgressEvent


ProgressCallback = Callable[[ProgressEvent], None]


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@contextmanager
def tqdm_progress() -> Iterator[ProgressCallback]:
    bar: tqdm | None = None
    started = time.time()

    def callback(event: ProgressEvent) -> None:
        nonlocal bar
        total = event.total or 1
        current = event.current or 0
        if bar is None:
            bar = tqdm(total=total, desc=event.message, unit="step", leave=False, file=sys.stdout)
        if bar.total != total:
            bar.total = total
        bar.set_description(event.message)
        delta = max(0, current - bar.n)
        if delta:
            bar.update(delta)
        elapsed = event.elapsed_seconds
        if elapsed is None:
            elapsed = time.time() - started
        if event.percent is not None:
            bar.set_postfix_str(f"{event.percent:.1f}% elapsed {format_duration(elapsed)}")
        else:
            bar.set_postfix_str(f"elapsed {format_duration(elapsed)}")

    try:
        yield callback
    finally:
        if bar is not None:
            bar.close()
