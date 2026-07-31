from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_estimate_module():
    module_path = Path(__file__).parents[1] / "tools" / "estimate-time.py"
    spec = importlib.util.spec_from_file_location("estimate_time", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load estimate-time.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_transcription_samples(tmp_path: Path) -> None:
    module = load_estimate_module()
    (tmp_path / "sample.process.json").write_text(
        json.dumps(
            {
                "status": "success",
                "media_duration_seconds": 1000,
                "processing_seconds": 125,
                "realtime_factor": 0.125,
            }
        ),
        encoding="utf-8",
    )

    samples = module.load_transcription_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0].media_duration_seconds == 1000
    assert samples[0].processing_seconds == 125
    assert samples[0].realtime_factor == 0.125


def test_load_translation_samples(tmp_path: Path) -> None:
    module = load_estimate_module()
    (tmp_path / "sample.ko.translation.json").write_text(
        json.dumps(
            {
                "status": "success",
                "subtitle_count": 200,
                "processing_seconds": 50,
                "model": "gemma",
                "batch_translate": True,
                "batch_size": 50,
                "korean_style": "polite",
            }
        ),
        encoding="utf-8",
    )

    samples = module.load_translation_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0].seconds_per_subtitle == 0.25
    assert samples[0].model == "gemma"
    assert samples[0].batch_translate is True
    assert samples[0].batch_size == 50


def test_format_seconds() -> None:
    module = load_estimate_module()

    assert module.format_seconds(47.5) == "48s"
    assert module.format_seconds(127.4) == "2m 7s"
    assert module.format_seconds(3671) == "1h 1m 11s"


def test_percentile_interpolates() -> None:
    module = load_estimate_module()

    assert module.percentile([1.0, 2.0, 3.0, 4.0], 75) == 3.25
