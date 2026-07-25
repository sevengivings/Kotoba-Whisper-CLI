from pathlib import Path

from app.config import load_config


def test_config_loading() -> None:
    config = load_config(Path("config/config.yaml"))
    assert config.model.name == "kotoba-tech/kotoba-whisper-v2.2"
    assert config.inference.batch_size == 8
    assert config.paths.output.as_posix() == "/workspace/output"

