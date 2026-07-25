from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    input: Path
    processing: Path
    output: Path
    archive: Path
    failed: Path
    logs: Path
    temp: Path


@dataclass(frozen=True)
class WatcherConfig:
    scan_interval_seconds: int
    stable_check_interval_seconds: int
    stable_required_checks: int
    minimum_file_age_seconds: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str | None
    device: str
    dtype: str
    attention_implementation: str


@dataclass(frozen=True)
class InferenceConfig:
    language: str
    chunk_length_s: int
    batch_size: int
    return_timestamps: bool
    word_timestamps: bool
    punctuation: bool
    fallback_batch_sizes: list[int]


@dataclass(frozen=True)
class OutputConfig:
    write_srt: bool
    write_txt: bool
    write_raw_json: bool
    write_process_json: bool
    utf8_bom: bool
    overwrite_existing: bool


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool
    minimum_coverage_ratio: float
    maximum_uncovered_tail_seconds: float
    suspicious_result_destination: str


@dataclass(frozen=True)
class RecoveryConfig:
    retry_processing_files_on_start: bool
    maximum_attempts: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    keep_days: int


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    watcher: WatcherConfig
    model: ModelConfig
    inference: InferenceConfig
    output: OutputConfig
    validation: ValidationConfig
    recovery: RecoveryConfig
    logging: LoggingConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config key: {key}")
    return mapping[key]


def _path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid path value: {value!r}")
    return Path(value)


def load_config(config_path: Path) -> AppConfig:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read config file {config_path}: {exc}") from exc

    try:
        paths = _require(raw, "paths")
        watcher = _require(raw, "watcher")
        model = _require(raw, "model")
        inference = _require(raw, "inference")
        output = _require(raw, "output")
        validation = _require(raw, "validation")
        recovery = _require(raw, "recovery")
        logging_cfg = _require(raw, "logging")

        app_config = AppConfig(
            paths=PathsConfig(
                input=_path(_require(paths, "input")),
                processing=_path(_require(paths, "processing")),
                output=_path(_require(paths, "output")),
                archive=_path(_require(paths, "archive")),
                failed=_path(_require(paths, "failed")),
                logs=_path(_require(paths, "logs")),
                temp=_path(_require(paths, "temp")),
            ),
            watcher=WatcherConfig(
                scan_interval_seconds=int(_require(watcher, "scan_interval_seconds")),
                stable_check_interval_seconds=int(_require(watcher, "stable_check_interval_seconds")),
                stable_required_checks=int(_require(watcher, "stable_required_checks")),
                minimum_file_age_seconds=int(_require(watcher, "minimum_file_age_seconds")),
            ),
            model=ModelConfig(
                name=str(_require(model, "name")),
                revision=model.get("revision"),
                device=str(_require(model, "device")),
                dtype=str(_require(model, "dtype")),
                attention_implementation=str(_require(model, "attention_implementation")),
            ),
            inference=InferenceConfig(
                language=str(_require(inference, "language")),
                chunk_length_s=int(_require(inference, "chunk_length_s")),
                batch_size=int(_require(inference, "batch_size")),
                return_timestamps=bool(_require(inference, "return_timestamps")),
                word_timestamps=bool(_require(inference, "word_timestamps")),
                punctuation=bool(inference.get("punctuation", True)),
                fallback_batch_sizes=[int(v) for v in _require(inference, "fallback_batch_sizes")],
            ),
            output=OutputConfig(
                write_srt=bool(_require(output, "write_srt")),
                write_txt=bool(_require(output, "write_txt")),
                write_raw_json=bool(_require(output, "write_raw_json")),
                write_process_json=bool(_require(output, "write_process_json")),
                utf8_bom=bool(_require(output, "utf8_bom")),
                overwrite_existing=bool(_require(output, "overwrite_existing")),
            ),
            validation=ValidationConfig(
                enabled=bool(_require(validation, "enabled")),
                minimum_coverage_ratio=float(_require(validation, "minimum_coverage_ratio")),
                maximum_uncovered_tail_seconds=float(_require(validation, "maximum_uncovered_tail_seconds")),
                suspicious_result_destination=str(_require(validation, "suspicious_result_destination")),
            ),
            recovery=RecoveryConfig(
                retry_processing_files_on_start=bool(_require(recovery, "retry_processing_files_on_start")),
                maximum_attempts=int(_require(recovery, "maximum_attempts")),
            ),
            logging=LoggingConfig(
                level=str(_require(logging_cfg, "level")),
                keep_days=int(_require(logging_cfg, "keep_days")),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid config in {config_path}: {exc}") from exc

    validate_config(app_config)
    return app_config


def validate_config(config: AppConfig) -> None:
    if config.inference.batch_size < 1:
        raise ValueError("inference.batch_size must be >= 1")
    if not 0 < config.validation.minimum_coverage_ratio <= 1:
        raise ValueError("validation.minimum_coverage_ratio must be between 0 and 1")
    if config.validation.suspicious_result_destination not in {"failed", "archive"}:
        raise ValueError("validation.suspicious_result_destination must be 'failed' or 'archive'")
    if config.model.device != "cuda:0":
        raise ValueError("model.device must be cuda:0; CPU fallback is intentionally disabled")


def ensure_directories(config: AppConfig) -> None:
    for path in (
        config.paths.input,
        config.paths.processing,
        config.paths.output,
        config.paths.archive,
        config.paths.failed,
        config.paths.logs,
        config.paths.temp,
    ):
        path.mkdir(parents=True, exist_ok=True)
