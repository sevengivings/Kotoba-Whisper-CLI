from __future__ import annotations

import hashlib
from pathlib import Path

from kotoba_standalone.pyannote_vad import (
    BUNDLED_PYANNOTE_MODEL_CHECKPOINT,
    DEFAULT_PYANNOTE_VAD_MODEL,
    PyannoteVadAccessError,
    _model_load_error,
    _progress_hook,
    _resolve_model_checkpoint,
)


def test_model_load_error_explains_gated_model_access() -> None:
    error = _model_load_error("pyannote/segmentation-3.0", RuntimeError("403 gated repo"))

    assert isinstance(error, PyannoteVadAccessError)
    assert "accept the model conditions" in str(error)
    assert "huggingface-cli login" in str(error)


def test_progress_hook_forwards_completed_work() -> None:
    updates: list[tuple[int, int]] = []
    hook = _progress_hook(lambda current, total: updates.append((current, total)))

    assert hook is not None
    hook("segmentation", None, completed=3, total=10)

    assert updates == [(3, 10)]


def test_progress_hook_clamps_batch_overshoot() -> None:
    updates: list[tuple[int, int]] = []
    hook = _progress_hook(lambda current, total: updates.append((current, total)))

    assert hook is not None
    hook("segmentation", None, completed=32, total=4)

    assert updates == [(4, 4)]


def test_default_pyannote_model_uses_bundled_checkpoint() -> None:
    checkpoint, source = _resolve_model_checkpoint(DEFAULT_PYANNOTE_VAD_MODEL)

    assert source == "bundled"
    assert checkpoint == BUNDLED_PYANNOTE_MODEL_CHECKPOINT
    assert BUNDLED_PYANNOTE_MODEL_CHECKPOINT.is_file()
    assert hashlib.sha256(BUNDLED_PYANNOTE_MODEL_CHECKPOINT.read_bytes()).hexdigest() == (
        "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
    )


def test_local_pyannote_directory_resolves_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"test")

    resolved, source = _resolve_model_checkpoint(str(tmp_path))

    assert resolved == checkpoint
    assert source == "local"
