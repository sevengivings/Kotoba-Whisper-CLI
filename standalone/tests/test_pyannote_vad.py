from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from kotoba_standalone.pyannote_vad import (
    BUNDLED_PYANNOTE_MODEL_CHECKPOINT,
    DEFAULT_PYANNOTE_VAD_MODEL,
    PyannoteVadAccessError,
    _chunk_frame_ranges,
    _disable_matplotlib_imports,
    _merge_overlapping_spans,
    _model_load_error,
    _progress_hook,
    _resolve_model_checkpoint,
    _trusted_pyannote_checkpoint_load_compat,
)
from kotoba_standalone.media import SilenceSpan


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


def test_chunk_frame_ranges_cover_long_audio_with_overlap() -> None:
    ranges = _chunk_frame_ranges(
        total_frames=100,
        sample_rate=10,
        chunk_duration_s=4.0,
        overlap_s=1.0,
    )

    assert ranges == [(0, 40), (30, 40), (60, 40)]


def test_merge_overlapping_spans_combines_chunk_overlap() -> None:
    spans = _merge_overlapping_spans(
        [
            SilenceSpan(30.0, 35.0),
            SilenceSpan(34.98, 40.0),
            SilenceSpan(50.0, 51.0),
        ]
    )

    assert spans == [SilenceSpan(30.0, 40.0), SilenceSpan(50.0, 51.0)]


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


def test_trusted_checkpoint_load_compat_forces_full_checkpoint_load() -> None:
    calls: list[bool | None] = []

    class FakeTorch:
        @staticmethod
        def load(*args: object, **kwargs: object) -> object:
            del args
            calls.append(kwargs.get("weights_only"))
            return {}

    original_load = FakeTorch.load

    with _trusted_pyannote_checkpoint_load_compat(FakeTorch, "bundled"):
        FakeTorch.load("checkpoint.bin", weights_only=True)

    assert calls == [False]
    assert FakeTorch.load is original_load


def test_disable_matplotlib_imports_installs_plot_stubs() -> None:
    _disable_matplotlib_imports()

    import matplotlib.pyplot as plt
    from torchmetrics.utilities.plot import plot_curve

    assert importlib.util.find_spec("matplotlib") is not None
    assert importlib.util.find_spec("matplotlib.pyplot") is not None
    assert importlib.util.find_spec("torchmetrics.utilities.plot") is not None
    assert plt.Figure is object
    try:
        plot_curve()
    except ModuleNotFoundError as exc:
        assert "Plotting is disabled" in str(exc)
    else:
        raise AssertionError("plot_curve should be disabled")
