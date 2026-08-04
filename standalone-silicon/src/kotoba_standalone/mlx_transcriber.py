from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kotoba_standalone.transcriber import TranscriptionDependencyError, TranscriptionResult
from kotoba_standalone.types import ProcessOptions


class MlxKotobaTranscriber:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self._mlx_whisper: Any | None = None
        self.device_name = ""
        self.torch_version = ""
        self.torch_cuda_version: str | None = None
        self.model_path = ""

    def load(self) -> None:
        try:
            import mlx
            import mlx.core as mx
            import mlx_whisper
        except ImportError as exc:
            raise TranscriptionDependencyError(
                "MLX Whisper dependencies are not installed. "
                "Run './install-silicon.sh', then retry."
            ) from exc

        if self.options.mlx_device == "cpu":
            mx.set_default_device(mx.cpu)
        elif self.options.mlx_device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            raise TranscriptionDependencyError(f"Unsupported MLX device: {self.options.mlx_device}")

        model_path = resolve_mlx_model_path(self.options.mlx_model_path)
        ensure_mlx_weights_name(model_path)
        self.model_path = str(model_path)
        self.device_name = f"mlx-{self.options.mlx_device}"
        self.torch_version = f"mlx {getattr(mlx, '__version__', 'unknown')}; mlx-whisper {getattr(mlx_whisper, '__version__', 'unknown')}"
        self._mlx_whisper = mlx_whisper

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        if self._mlx_whisper is None:
            raise RuntimeError("Transcriber is not loaded")
        result = self._mlx_whisper.transcribe(
            wav_path,
            path_or_hf_repo=self.model_path,
            language=mlx_language_code(self.options.language),
            verbose=False,
            word_timestamps=self.options.mlx_word_timestamps,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.45,
        )
        chunks = [
            {
                "timestamp": [float(segment.get("start") or 0.0), float(segment.get("end") or 0.0)],
                "text": str(segment.get("text") or "").strip(),
            }
            for segment in result.get("segments", [])
            if str(segment.get("text") or "").strip()
        ]
        if not chunks and str(result.get("text") or "").strip():
            chunks = [{"timestamp": [0.0, 0.01], "text": str(result["text"]).strip()}]
        return TranscriptionResult(
            raw={"text": str(result.get("text") or ""), "chunks": chunks},
            batch_size_used=1,
            device_name=self.device_name,
            torch_version=self.torch_version,
            torch_cuda_version=self.torch_cuda_version,
            word_timestamps_used=self.options.mlx_word_timestamps,
        )


def resolve_mlx_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        raise TranscriptionDependencyError(
            f"MLX Kotoba model was not found: {path}. "
            "Run './install-silicon.sh' or './tools/convert-kotoba-v22-mlx.sh', then retry."
        )
    return path


def ensure_mlx_weights_name(model_dir: Path) -> None:
    if (model_dir / "weights.safetensors").exists() or (model_dir / "weights.npz").exists():
        return
    if (model_dir / "model.safetensors").exists():
        _link_or_copy(model_dir / "model.safetensors", model_dir / "weights.safetensors")
        return
    if (model_dir / "model.npz").exists():
        _link_or_copy(model_dir / "model.npz", model_dir / "weights.npz")
        return
    raise TranscriptionDependencyError(
        f"MLX model directory does not contain weights.safetensors, weights.npz, model.safetensors, or model.npz: {model_dir}"
    )


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source.name)
    except OSError:
        shutil.copy2(source, target)


def mlx_language_code(language: str) -> str:
    normalized = language.strip().lower()
    return {"japanese": "ja", "english": "en", "korean": "ko", "chinese": "zh"}.get(normalized, language or "ja")
