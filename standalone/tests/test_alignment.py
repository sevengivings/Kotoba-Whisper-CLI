from __future__ import annotations

import sys
import builtins
from pathlib import Path

import pytest

import kotoba_standalone.alignment as alignment
from kotoba_standalone.alignment import WhisperXDependencyError, align_chunks_with_whisperx
from kotoba_standalone.subtitle import SubtitleChunk, parse_srt_chunks


class FakeWhisperX:
    @staticmethod
    def load_audio(path: str) -> str:
        return path

    @staticmethod
    def load_align_model(**kwargs: object) -> tuple[str, dict[str, str]]:
        return "model", {"model": "fake-align-model"}

    @staticmethod
    def align(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "segments": [
                {
                    "start": 1.2,
                    "end": 2.4,
                    "text": "こんにちは",
                }
            ]
        }


def test_align_chunks_with_whisperx_uses_aligned_segment_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "whisperx", FakeWhisperX)
    monkeypatch.setattr(alignment, "_load_audio", lambda *args, **kwargs: "audio")

    result = align_chunks_with_whisperx(
        [SubtitleChunk(1.0, 3.0, "こんにちは")],
        tmp_path / "sample.wav",
        language_code="ja",
        device="cuda:0",
    )

    assert result.chunks == [SubtitleChunk(1.2, 3.0, "こんにちは")]
    assert result.metadata["model"] == "fake-align-model"
    assert result.metadata["changed_count"] == 1


def test_align_chunks_with_whisperx_reports_missing_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisperx":
            raise ImportError("missing whisperx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(WhisperXDependencyError):
        align_chunks_with_whisperx([SubtitleChunk(0.0, 1.0, "text")], tmp_path / "sample.wav")


def test_parse_srt_chunks_reads_utf8_sig_srt() -> None:
    chunks = parse_srt_chunks("\ufeff1\n00:00:01,000 --> 00:00:02,500\nこんにちは\n\n")

    assert chunks == [SubtitleChunk(1.0, 2.5, "こんにちは")]
