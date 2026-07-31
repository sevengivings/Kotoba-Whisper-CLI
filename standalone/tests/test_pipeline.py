from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import kotoba_standalone.pipeline as pipeline
from kotoba_standalone.media import SilenceSpan
from kotoba_standalone.pipeline import process_video, validate_silence_threshold
from kotoba_standalone.transcriber import TranscriptionDependencyError
from kotoba_standalone.types import ProcessOptions, ProgressEvent


class FakeResult:
    def __init__(self, start: float = 0.0, end: float = 1.0, text: str = "konnichiwa") -> None:
        self.raw = {"chunks": [{"timestamp": [start, end], "text": text}]}
        self.batch_size_used = 8
        self.device_name = "fake gpu"
        self.torch_version = "test"
        self.torch_cuda_version = "test"
        self.word_timestamps_used = False


def test_validate_silence_threshold_accepts_db_values() -> None:
    assert validate_silence_threshold("-42dB") == "-42dB"


def test_validate_silence_threshold_rejects_missing_db_suffix() -> None:
    with pytest.raises(ValueError):
        validate_silence_threshold("-42")


def test_process_video_handles_missing_transcription_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            raise TranscriptionDependencyError("missing test dependency")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    events: list[ProgressEvent] = []
    monkeypatch.setattr(pipeline, "KotobaTranscriber", MissingTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out"), progress=events.append)

    assert result.status == "missing_transcription_dependency"
    assert result.output_dir == tmp_path / "out"
    assert result.wav_path == tmp_path / "out" / "ja_short_test.standalone.wav"
    assert (tmp_path / "out" / "ja_short_test.standalone.wav").exists()
    assert [event.stage for event in events] == [
        "prepare",
        "extract_audio",
        "probe_audio",
        "detect_silence",
        "load_model",
        "dependency",
    ]


def test_process_video_writes_subtitles_with_fake_transcriber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult()

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "KotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out"))

    assert result.status == "success"
    assert result.ja_srt_path == tmp_path / "out" / "ja_short_test.ja.srt"
    assert (tmp_path / "out" / "ja_short_test.ja.srt").read_text(encoding="utf-8").strip()


def test_process_video_can_translate_after_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult()

    class FakeTranslation:
        output_srt = tmp_path / "out" / "ja_short_test.ko.srt"

    def fake_translate_srt(*args: object, **kwargs: object) -> FakeTranslation:
        FakeTranslation.output_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nannyeonghaseyo\n", encoding="utf-8")
        return FakeTranslation()

    source_media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    media = tmp_path / "media" / "ja_short_test.mp4"
    media.parent.mkdir()
    shutil.copy2(source_media, media)
    monkeypatch.setattr(pipeline, "KotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out", translate=True))

    assert result.status == "success"
    assert result.ko_srt_path == tmp_path / "out" / "ja_short_test.ko.srt"
    assert result.copied_ko_srt_path == tmp_path / "media" / "ja_short_test.srt"
    assert (tmp_path / "media" / "ja_short_test.srt").read_text(encoding="utf-8").strip()


def test_process_video_copies_translated_subtitle_with_ko_suffix_when_srt_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult()

    class FakeTranslation:
        output_srt = tmp_path / "out" / "video.ko.srt"

    def fake_translate_srt(*args: object, **kwargs: object) -> FakeTranslation:
        FakeTranslation.output_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nannyeonghaseyo\n", encoding="utf-8")
        return FakeTranslation()

    source_media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    media = tmp_path / "media" / "video.mp4"
    media.parent.mkdir()
    shutil.copy2(source_media, media)
    (tmp_path / "media" / "video.srt").write_text("existing subtitle\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "KotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out", translate=True))

    assert result.status == "success"
    assert result.copied_ko_srt_path == tmp_path / "media" / "video.ko.srt"
    assert (tmp_path / "media" / "video.srt").read_text(encoding="utf-8") == "existing subtitle\n"
    assert (tmp_path / "media" / "video.ko.srt").read_text(encoding="utf-8").strip()


def test_process_video_offsets_vad_segment_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranscriber:
        calls: list[str] = []

        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            self.calls.append(wav_path)
            return FakeResult(0.2, 0.7, f"text {len(self.calls)}")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "KotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [SilenceSpan(1.0, 2.0)])
    monkeypatch.setattr(
        pipeline,
        "speech_spans_from_silences",
        lambda *args: [SilenceSpan(0.0, 1.0), SilenceSpan(2.0, 3.0)],
    )

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out"))

    assert result.status == "success"
    raw = json.loads((tmp_path / "out" / "ja_short_test.raw.json").read_text(encoding="utf-8"))
    assert raw["chunks"][0]["timestamp"] == [0.2, 0.7]
    assert raw["chunks"][1]["timestamp"] == [2.2, 2.7]
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["transcription_segment_count"] == 2
