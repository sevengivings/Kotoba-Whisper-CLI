from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import kotoba_standalone.pipeline as pipeline
from kotoba_standalone.alignment import WhisperXAlignmentResult
from kotoba_standalone.media import SilenceSpan
from kotoba_standalone.pipeline import process_video, tail_retranscribe_long_subtitles, validate_silence_threshold
from kotoba_standalone.pyannote_vad import PyannoteVadDependencyError, PyannoteVadResult
from kotoba_standalone.subtitle import SubtitleChunk, SubtitleQualityIssue
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
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", MissingTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg"),
        progress=events.append,
    )

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
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg"))

    assert result.status == "success"
    assert result.ja_srt_path == tmp_path / "out" / "ja_short_test.ja.srt"
    assert (tmp_path / "out" / "ja_short_test.ja.srt").read_text(encoding="utf-8").strip()


def test_process_video_uses_qwen_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQwenTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(text="こんにちは")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "Qwen3Transcriber", FakeQwenTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg", asr_backend="qwen3"),
    )

    assert result.status == "success"
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["asr_backend"] == "qwen3"
    assert process_meta["asr_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert process_meta["qwen_aligner_model"] == "Qwen/Qwen3-ForcedAligner-0.6B"


def test_process_video_uses_faster_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFasterTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(text="こんにちは")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeFasterTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg", asr_backend="faster-kotoba"),
    )

    assert result.status == "success"
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["asr_backend"] == "faster-kotoba"
    assert process_meta["asr_model"] == "RoachLin/kotoba-whisper-v2.2-faster"


def test_process_video_uses_mlx_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMlxTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(text="さあ")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "MlxKotobaTranscriber", FakeMlxTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(
            output_dir=tmp_path / "out",
            vad_engine="ffmpeg",
            asr_backend="kotoba-mlx",
            mlx_model_path="/tmp/kotoba-mlx",
        ),
    )

    assert result.status == "success"
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["asr_backend"] == "kotoba-mlx"
    assert process_meta["asr_model"] == "/tmp/kotoba-mlx"


def test_process_video_can_write_whisperx_aligned_subtitles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(0.0, 3.0, "こんにちは")

    def fake_align(*args: object, **kwargs: object) -> WhisperXAlignmentResult:
        return WhisperXAlignmentResult(
            [SubtitleChunk(1.0, 2.0, "こんにちは")],
            {"engine": "whisperx", "changed_count": 1},
        )

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])
    monkeypatch.setattr(pipeline, "align_chunks_with_whisperx", fake_align)

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg", alignment_engine="whisperx"),
    )

    assert result.status == "success"
    assert result.ja_aligned_srt_path == tmp_path / "out" / "ja_short_test.whisperx.ja.srt"
    assert "00:00:01,000 --> 00:00:02,000" in result.ja_aligned_srt_path.read_text(encoding="utf-8")
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["alignment_engine"] == "whisperx"
    assert process_meta["whisperx_changed_subtitle_count"] == 1


def test_process_video_can_drop_likely_hallucinations_and_write_quality_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(0.0, 1.0, "ありがとうございました")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(
            output_dir=tmp_path / "out",
            drop_likely_hallucinations=True,
            vad_engine="ffmpeg",
        ),
    )

    assert result.status == "success"
    assert (tmp_path / "out" / "ja_short_test.ja.srt").read_text(encoding="utf-8") == ""
    quality = json.loads((tmp_path / "out" / "ja_short_test.subtitle-quality.json").read_text(encoding="utf-8"))
    assert quality["dropped_count"] == 1
    assert quality["dropped"][0]["flags"] == ["blocked_phrase"]
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["dropped_likely_hallucination_count"] == 1


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
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", translate=True, vad_engine="ffmpeg"),
    )

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
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [])

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", translate=True, vad_engine="ffmpeg"),
    )

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
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [SilenceSpan(1.0, 2.0)])
    monkeypatch.setattr(
        pipeline,
        "speech_spans_from_silences",
        lambda *args: [SilenceSpan(0.0, 1.0), SilenceSpan(2.0, 3.0)],
    )

    result = process_video(media, ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg"))

    assert result.status == "success"
    raw = json.loads((tmp_path / "out" / "ja_short_test.raw.json").read_text(encoding="utf-8"))
    assert raw["chunks"][0]["timestamp"] == [0.2, 0.7]
    assert raw["chunks"][1]["timestamp"] == [2.2, 2.7]
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["transcription_segment_count"] == 2


def test_process_video_transcribes_faster_segments_without_temp_wavs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFasterTranscriber(pipeline.FasterKotobaTranscriber):
        calls: list[tuple[str, float, float]] = []

        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe_segment(self, wav_path: str, start_s: float, end_s: float) -> FakeResult:
            self.calls.append((wav_path, start_s, end_s))
            return FakeResult(0.2, 0.7, f"text {len(self.calls)}")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeFasterTranscriber)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: [SilenceSpan(1.0, 2.0)])
    monkeypatch.setattr(
        pipeline,
        "speech_spans_from_silences",
        lambda *args: [SilenceSpan(0.0, 1.0), SilenceSpan(2.0, 3.0)],
    )
    monkeypatch.setattr(pipeline, "extract_audio_segment", lambda *args: pytest.fail("temporary segment WAV should not be created"))

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="ffmpeg", asr_backend="faster-kotoba"),
    )

    assert result.status == "success"
    assert FakeFasterTranscriber.calls == [
        (str(tmp_path / "out" / "ja_short_test.standalone.wav"), 0.0, 1.0),
        (str(tmp_path / "out" / "ja_short_test.standalone.wav"), 2.0, 3.0),
    ]
    assert not list((tmp_path / "out").glob("*.segment.*.wav"))


def test_process_video_uses_pyannote_speech_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTranscriber:
        calls: list[str] = []

        def __init__(self, options: ProcessOptions) -> None:
            self.options = options

        def load(self) -> None:
            return None

        def transcribe(self, wav_path: str) -> FakeResult:
            self.calls.append(wav_path)
            return FakeResult(0.1, 0.6, f"text {len(self.calls)}")

    def fake_pyannote(*args: object, **kwargs: object) -> PyannoteVadResult:
        return PyannoteVadResult(
            speech_spans=[SilenceSpan(1.0, 2.0), SilenceSpan(4.0, 5.0)],
            model_name="pyannote/segmentation-3.0",
            device="cuda:0",
            pyannote_audio_version="3.4.0",
            model_source="bundled",
            model_revision="e66f3d3b9eb0873085418a7b813d3b369bf160bb",
        )

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "detect_speech_spans_pyannote", fake_pyannote)
    monkeypatch.setattr(pipeline, "detect_silences", lambda *args: pytest.fail("FFmpeg VAD should not run"))

    result = process_video(
        media,
        ProcessOptions(
            output_dir=tmp_path / "out",
            vad_engine="pyannote",
            vad_padding_s=0.0,
        ),
    )

    assert result.status == "success"
    raw = json.loads((tmp_path / "out" / "ja_short_test.raw.json").read_text(encoding="utf-8"))
    assert raw["chunks"][0]["timestamp"] == [1.1, 1.6]
    assert raw["chunks"][1]["timestamp"] == [4.1, 4.6]
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["vad_engine"] == "pyannote"
    assert process_meta["detected_speech_count"] == 2
    assert process_meta["pyannote_vad"]["pyannote_audio_version"] == "3.4.0"
    assert process_meta["pyannote_vad"]["model_source"] == "bundled"
    assert process_meta["pyannote_vad"]["model_revision"] == "e66f3d3b9eb0873085418a7b813d3b369bf160bb"
    assert process_meta["vad_report"].endswith("ja_short_test.vad.json")
    vad_report = json.loads((tmp_path / "out" / "ja_short_test.vad.json").read_text(encoding="utf-8"))
    assert vad_report["raw_speech_spans"][0] == {"start": 1.0, "end": 2.0, "duration": 1.0}
    assert len(vad_report["transcription_spans"]) == 2


def test_process_video_reports_pyannote_dependency_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_pyannote(*args: object, **kwargs: object) -> PyannoteVadResult:
        raise PyannoteVadDependencyError("install the pyannote group")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "detect_speech_spans_pyannote", missing_pyannote)

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="pyannote"),
    )

    assert result.status == "vad_error"
    assert result.ja_srt_path is None
    assert result.message == "install the pyannote group"


def test_process_video_skips_kotoba_when_pyannote_finds_no_speech(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedTranscriber:
        def __init__(self, options: ProcessOptions) -> None:
            pytest.fail("ASR should not load when pyannote found no speech")

    def no_speech(*args: object, **kwargs: object) -> PyannoteVadResult:
        return PyannoteVadResult([], "pyannote/segmentation-3.0", "cuda:0", "3.4.0")

    media = Path(__file__).parents[2] / "sample" / "ja_short_test.mp4"
    monkeypatch.setattr(pipeline, "FasterKotobaTranscriber", UnexpectedTranscriber)
    monkeypatch.setattr(pipeline, "detect_speech_spans_pyannote", no_speech)

    result = process_video(
        media,
        ProcessOptions(output_dir=tmp_path / "out", vad_engine="pyannote"),
    )

    assert result.status == "success"
    assert (tmp_path / "out" / "ja_short_test.ja.srt").read_text(encoding="utf-8") == ""
    process_meta = json.loads((tmp_path / "out" / "ja_short_test.process.json").read_text(encoding="utf-8"))
    assert process_meta["transcription_segment_count"] == 0


def test_tail_retranscribe_rejects_too_short_tail_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTailTranscriber:
        def transcribe(self, wav_path: str) -> FakeResult:
            return FakeResult(0.0, 1.0, "\u3042\u3042")

    chunk = SubtitleChunk(10.0, 40.0, "\u3042\u3042\u306f\u3044\u3042\u3042")
    issue = SubtitleQualityIssue(
        index=1,
        start=chunk.start,
        end=chunk.end,
        duration=chunk.end - chunk.start,
        text=chunk.text,
        flags=["long_duration_short_text"],
        recommended_action="drop_candidate",
    )
    monkeypatch.setattr(pipeline, "extract_audio_segment", lambda *args: None)

    refined, applied, attempts = tail_retranscribe_long_subtitles(
        [chunk],
        [issue],
        tmp_path / "source.wav",
        tmp_path,
        "sample",
        FakeTailTranscriber(),  # type: ignore[arg-type]
    )

    assert refined == [chunk]
    assert applied == []
    assert attempts[0]["accepted"] is False
