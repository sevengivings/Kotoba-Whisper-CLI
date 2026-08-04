from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import kotoba_standalone.cli as cli
from kotoba_standalone.media import FFMPEG_PATH_ENV, FFmpegAudioExtractionError
from kotoba_standalone.translate.ollama import OllamaUnavailableError
from kotoba_standalone.types import ProcessResult, TranslationResult


@contextmanager
def null_progress() -> Iterator[None]:
    yield None


def test_iter_media_files_defaults_to_top_level_only(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.mkv").write_text("", encoding="utf-8")

    assert cli.iter_media_files(tmp_path) == [tmp_path / "a.mp4"]


def test_process_parser_defaults_to_pyannote(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["process", str(tmp_path / "a.mp4")])

    assert args.vad_engine == "pyannote"
    assert args.asr_backend == "kotoba"
    assert args.auto_silence_threshold is False
    assert args.alignment_engine == "none"


def test_process_parser_accepts_hidden_qwen_backend(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["process", str(tmp_path / "a.mp4"), "--asr-backend", "qwen3"])

    assert args.asr_backend == "qwen3"
    assert args.qwen_model_name == "Qwen/Qwen3-ASR-1.7B"
    assert args.qwen_aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert args.qwen_return_timestamps is True


def test_process_parser_accepts_hidden_faster_backend(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["process", str(tmp_path / "a.mp4"), "--asr-backend", "faster-kotoba"])

    assert args.asr_backend == "faster-kotoba"


def test_qwen_backend_reports_missing_experiment_environment(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "qwen_dependencies_available", lambda: False)
    monkeypatch.setattr(cli, "qwen_environment_python", lambda: None)

    exit_code = cli.main(["process", str(tmp_path / "a.mp4"), "--asr-backend", "qwen3"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Qwen3-ASR experimental environment was not found" in captured.out
    assert "uv sync --group cuda --group pyannote --group qwen" in captured.out


def test_qwen_backend_reexecs_with_experiment_environment(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    qwen_python = tmp_path / ".venv-qwen" / "Scripts" / "python.exe"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.write_text("", encoding="utf-8")
    monkeypatch.delenv(FFMPEG_PATH_ENV, raising=False)

    def fake_run(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
        calls.append((command, cwd, env))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli, "qwen_dependencies_available", lambda: False)
    monkeypatch.setattr(cli, "qwen_environment_python", lambda: qwen_python)
    monkeypatch.setattr(cli, "standalone_root", lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    ffmpeg_path = r"C:\Tools\ffmpeg.exe"
    exit_code = cli.main([
        "process",
        str(tmp_path / "a.mp4"),
        "--asr-backend",
        "qwen3",
        "--ffmpeg-path",
        ffmpeg_path,
    ])

    assert exit_code == 7
    command, cwd, env = calls[0]
    assert command[:3] == [str(qwen_python), "-m", "kotoba_standalone.cli"]
    assert "--asr-backend" in command
    assert "--ffmpeg-path" in command
    assert cwd == tmp_path
    assert env["KOTOBA_QWEN_REEXEC"] == "1"
    assert env[FFMPEG_PATH_ENV] == ffmpeg_path


def test_process_parser_applies_ffmpeg_path_environment(tmp_path: Path, monkeypatch) -> None:
    ffmpeg_path = r"C:\Tools\ffmpeg.exe"
    seen_env: list[str | None] = []
    monkeypatch.delenv(FFMPEG_PATH_ENV, raising=False)
    monkeypatch.setenv("KOTOBA_QWEN_REEXEC", "1")

    def fake_process_video(*args: object, **kwargs: object) -> ProcessResult:
        seen_env.append(cli.os.environ.get(FFMPEG_PATH_ENV))
        return ProcessResult(
            input_path=tmp_path / "a.mp4",
            output_dir=tmp_path,
            wav_path=None,
            ja_srt_path=None,
            ko_srt_path=None,
            copied_ko_srt_path=None,
            status="success",
            message="ok",
        )

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(tmp_path / "a.mp4"), "--ffmpeg-path", ffmpeg_path])

    assert exit_code == 0
    assert seen_env == [ffmpeg_path]
    assert cli.os.environ.get(FFMPEG_PATH_ENV) is None


def test_process_parser_accepts_whisperx_alignment(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["process", str(tmp_path / "a.mp4"), "--whisperx-align"])

    assert args.alignment_engine == "whisperx"


def test_process_directory_processes_each_media_file(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []
    (tmp_path / "a.mp4").write_text("", encoding="utf-8")
    (tmp_path / "b.mkv").write_text("", encoding="utf-8")

    def fake_process_video(input_path: Path, options, progress=None) -> ProcessResult:
        calls.append(input_path)
        return ProcessResult(
            input_path=input_path,
            output_dir=tmp_path,
            wav_path=None,
            ja_srt_path=tmp_path / f"{input_path.stem}.ja.srt",
            ko_srt_path=None,
            copied_ko_srt_path=None,
            status="success",
            message="ok",
        )

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(tmp_path)])

    assert exit_code == 0
    assert calls == [tmp_path / "a.mp4", tmp_path / "b.mkv"]


def test_translate_reports_ollama_unavailable_without_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    srt_path = tmp_path / "sample.ja.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nkonnichiwa\n", encoding="utf-8")

    def fake_translate_srt(*args: object, **kwargs: object) -> object:
        raise OllamaUnavailableError("Ollama is not reachable at http://localhost:11434/api/tags.")

    monkeypatch.setattr(cli, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["translate", str(srt_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed." in captured.out
    assert "Ollama is not reachable" in captured.out


def test_process_file_prints_human_readable_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    media_path = tmp_path / "a.mp4"
    media_path.write_text("", encoding="utf-8")

    def fake_process_video(input_path: Path, options, progress=None) -> ProcessResult:
        return ProcessResult(
            input_path=input_path,
            output_dir=tmp_path,
            wav_path=tmp_path / "a.wav",
            ja_srt_path=tmp_path / "a.ja.srt",
            ko_srt_path=tmp_path / "a.ko.srt",
            copied_ko_srt_path=tmp_path / "a.srt",
            status="success",
            message="ok",
            ja_aligned_srt_path=tmp_path / "a.whisperx.ja.srt",
        )

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(media_path), "--translate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Done." in captured.out
    assert "Japanese SRT:" in captured.out
    assert "WhisperX Japanese SRT:" in captured.out
    assert "Korean SRT:" in captured.out
    assert "Copied Korean SRT:" in captured.out
    assert "{" not in captured.out


def test_process_file_returns_failure_for_vad_error(tmp_path: Path, monkeypatch, capsys) -> None:
    media_path = tmp_path / "a.mp4"
    media_path.write_text("", encoding="utf-8")

    def fake_process_video(input_path: Path, options, progress=None) -> ProcessResult:
        return ProcessResult(
            input_path=input_path,
            output_dir=tmp_path,
            wav_path=tmp_path / "a.wav",
            ja_srt_path=None,
            ko_srt_path=None,
            copied_ko_srt_path=None,
            status="vad_error",
            message="Hugging Face access is required",
        )

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(media_path), "--vad-engine", "pyannote"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Finished with status: vad_error" in captured.out
    assert "Hugging Face access is required" in captured.out


def test_process_file_reports_ffmpeg_audio_error_without_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    media_path = tmp_path / "broken.mp4"
    media_path.write_text("", encoding="utf-8")

    def fake_process_video(*args: object, **kwargs: object) -> ProcessResult:
        raise FFmpegAudioExtractionError("FFmpeg could not extract audio\nSet KOTOBA_FFMPEG_PATH")

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(media_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed." in captured.out
    assert "KOTOBA_FFMPEG_PATH" in captured.out
    assert "Traceback" not in captured.out


def test_resolve_translation_model_uses_saved_model(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_saved_translation_model", lambda: "saved:model")

    assert cli.resolve_translation_model(None, False, "localhost", 11434) == "saved:model"


def test_choose_ollama_model_selects_by_number(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_ollama_models", lambda options: ["first:model", "second:model"])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    selected = cli.choose_ollama_model("localhost", 11434)

    captured = capsys.readouterr()
    assert selected == "second:model"
    assert "[번역 미확인] first:model" in captured.out
    assert "[번역 미확인] second:model" in captured.out


def test_translate_saves_successful_model(tmp_path: Path, monkeypatch) -> None:
    srt_path = tmp_path / "sample.ja.srt"
    output_path = tmp_path / "sample.ko.srt"
    metadata_path = tmp_path / "sample.translation.json"
    saved: list[str] = []
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nkonnichiwa\n", encoding="utf-8")

    def fake_translate_srt(*args: object, **kwargs: object) -> TranslationResult:
        return TranslationResult(srt_path, output_path, metadata_path, 1, 0.1)

    monkeypatch.setattr(cli, "translate_srt", fake_translate_srt)
    monkeypatch.setattr(cli, "save_translation_model", saved.append)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["translate", str(srt_path), "--model", "chosen:model"])

    assert exit_code == 0
    assert saved == ["chosen:model"]
