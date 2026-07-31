from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import kotoba_standalone.cli as cli
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
    assert args.auto_silence_threshold is False


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
        )

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(cli, "tqdm_progress", null_progress)

    exit_code = cli.main(["process", str(media_path), "--translate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Done." in captured.out
    assert "Japanese SRT:" in captured.out
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


def test_resolve_translation_model_uses_saved_model(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_saved_translation_model", lambda: "saved:model")

    assert cli.resolve_translation_model(None, False, "localhost", 11434) == "saved:model"


def test_choose_ollama_model_selects_by_number(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_ollama_models", lambda options: ["first:model", "second:model"])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    selected = cli.choose_ollama_model("localhost", 11434)

    captured = capsys.readouterr()
    assert selected == "second:model"
    assert "[1] first:model" in captured.out
    assert "[2] second:model" in captured.out


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
