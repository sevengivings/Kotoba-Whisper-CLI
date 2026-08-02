from __future__ import annotations

import json
import sys
from pathlib import Path

import kotoba_standalone.launcher as launcher
from kotoba_standalone.launcher import (
    LauncherOptions,
    LauncherTranslationOptions,
    asr_backend_from_label,
    asr_backend_label,
    build_process_command,
    build_translate_command,
    copy_korean_subtitles_to_input_location,
    default_app_for_extension,
    estimate_work_text,
    existing_korean_subtitles,
    expected_output_paths,
    ffmpeg_status_text,
    find_existing_japanese_subtitles,
    format_elapsed_korean,
    launcher_output_dir_from_state,
    launcher_state_from_values,
    media_path_for_subtitle,
    media_filetypes,
    ollama_server_text,
    pending_translation_subtitles,
    estimate_history_text,
    recent_work_time_text,
    summarize_existing_translation,
    summarize_progress_line,
    validate_input_path,
)


def test_format_elapsed_korean() -> None:
    assert format_elapsed_korean(0) == "0초"
    assert format_elapsed_korean(18 * 60 + 20) == "18분 20초"
    assert format_elapsed_korean(3661) == "1시간 1분 1초"


def test_summarize_progress_line_maps_common_steps() -> None:
    assert summarize_progress_line("Extracting audio with ffmpeg") == "음성을 추출하고 있습니다."
    assert summarize_progress_line("Translating subtitles 1-50") == "한국어로 번역하고 있습니다."
    assert summarize_progress_line("Done.") == "작업이 완료되었습니다."
    assert summarize_progress_line("") is None


def test_build_process_command_defaults_to_pyannote(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(input_path=tmp_path / "sample.mp4", output_dir=tmp_path / "out")
    )

    assert command[:4] == [sys.executable, "-m", "kotoba_standalone.cli", "process"]
    assert str(tmp_path / "sample.mp4") in command
    assert "--auto-silence-threshold" not in command
    assert command[command.index("--vad-engine") + 1] == "pyannote"
    assert "--asr-backend" not in command
    assert "--translate" not in command


def test_build_process_command_adds_qwen_backend(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
            asr_backend="qwen3",
        )
    )

    assert command[command.index("--asr-backend") + 1] == "qwen3"
    assert command[command.index("--model-dtype") + 1] == "bfloat16"


def test_asr_backend_label_helpers() -> None:
    assert asr_backend_from_label("Qwen3-ASR 1.7B (실험)") == "qwen3"
    assert asr_backend_from_label("Kotoba-Whisper v2.2") == "kotoba"
    assert asr_backend_label("qwen3").startswith("Qwen3-ASR")
    assert asr_backend_label("unknown") == "Kotoba-Whisper v2.2"


def test_build_process_command_adds_translation_options(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
            translate=True,
            translation_model="chosen:model",
            korean_style="strict-banmal",
            ollama_host="ollama.local",
            ollama_port=11435,
        )
    )

    assert "--translate" in command
    assert command[command.index("--translation-model") + 1] == "chosen:model"
    assert command[command.index("--korean-style") + 1] == "strict-banmal"
    assert command[command.index("--ollama-host") + 1] == "ollama.local"
    assert command[command.index("--ollama-port") + 1] == "11435"


def test_build_translate_command_writes_to_output_dir(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.ja.srt"
    command = build_translate_command(
        srt_path,
        LauncherTranslationOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
            translation_model="chosen:model",
            korean_style="banmal",
            ollama_host="ollama.local",
            ollama_port=11435,
        ),
    )

    assert command[:4] == [sys.executable, "-m", "kotoba_standalone.cli", "translate"]
    assert str(srt_path) in command
    assert command[command.index("--output") + 1] == str(tmp_path / "out")
    assert command[command.index("--model") + 1] == "chosen:model"
    assert command[command.index("--korean-style") + 1] == "banmal"
    assert command[command.index("--ollama-host") + 1] == "ollama.local"
    assert command[command.index("--ollama-port") + 1] == "11435"


def test_build_process_command_does_not_expose_ffmpeg_vad_options(tmp_path: Path) -> None:
    command = build_process_command(
        LauncherOptions(
            input_path=tmp_path / "sample.mp4",
            output_dir=tmp_path / "out",
        )
    )

    assert command[command.index("--vad-engine") + 1] == "pyannote"
    assert "--auto-silence-threshold" not in command
    assert "--report-subtitle-quality" not in command
    assert "--drop-likely-hallucinations" not in command
    assert "--split-long-subtitles" not in command
    assert "--annotate-subtitle-quality" not in command
    assert "--tail-retranscribe-long-subtitles" not in command


def test_media_filetypes_include_video_and_audio_patterns() -> None:
    filetypes = media_filetypes()

    assert "*.mp4" in filetypes[0][1]
    assert "*.wav" in filetypes[0][1]


def test_validate_input_path_accepts_supported_media_file(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_text("", encoding="utf-8")

    assert validate_input_path(media) is None


def test_validate_input_path_rejects_non_media_file(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("", encoding="utf-8")

    assert validate_input_path(document) == "동영상 또는 음성 파일만 선택할 수 있습니다."


def test_validate_input_path_rejects_folder_without_media(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("", encoding="utf-8")

    assert validate_input_path(tmp_path) == "선택한 폴더에 처리 가능한 동영상 또는 음성 파일이 없습니다."


def test_find_existing_japanese_subtitles_for_media_file(tmp_path: Path) -> None:
    media = tmp_path / "media" / "sample.mp4"
    output_dir = tmp_path / "out"
    media.parent.mkdir()
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    expected = output_dir / "sample.ja.srt"
    expected.write_text("", encoding="utf-8")

    assert find_existing_japanese_subtitles(media, output_dir) == [expected]


def test_find_existing_japanese_subtitles_for_folder_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "a.mp4").write_text("", encoding="utf-8")
    (input_dir / "b.mkv").write_text("", encoding="utf-8")
    first = output_dir / "a.ja.srt"
    second = output_dir / "b.ja.srt"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    (output_dir / "b.ko.srt").write_text("", encoding="utf-8")

    assert find_existing_japanese_subtitles(input_dir, output_dir) == [first, second]


def test_existing_korean_subtitles_reports_existing_outputs(tmp_path: Path) -> None:
    input_srt = tmp_path / "sample.ja.srt"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "sample.ko.srt"
    existing.write_text("", encoding="utf-8")

    assert existing_korean_subtitles([input_srt], output_dir) == [existing]


def test_expected_output_paths_include_output_and_source_subtitles(tmp_path: Path) -> None:
    media = tmp_path / "media" / "sample.mp4"
    output_dir = tmp_path / "out"
    media.parent.mkdir()
    media.write_text("", encoding="utf-8")

    paths = expected_output_paths(media, output_dir)

    assert output_dir / "sample.ja.srt" in paths
    assert output_dir / "sample.ko.srt" in paths
    assert media.with_suffix(".srt") in paths


def test_media_path_for_subtitle_matches_media_file(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    subtitle = Path("sample.ja.srt")
    media.write_text("", encoding="utf-8")

    assert media_path_for_subtitle(media, subtitle) == media


def test_copy_korean_subtitles_to_input_location_uses_primary_srt_when_missing(tmp_path: Path) -> None:
    media = tmp_path / "media" / "sample.mp4"
    output_dir = tmp_path / "out"
    media.parent.mkdir()
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    japanese = output_dir / "sample.ja.srt"
    japanese.write_text("", encoding="utf-8")
    korean = output_dir / "sample.ko.srt"
    korean.write_text("translated\n", encoding="utf-8")

    copied = copy_korean_subtitles_to_input_location(media, output_dir, [japanese])

    assert copied == [media.with_suffix(".srt")]
    assert media.with_suffix(".srt").read_text(encoding="utf-8") == "translated\n"


def test_copy_korean_subtitles_to_input_location_uses_ko_suffix_when_srt_exists(tmp_path: Path) -> None:
    media = tmp_path / "media" / "sample.mp4"
    output_dir = tmp_path / "out"
    media.parent.mkdir()
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    media.with_suffix(".srt").write_text("existing\n", encoding="utf-8")
    japanese = output_dir / "sample.ja.srt"
    japanese.write_text("", encoding="utf-8")
    korean = output_dir / "sample.ko.srt"
    korean.write_text("translated\n", encoding="utf-8")

    copied = copy_korean_subtitles_to_input_location(media, output_dir, [japanese])

    assert copied == [media.with_suffix(".ko.srt")]
    assert media.with_suffix(".srt").read_text(encoding="utf-8") == "existing\n"
    assert media.with_suffix(".ko.srt").read_text(encoding="utf-8") == "translated\n"


def test_copy_korean_subtitles_to_input_location_handles_folder_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    media = input_dir / "sample.mkv"
    media.write_text("", encoding="utf-8")
    japanese = output_dir / "sample.ja.srt"
    japanese.write_text("", encoding="utf-8")
    korean = output_dir / "sample.ko.srt"
    korean.write_text("translated\n", encoding="utf-8")

    copied = copy_korean_subtitles_to_input_location(input_dir, output_dir, [japanese])

    assert copied == [media.with_suffix(".srt")]
    assert media.with_suffix(".srt").read_text(encoding="utf-8") == "translated\n"


def test_summarize_existing_translation_counts_japanese_subtitles(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "sample.ja.srt").write_text("", encoding="utf-8")

    assert summarize_existing_translation(media, output_dir) == "1개"


def test_pending_translation_subtitles_excludes_completed_translation(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    japanese = output_dir / "sample.ja.srt"
    japanese.write_text("", encoding="utf-8")
    (output_dir / "sample.ko.srt").write_text("", encoding="utf-8")

    assert pending_translation_subtitles(media, output_dir) == []
    assert summarize_existing_translation(media, output_dir) == "없음 (번역 완료)"


def test_pending_translation_subtitles_checks_copied_media_subtitle(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "sample.ja.srt").write_text("", encoding="utf-8")
    media.with_suffix(".srt").write_text("", encoding="utf-8")

    assert pending_translation_subtitles(media, output_dir) == []


def test_find_existing_japanese_subtitles_for_folder_ignores_unrelated_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "current.mp4").write_text("", encoding="utf-8")
    current = output_dir / "current.ja.srt"
    current.write_text("", encoding="utf-8")
    (output_dir / "old.ja.srt").write_text("", encoding="utf-8")

    assert find_existing_japanese_subtitles(input_dir, output_dir) == [current]


def test_ffmpeg_status_text_reports_path_source(monkeypatch) -> None:
    monkeypatch.delenv(launcher.FFMPEG_PATH_ENV, raising=False)
    monkeypatch.setattr(launcher, "ffmpeg_exe", lambda: r"C:\Tools\ffmpeg.exe")

    assert ffmpeg_status_text().startswith("PATH 사용")


def test_default_app_for_extension_reports_system_default_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(launcher.os, "name", "posix")

    assert default_app_for_extension(".srt") == "시스템 기본 연결 앱"


def test_ollama_server_text_formats_host_and_port() -> None:
    assert ollama_server_text("example.local", "11435") == "example.local:11435"
    assert ollama_server_text("", "bad") == "localhost:11434"


def test_ffmpeg_status_text_reports_configured_external_path(tmp_path: Path) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_text("", encoding="utf-8")

    assert ffmpeg_status_text(str(ffmpeg)) == f"환경 변수 사용({ffmpeg})"


def test_ffmpeg_status_text_reports_missing_configured_path(tmp_path: Path) -> None:
    ffmpeg = tmp_path / "missing-ffmpeg.exe"

    assert ffmpeg_status_text(str(ffmpeg)) == f"지정 경로 확인 필요({ffmpeg})"


def test_estimate_work_text_uses_processing_and_translation_history(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "old.process.json").write_text(
        json.dumps({"media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    (output_dir / "old.translation.json").write_text(
        json.dumps({"subtitle_count": 10, "processing_seconds": 30}),
        encoding="utf-8",
    )
    (output_dir / "sample.ja.srt").write_text(
        "\n\n".join(
            f"{index}\n00:00:00,000 --> 00:00:01,000\ntext"
            for index in range(1, 6)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "probe_duration_seconds", lambda path: 200.0)

    assert estimate_work_text(media, output_dir, translate=True) == "예상 전사 40초 / 예상 번역 15초"


def test_estimate_work_text_filters_process_history_by_asr_backend(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "kotoba.process.json").write_text(
        json.dumps({"asr_backend": "kotoba", "media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    (output_dir / "qwen.process.json").write_text(
        json.dumps({"asr_backend": "qwen3", "audio_duration_seconds": 100, "processing_seconds": 50}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "probe_duration_seconds", lambda path: 200.0)

    assert estimate_work_text(media, output_dir, translate=False, asr_backend="kotoba") == (
        f"예상 전사 {format_elapsed_korean(40)}"
    )
    assert estimate_work_text(media, output_dir, translate=False, asr_backend="qwen3") == (
        f"예상 전사 {format_elapsed_korean(100)}"
    )


def test_estimate_work_text_reports_available_history_without_target_duration(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "old.process.json").write_text(
        json.dumps({"media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "probe_duration_seconds", lambda path: (_ for _ in ()).throw(RuntimeError("bad media")))

    assert estimate_work_text(media, output_dir, translate=False) == "입력 선택 시 계산 가능 (전사 이력 1개)"


def test_estimate_work_text_uses_configured_ffmpeg_path_for_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    ffmpeg = tmp_path / "tools" / "ffmpeg.exe"
    output_dir.mkdir()
    ffmpeg.parent.mkdir()
    media.write_text("", encoding="utf-8")
    ffmpeg.write_text("", encoding="utf-8")
    (output_dir / "old.process.json").write_text(
        json.dumps({"media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    seen_env: list[str | None] = []

    def fake_probe(path: Path) -> float:
        seen_env.append(launcher.os.environ.get(launcher.FFMPEG_PATH_ENV))
        return 200.0

    monkeypatch.delenv(launcher.FFMPEG_PATH_ENV, raising=False)
    monkeypatch.setattr(launcher, "probe_duration_seconds", fake_probe)

    assert estimate_work_text(media, output_dir, translate=False, ffmpeg_path=str(ffmpeg)) == "예상 전사 40초"
    assert seen_env == [str(ffmpeg)]
    assert launcher.os.environ.get(launcher.FFMPEG_PATH_ENV) is None


def test_estimate_history_text_counts_process_and_translation_history(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old.process.json").write_text(
        json.dumps({"media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    (output_dir / "old.translation.json").write_text(
        json.dumps({"subtitle_count": 10, "processing_seconds": 30}),
        encoding="utf-8",
    )

    assert estimate_history_text(output_dir) == "입력 선택 시 계산 가능 (전사 이력 1개, 번역 이력 1개)"


def test_estimate_work_text_sums_folder_media_durations(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    first = input_dir / "a.mp4"
    second = input_dir / "b.mkv"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    (output_dir / "old.process.json").write_text(
        json.dumps({"media_duration_seconds": 100, "processing_seconds": 20}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "probe_duration_seconds", lambda path: 100.0)

    assert estimate_work_text(input_dir, output_dir, translate=False) == "예상 전사 40초"


def test_recent_work_time_text_reports_completed_process_and_translation(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    media.write_text("", encoding="utf-8")
    (output_dir / "sample.process.json").write_text(
        json.dumps({"processing_seconds": 276}),
        encoding="utf-8",
    )
    (output_dir / "sample.ko.translation.json").write_text(
        json.dumps({"processing_seconds": 571}),
        encoding="utf-8",
    )

    assert recent_work_time_text(media, output_dir, include_translation=True) == "최근 결과 전사 4분 36초 / 최근 결과 번역 9분 31초"


def test_recent_work_time_text_sums_folder_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    first = input_dir / "a.mp4"
    second = input_dir / "b.mkv"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    for name, seconds in [("a", 10), ("b", 20)]:
        (output_dir / f"{name}.process.json").write_text(
            json.dumps({"processing_seconds": seconds}),
            encoding="utf-8",
        )
        (output_dir / f"{name}.ja.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
        (output_dir / f"{name}.ko.translation.json").write_text(
            json.dumps({"processing_seconds": seconds * 2}),
            encoding="utf-8",
        )

    assert recent_work_time_text(input_dir, output_dir, include_translation=True) == "최근 결과 전사 30초 / 최근 결과 번역 1분 0초"


def test_recent_work_time_text_ignores_unrelated_folder_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "media"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "current.mp4").write_text("", encoding="utf-8")
    (output_dir / "current.process.json").write_text(
        json.dumps({"processing_seconds": 10}),
        encoding="utf-8",
    )
    (output_dir / "current.ja.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
    (output_dir / "current.ko.translation.json").write_text(
        json.dumps({"processing_seconds": 20}),
        encoding="utf-8",
    )
    (output_dir / "old.process.json").write_text(
        json.dumps({"processing_seconds": 1000}),
        encoding="utf-8",
    )
    (output_dir / "old.ja.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
    (output_dir / "old.ko.translation.json").write_text(
        json.dumps({"processing_seconds": 2000}),
        encoding="utf-8",
    )

    assert recent_work_time_text(input_dir, output_dir, include_translation=True) == (
        f"최근 결과 전사 {format_elapsed_korean(10)} / 최근 결과 번역 {format_elapsed_korean(20)}"
    )


def test_launcher_state_from_values(tmp_path: Path) -> None:
    assert launcher_state_from_values("out", "cpu", "ffmpeg", "ollama.local", 11435, install_root=tmp_path) == {
        "install_root": str(tmp_path),
        "last_output_dir": "out",
        "last_model_device": "cpu",
        "external_ffmpeg_path": "ffmpeg",
        "ollama_host": "ollama.local",
        "ollama_port": 11435,
        "asr_backend": "kotoba",
    }


def test_launcher_output_dir_ignores_state_from_other_install(tmp_path: Path) -> None:
    current_root = tmp_path / "current"
    old_root = tmp_path / "old"
    state = launcher_state_from_values(
        str(old_root / "tmp-output"),
        "cuda:0",
        install_root=old_root,
    )

    assert launcher_output_dir_from_state(state, current_root) == current_root / "tmp-output"


def test_launcher_output_dir_reuses_state_for_same_install(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom-output"
    state = launcher_state_from_values(str(output_dir), "cuda:0", install_root=tmp_path)

    assert launcher_output_dir_from_state(state, tmp_path) == output_dir
