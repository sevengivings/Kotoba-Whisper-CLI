from __future__ import annotations

import queue
import json
import shutil
import subprocess
import sys
import threading
import time
import os
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, Listbox, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from kotoba_standalone.media import (
    AUDIO_EXTENSIONS,
    FFMPEG_PATH_ENV,
    VIDEO_EXTENSIONS,
    ffmpeg_exe,
    is_supported_media,
    probe_duration_seconds,
)
from kotoba_standalone.settings import (
    DEFAULT_TRANSLATION_MODEL,
    load_launcher_state,
    load_saved_translation_model,
    save_launcher_state,
)
from kotoba_standalone.translate.ollama import (
    OllamaUnavailableError,
    TranslationOptions,
    default_output_srt,
    format_ollama_model_choice,
    get_ollama_models,
    sort_ollama_models_for_translation,
)


@dataclass(frozen=True)
class LauncherOptions:
    input_path: Path
    output_dir: Path
    translate: bool = False
    translation_model: str = ""
    korean_style: str = "polite"
    model_device: str = "cuda:0"
    asr_backend: str = "kotoba"
    ollama_host: str = "localhost"
    ollama_port: int = 11434


@dataclass(frozen=True)
class LauncherTranslationOptions:
    input_path: Path
    output_dir: Path
    translation_model: str = ""
    korean_style: str = "polite"
    ollama_host: str = "localhost"
    ollama_port: int = 11434


def format_elapsed_korean(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours}시간 {minutes}분 {secs}초"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def build_process_command(options: LauncherOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "kotoba_standalone.cli",
        "process",
        str(options.input_path),
        "--output-dir",
        str(options.output_dir),
        "--model-device",
        options.model_device,
        "--vad-engine",
        "pyannote",
    ]
    if options.asr_backend == "qwen3":
        command.extend(["--asr-backend", "qwen3", "--model-dtype", "bfloat16"])
    if options.translate:
        command.append("--translate")
        if options.translation_model.strip():
            command.extend(["--translation-model", options.translation_model.strip()])
        command.extend(
            [
                "--korean-style",
                options.korean_style,
                "--ollama-host",
                options.ollama_host,
                "--ollama-port",
                str(options.ollama_port),
            ]
        )
    return command


def qwen_environment_status_text() -> str:
    if qwen_environment_python().exists():
        return "설치됨 (.venv-qwen)"
    return "설치 필요 (install-qwen3.bat)"


def qwen_environment_python() -> Path:
    root = Path(__file__).resolve().parents[2]
    if sys.platform == "win32":
        return root / ".venv-qwen" / "Scripts" / "python.exe"
    return root / ".venv-qwen" / "bin" / "python"


def asr_backend_from_label(label: str) -> str:
    return "qwen3" if "Qwen3" in label else "kotoba"


def asr_backend_label(value: str) -> str:
    if value == "qwen3":
        return "Qwen3-ASR 1.7B (실험)"
    return "Kotoba-Whisper v2.2"


def build_translate_command(input_srt: Path, options: LauncherTranslationOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "kotoba_standalone.cli",
        "translate",
        str(input_srt),
        "--output",
        str(options.output_dir),
    ]
    if options.translation_model.strip():
        command.extend(["--model", options.translation_model.strip()])
    command.extend(
        [
            "--korean-style",
            options.korean_style,
            "--ollama-host",
            options.ollama_host,
            "--ollama-port",
            str(options.ollama_port),
        ]
    )
    return command


def find_existing_japanese_subtitles(input_path: Path, output_dir: Path) -> list[Path]:
    if input_path.is_file():
        candidate = output_dir / f"{input_path.stem}.ja.srt"
        return [candidate] if candidate.exists() else []
    if input_path.is_dir():
        return sorted(path for path in output_dir.glob("*.ja.srt") if path.is_file())
    return []


def existing_korean_subtitles(input_srt_paths: list[Path], output_dir: Path) -> list[Path]:
    return [
        output_dir / default_output_srt(input_srt).name
        for input_srt in input_srt_paths
        if (output_dir / default_output_srt(input_srt).name).exists()
    ]


def pending_translation_subtitles(input_path: Path, output_dir: Path) -> list[Path]:
    return [
        subtitle
        for subtitle in find_existing_japanese_subtitles(input_path, output_dir)
        if not translated_subtitle_exists(subtitle, input_path, output_dir)
    ]


def translated_subtitle_exists(input_srt: Path, input_path: Path, output_dir: Path) -> bool:
    candidates = [output_dir / default_output_srt(input_srt).name]
    if input_path.is_file():
        candidates.extend(
            [
                input_path.with_suffix(".srt"),
                input_path.with_name(f"{input_path.stem}.ko.srt"),
            ]
        )
    return any(path.exists() for path in candidates)


def expected_output_paths(input_path: Path, output_dir: Path) -> list[Path]:
    if input_path.is_file():
        return [
            output_dir / f"{input_path.stem}.ja.srt",
            output_dir / f"{input_path.stem}.ko.srt",
            output_dir / f"{input_path.stem}.srt",
            input_path.with_suffix(".srt"),
            input_path.with_name(f"{input_path.stem}.ko.srt"),
        ]
    if input_path.is_dir():
        return sorted(output_dir.glob("*.srt"))
    return []


def media_path_for_subtitle(input_path: Path, input_srt: Path) -> Path | None:
    stem = input_srt.name.removesuffix(".ja.srt") if input_srt.name.endswith(".ja.srt") else input_srt.stem
    if input_path.is_file():
        return input_path if input_path.stem == stem and is_supported_media(input_path) else None
    if not input_path.is_dir():
        return None
    for candidate in sorted(input_path.iterdir()):
        if candidate.is_file() and candidate.stem == stem and is_supported_media(candidate):
            return candidate
    return None


def copy_korean_subtitles_to_input_location(input_path: Path, output_dir: Path, input_srt_paths: list[Path]) -> list[Path]:
    copied_paths: list[Path] = []
    for input_srt in input_srt_paths:
        ko_srt_path = output_dir / default_output_srt(input_srt).name
        media_path = media_path_for_subtitle(input_path, input_srt)
        if media_path is None or not ko_srt_path.exists():
            continue
        primary_target = media_path.with_suffix(".srt")
        target = primary_target if not primary_target.exists() else media_path.with_suffix(".ko.srt")
        shutil.copy2(ko_srt_path, target)
        copied_paths.append(target)
    return copied_paths


def ffmpeg_status_text(configured_path: str = "") -> str:
    configured_path = configured_path.strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.exists():
            return f"환경 변수 사용({path})"
        return f"지정 경로 확인 필요({path})"
    try:
        selected = ffmpeg_exe()
    except Exception as exc:
        return f"확인 실패 ({exc})"
    normalized = selected.replace("\\", "/").lower()
    if os.environ.get(FFMPEG_PATH_ENV):
        source = "환경 변수"
    elif "imageio_ffmpeg" in normalized:
        source = "번들"
    else:
        source = "PATH"
    return f"{source} 사용({selected})"


def summarize_existing_translation(input_path: Path, output_dir: Path) -> str:
    subtitles = find_existing_japanese_subtitles(input_path, output_dir)
    if not subtitles:
        return "없음"
    pending = pending_translation_subtitles(input_path, output_dir)
    if not pending:
        return "없음 (번역 완료)"
    return f"{len(pending)}개"


def estimate_work_text(input_path: Path, output_dir: Path, translate: bool, ffmpeg_path: str = "") -> str:
    process_seconds = _estimate_process_seconds(input_path, output_dir, ffmpeg_path)
    translation_seconds = _estimate_translation_seconds(input_path, output_dir) if translate else None
    parts: list[str] = []
    if process_seconds is not None:
        parts.append(f"예상 전사 {format_elapsed_korean(process_seconds)}")
    if translation_seconds is not None:
        parts.append(f"예상 번역 {format_elapsed_korean(translation_seconds)}")
    return " / ".join(parts) if parts else estimate_history_text(output_dir)


def recent_work_time_text(input_path: Path, output_dir: Path, include_translation: bool) -> str | None:
    parts: list[str] = []
    process_seconds = _process_seconds_for_input(input_path, output_dir)
    if process_seconds is not None:
        parts.append(f"최근 결과 전사 {format_elapsed_korean(process_seconds)}")
    if include_translation:
        translation_seconds = _translation_seconds_for_input(input_path, output_dir)
        if translation_seconds is not None:
            parts.append(f"최근 결과 번역 {format_elapsed_korean(translation_seconds)}")
    return " / ".join(parts) if parts else None


def estimate_history_text(output_dir: Path) -> str:
    process_count = len(_process_history_ratios(output_dir))
    translation_count = len(_translation_history_ratios(output_dir))
    details: list[str] = []
    if process_count:
        details.append(f"전사 이력 {process_count}개")
    if translation_count:
        details.append(f"번역 이력 {translation_count}개")
    if not details:
        return "이력 부족"
    return f"입력 선택 시 계산 가능 ({', '.join(details)})"


def summarize_progress_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("> "):
        return "작업을 준비하고 있습니다."
    if "Extracting audio" in text:
        return "음성을 추출하고 있습니다."
    if "Audio extracted" in text:
        return "음성 추출을 완료했습니다."
    if "Loading pyannote" in text:
        return "음성 구간 분석 모델을 불러오고 있습니다."
    if "Detecting speech" in text or "Detected speech" in text:
        return "말소리 구간을 찾고 있습니다."
    if "Transcribing" in text or "transcription" in text.lower():
        return "일본어 자막을 생성하고 있습니다."
    if "Translating subtitles" in text:
        return "한국어로 번역하고 있습니다."
    if "Retrying subtitle" in text:
        return "빠진 번역을 다시 확인하고 있습니다."
    if "Korean translation completed" in text or "Korean SRT:" in text:
        return "한국어 자막을 저장했습니다."
    if "Japanese SRT:" in text:
        return "일본어 자막을 저장했습니다."
    if text == "Done." or "완료되었습니다" in text:
        return "작업이 완료되었습니다."
    if "Failed." in text or "실패했습니다" in text:
        return "작업이 실패했습니다. 자세한 로그를 확인하세요."
    return None


def ollama_server_text(host: str, port: int | str) -> str:
    host = host.strip() or "localhost"
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        port_number = 11434
    return f"{host}:{port_number}"


def default_app_for_extension(extension: str) -> str:
    if os.name != "nt":
        return "시스템 기본 연결 앱"
    try:
        import winreg
    except ImportError:
        return "Windows 기본 연결 앱"

    prog_id = _registry_value(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\{extension}\UserChoice", "ProgId")
    if not prog_id:
        prog_id = _registry_value(winreg.HKEY_CLASSES_ROOT, extension, "")
    if not prog_id:
        return "Windows 기본 연결 앱"

    display_name = _registry_value(winreg.HKEY_CLASSES_ROOT, prog_id, "")
    command = _registry_value(winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command", "")
    if display_name and command:
        return f"{display_name} ({command})"
    if command:
        return command
    return display_name or prog_id


def _registry_value(root: object, sub_key: str, value_name: str) -> str:
    try:
        import winreg

        with winreg.OpenKey(root, sub_key) as key:
            value, _value_type = winreg.QueryValueEx(key, value_name)
    except OSError:
        return ""
    return str(value).strip() if value else ""


def launcher_state_from_values(
    output_dir: str,
    model_device: str,
    ffmpeg_path: str = "",
    ollama_host: str = "localhost",
    ollama_port: int = 11434,
    asr_backend: str = "kotoba",
    install_root: Path | None = None,
) -> dict:
    root = install_root or standalone_root()
    return {
        "install_root": str(root),
        "last_output_dir": output_dir,
        "last_model_device": model_device,
        "external_ffmpeg_path": ffmpeg_path,
        "ollama_host": ollama_host,
        "ollama_port": ollama_port,
        "asr_backend": asr_backend if asr_backend in {"kotoba", "qwen3"} else "kotoba",
    }


def standalone_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launcher_output_dir_from_state(state: dict, app_root: Path | None = None) -> Path:
    root = app_root or standalone_root()
    if str(state.get("install_root") or "") == str(root):
        saved_output = state.get("last_output_dir")
        if isinstance(saved_output, str) and saved_output.strip():
            return Path(saved_output)
    return root / "tmp-output"


def media_filetypes() -> list[tuple[str, str]]:
    video_patterns = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
    audio_patterns = " ".join(f"*{extension}" for extension in sorted(AUDIO_EXTENSIONS))
    all_patterns = " ".join([video_patterns, audio_patterns])
    return [
        ("Video and audio files", all_patterns),
        ("Video files", video_patterns),
        ("Audio files", audio_patterns),
    ]


def validate_input_path(path: Path) -> str | None:
    if not path.exists():
        return "선택한 입력 경로가 존재하지 않습니다."
    if path.is_file() and not is_supported_media(path):
        return "동영상 또는 음성 파일만 선택할 수 있습니다."
    if path.is_dir() and not any(child.is_file() and is_supported_media(child) for child in path.iterdir()):
        return "선택한 폴더에 처리 가능한 동영상 또는 음성 파일이 없습니다."
    return None


def _estimate_process_seconds(input_path: Path, output_dir: Path, ffmpeg_path: str = "") -> float | None:
    target_duration = _target_media_duration_seconds(input_path, output_dir, ffmpeg_path)
    ratios = _process_history_ratios(output_dir)
    if not ratios:
        return None
    average_ratio = sum(ratios[-10:]) / min(len(ratios), 10)
    if target_duration:
        return target_duration * average_ratio
    return None


def _estimate_translation_seconds(input_path: Path, output_dir: Path) -> float | None:
    target_count = _target_subtitle_count(input_path, output_dir)
    ratios = _translation_history_ratios(output_dir)
    if not ratios:
        return None
    average_ratio = sum(ratios[-10:]) / min(len(ratios), 10)
    if target_count:
        return target_count * average_ratio
    return None


def _process_history_ratios(output_dir: Path) -> list[float]:
    ratios: list[float] = []
    if not output_dir.exists():
        return ratios
    for metadata_path in output_dir.glob("*.process.json"):
        data = _read_json(metadata_path)
        media_duration = _positive_float(data.get("media_duration_seconds"))
        processing_seconds = _positive_float(data.get("processing_seconds"))
        if media_duration and processing_seconds:
            ratios.append(processing_seconds / media_duration)
    return ratios


def _translation_history_ratios(output_dir: Path) -> list[float]:
    ratios: list[float] = []
    if not output_dir.exists():
        return ratios
    for metadata_path in output_dir.glob("*.translation.json"):
        data = _read_json(metadata_path)
        subtitle_count = _positive_float(data.get("subtitle_count"))
        processing_seconds = _positive_float(data.get("processing_seconds"))
        if subtitle_count and processing_seconds:
            ratios.append(processing_seconds / subtitle_count)
    return ratios


def _target_media_duration_seconds(input_path: Path, output_dir: Path, ffmpeg_path: str = "") -> float | None:
    if input_path.is_file():
        target_duration = _media_duration_from_process_json(output_dir / f"{input_path.stem}.process.json")
        return target_duration if target_duration is not None else _safe_probe_duration(input_path, ffmpeg_path)
    if input_path.is_dir():
        total = 0.0
        for child in input_path.iterdir():
            if child.is_file() and is_supported_media(child):
                duration = _safe_probe_duration(child, ffmpeg_path)
                if duration:
                    total += duration
        return total or None
    return None


def _target_subtitle_count(input_path: Path, output_dir: Path) -> int | None:
    if input_path.is_file():
        return _subtitle_count(output_dir / f"{input_path.stem}.ja.srt")
    if input_path.is_dir():
        total = 0
        for subtitle in find_existing_japanese_subtitles(input_path, output_dir):
            count = _subtitle_count(subtitle)
            if count:
                total += count
        return total or None
    return None


def _safe_probe_duration(input_path: Path, ffmpeg_path: str = "") -> float | None:
    old_ffmpeg_path = os.environ.get(FFMPEG_PATH_ENV)
    try:
        if ffmpeg_path.strip():
            os.environ[FFMPEG_PATH_ENV] = ffmpeg_path.strip()
        return probe_duration_seconds(input_path)
    except Exception:
        return None
    finally:
        if ffmpeg_path.strip():
            if old_ffmpeg_path is None:
                os.environ.pop(FFMPEG_PATH_ENV, None)
            else:
                os.environ[FFMPEG_PATH_ENV] = old_ffmpeg_path


def _media_duration_from_process_json(path: Path) -> float | None:
    data = _read_json(path)
    return _positive_float(data.get("media_duration_seconds"))


def _process_seconds_for_input(input_path: Path, output_dir: Path) -> float | None:
    if input_path.is_file():
        return _positive_float(_read_json(output_dir / f"{input_path.stem}.process.json").get("processing_seconds"))
    if input_path.is_dir():
        total = 0.0
        for media_path in input_path.iterdir():
            if media_path.is_file() and is_supported_media(media_path):
                seconds = _positive_float(_read_json(output_dir / f"{media_path.stem}.process.json").get("processing_seconds"))
                if seconds:
                    total += seconds
        return total or None
    return None


def _translation_seconds_for_input(input_path: Path, output_dir: Path) -> float | None:
    if input_path.is_file():
        return _positive_float(_read_json(output_dir / f"{input_path.stem}.ko.translation.json").get("processing_seconds"))
    if input_path.is_dir():
        total = 0.0
        for subtitle in find_existing_japanese_subtitles(input_path, output_dir):
            ko_srt = output_dir / default_output_srt(subtitle).name
            seconds = _positive_float(_read_json(ko_srt.with_suffix(".translation.json")).get("processing_seconds"))
            if seconds:
                total += seconds
        return total or None
    return None


def _subtitle_count(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    count = sum(1 for block in text.replace("\r\n", "\n").split("\n\n") if "-->" in block)
    return count or None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def configure_theme(root: Tk) -> None:
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    elif "vista" in style.theme_names():
        style.theme_use("vista")

    root.option_add("*Font", "{Segoe UI} 10")
    background = "#f5f7fa"
    panel = "#ffffff"
    border = "#d9dee7"
    text = "#1f2937"
    muted = "#4b5563"
    accent = "#2563eb"

    style.configure(".", background=background, foreground=text)
    style.configure("TFrame", background=background)
    style.configure("Panel.TFrame", background=panel)
    style.configure("TLabel", background=background, foreground=text)
    style.configure("Panel.TLabel", background=panel, foreground=text)
    style.configure("Muted.TLabel", background=background, foreground=muted)
    style.configure("TCheckbutton", background=background, foreground=text)
    style.configure("TLabelframe", background=panel, bordercolor=border, lightcolor=border, darkcolor=border, padding=(12, 10))
    style.configure("TLabelframe.Label", background=background, foreground=text, padding=(2, 0))
    style.configure(
        "TButton",
        background="#f8fafc",
        foreground=text,
        bordercolor="#c7ceda",
        lightcolor="#ffffff",
        darkcolor="#c7ceda",
        padding=(12, 6),
    )
    style.map(
        "TButton",
        background=[("active", "#eef2f7"), ("pressed", "#e5eaf1"), ("disabled", "#eef0f3")],
        foreground=[("disabled", "#9aa3af")],
    )
    style.configure(
        "Accent.TButton",
        background=accent,
        foreground="#ffffff",
        bordercolor=accent,
        lightcolor=accent,
        darkcolor=accent,
        padding=(14, 6),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#1d4ed8"), ("pressed", "#1e40af"), ("disabled", "#d5dbe6")],
        foreground=[("disabled", "#8b95a1")],
    )
    style.configure("TEntry", fieldbackground="#ffffff", bordercolor="#c7ceda", padding=(6, 5))
    style.configure(
        "TCombobox",
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=text,
        bordercolor="#c7ceda",
        padding=(6, 5),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#ffffff"), ("focus", "#ffffff")],
        foreground=[("readonly", text), ("focus", text)],
        selectbackground=[("readonly", "#ffffff"), ("focus", "#ffffff")],
        selectforeground=[("readonly", text), ("focus", text)],
    )
    style.configure("Horizontal.TProgressbar", background=accent, troughcolor="#e6eaf0", bordercolor="#e6eaf0")
    root.configure(background=background)


class KotobaLauncher:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Kotoba Standalone")
        self.root.geometry("860x620")
        self.root.minsize(820, 620)
        self.events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.last_result_paths: list[Path] = []
        self.pending_translation_copy: tuple[Path, Path, list[Path]] | None = None
        self.log_buffer = ""
        self.log_window: Toplevel | None = None
        self.log_widget: ScrolledText | None = None
        self.app_root = standalone_root()
        state = load_launcher_state()

        self.input_path = StringVar(value="")
        self.output_dir = StringVar(value=str(launcher_output_dir_from_state(state, self.app_root)))
        self.translate = BooleanVar(value=False)
        self.model = StringVar(value=load_saved_translation_model() or DEFAULT_TRANSLATION_MODEL)
        self.korean_style = StringVar(value="polite")
        self.model_device = StringVar(value=str(state.get("last_model_device") or "cuda:0"))
        self.asr_engine = StringVar(value=asr_backend_label(str(state.get("asr_backend") or "kotoba")))
        self.external_ffmpeg_path = StringVar(value=str(state.get("external_ffmpeg_path") or ""))
        self.ollama_host = StringVar(value=str(state.get("ollama_host") or "localhost"))
        self.ollama_port = StringVar(value=str(state.get("ollama_port") or "11434"))
        self.status = StringVar(value="대기 중")
        self.ffmpeg_status = StringVar(value=ffmpeg_status_text(self.external_ffmpeg_path.get()))
        self.ollama_status = StringVar(value=ollama_server_text(self.ollama_host.get(), self.ollama_port.get()))
        self.qwen_status = StringVar(value=qwen_environment_status_text())
        self.translation_status = StringVar(value="확인 전")
        self.estimate_status = StringVar(value="이력 부족")
        self.progress_summary = StringVar(value="대기 중")
        self.progress_detail = StringVar(value="입력 영상 또는 폴더를 선택한 뒤 시작하세요.")

        self._build_ui()
        self.translate.trace_add("write", lambda *_args: self._refresh_derived_status())
        self.input_path.trace_add("write", lambda *_args: self._refresh_derived_status())
        self.output_dir.trace_add("write", lambda *_args: self._refresh_derived_status())
        self.model_device.trace_add("write", lambda *_args: self._remember_state())
        self.asr_engine.trace_add("write", lambda *_args: self._refresh_qwen_status())
        self.external_ffmpeg_path.trace_add("write", lambda *_args: self._refresh_ffmpeg_status())
        self.ollama_host.trace_add("write", lambda *_args: self._refresh_ollama_status())
        self.ollama_port.trace_add("write", lambda *_args: self._refresh_ollama_status())
        self._refresh_derived_status()
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        form_pady = 3

        ttk.Label(outer, text="입력 영상 또는 폴더").grid(row=0, column=0, sticky="w", pady=form_pady)
        ttk.Entry(outer, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", padx=8, pady=form_pady)
        ttk.Button(outer, text="파일 선택", command=self.choose_file).grid(row=0, column=2, padx=3, pady=form_pady)
        ttk.Button(outer, text="폴더 선택", command=self.choose_folder).grid(row=0, column=3, padx=3, pady=form_pady)

        ttk.Label(outer, text="작업 폴더").grid(row=1, column=0, sticky="w", pady=form_pady)
        ttk.Entry(outer, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8, pady=form_pady)
        ttk.Button(outer, text="선택", command=self.choose_output_dir).grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=3, pady=form_pady
        )

        ttk.Label(outer, text="전사 엔진").grid(row=2, column=0, sticky="w", pady=form_pady)
        ttk.Combobox(
            outer,
            textvariable=self.asr_engine,
            values=("Kotoba-Whisper v2.2", "Qwen3-ASR 1.7B (실험)"),
            state="readonly",
            width=24,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=form_pady)

        ttk.Label(outer, text="번역 모델").grid(row=3, column=0, sticky="w", pady=form_pady)
        ttk.Entry(outer, textvariable=self.model).grid(row=3, column=1, sticky="ew", padx=8, pady=form_pady)
        ttk.Button(outer, text="Ollama 모델", command=self.load_ollama_models).grid(
            row=3, column=2, sticky="ew", padx=3, pady=form_pady
        )
        ttk.Button(outer, text="Ollama 확인", command=self.check_ollama).grid(
            row=3, column=3, sticky="ew", padx=3, pady=form_pady
        )

        ttk.Label(outer, text="한국어 말투").grid(row=4, column=0, sticky="w", pady=form_pady)
        ttk.Combobox(
            outer,
            textvariable=self.korean_style,
            values=("polite", "banmal", "strict-banmal"),
            state="readonly",
            width=18,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=form_pady)

        ttk.Label(outer, text="처리 장치").grid(row=5, column=0, sticky="w", pady=form_pady)
        ttk.Combobox(
            outer,
            textvariable=self.model_device,
            values=("cuda:0", "cpu"),
            width=18,
        ).grid(row=5, column=1, sticky="w", padx=8, pady=form_pady)

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        buttons.columnconfigure(0, weight=1)
        ttk.Checkbutton(buttons, text="한국어 번역까지 실행", variable=self.translate).grid(row=0, column=1, padx=(0, 12))
        self.run_button = ttk.Button(buttons, text="시작", command=self.start, style="Accent.TButton")
        self.run_button.grid(row=0, column=2, padx=4)
        self.translate_button = ttk.Button(buttons, text="번역", command=self.translate_existing)
        self.translate_button.grid(row=0, column=3, padx=4)
        self.stop_button = ttk.Button(buttons, text="중지", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=4, padx=4)
        self.open_output_button = ttk.Button(buttons, text="작업 폴더 열기", command=self.open_output_dir)
        self.open_output_button.grid(row=0, column=5, padx=4)
        self.open_subtitle_button = ttk.Button(buttons, text="자막 열기", command=self.open_latest_subtitle)
        self.open_subtitle_button.grid(row=0, column=6, padx=4)

        progress_panel = ttk.LabelFrame(outer, text="진행", padding=(10, 8))
        progress_panel.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        progress_panel.columnconfigure(1, weight=1)
        ttk.Label(progress_panel, text="처리 시간:", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(progress_panel, textvariable=self.status, style="Panel.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(progress_panel, text="진행 상태:", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(3, 0)
        )
        ttk.Label(progress_panel, textvariable=self.progress_summary, style="Panel.TLabel").grid(
            row=1, column=1, sticky="w", pady=(3, 0)
        )
        ttk.Label(progress_panel, text="세부 정보:", style="Panel.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=(2, 6)
        )
        ttk.Label(progress_panel, textvariable=self.progress_detail, style="Panel.TLabel").grid(
            row=2, column=1, sticky="w", pady=(2, 6)
        )
        self.progress_bar = ttk.Progressbar(progress_panel, mode="determinate", value=0)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

        log_buttons = ttk.Frame(progress_panel, style="Panel.TFrame")
        log_buttons.grid(row=0, column=2, rowspan=4, sticky="ne", padx=(10, 0))
        self.toggle_log_button = ttk.Button(log_buttons, text="자세한 로그 보기", command=self.toggle_log)
        self.toggle_log_button.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(log_buttons, text="로그 복사", command=self.copy_log).grid(row=1, column=0, sticky="ew")

        status_panel = ttk.LabelFrame(outer, text="상태", padding=(10, 6))
        status_panel.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        status_panel.columnconfigure(0, weight=1)
        status_panel.columnconfigure(1, weight=0)

        status_text = ttk.Frame(status_panel, style="Panel.TFrame")
        status_text.grid(row=0, column=0, sticky="ew")
        status_text.columnconfigure(1, weight=1)
        status_actions = ttk.Frame(status_panel, style="Panel.TFrame")
        status_actions.grid(row=0, column=1, sticky="se", padx=(10, 0))

        status_rows = [
            ("번역 대기 파일:", self.translation_status),
            ("처리 시간:", self.estimate_status),
            ("음성 추출 방법:", self.ffmpeg_status),
            ("Qwen3-ASR:", self.qwen_status),
            ("Ollama 서버:", self.ollama_status),
        ]
        for row, (label, variable) in enumerate(status_rows):
            ttk.Label(status_text, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
            ttk.Label(status_text, textvariable=variable, style="Panel.TLabel").grid(row=row, column=1, sticky="w", pady=2)

        ttk.Button(status_actions, text="외부 FFmpeg 지정", command=self.choose_external_ffmpeg).grid(
            row=0, column=0, sticky="ew", pady=2
        )
        ttk.Button(status_actions, text="주소/포트 변경", command=self.change_ollama_server).grid(
            row=1, column=0, sticky="ew", pady=2
        )

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="처리할 동영상 또는 음성 파일을 선택하세요",
            filetypes=media_filetypes(),
        )
        if selected:
            self.input_path.set(selected)
            self._remember_state()

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="처리할 영상 폴더를 선택하세요")
        if selected:
            self.input_path.set(selected)
            self._remember_state()

    def choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="작업 폴더를 선택하세요")
        if selected:
            self.output_dir.set(selected)
            self._remember_state()

    def choose_external_ffmpeg(self) -> None:
        selected = filedialog.askopenfilename(
            title="사용할 ffmpeg.exe를 선택하세요",
            filetypes=[("FFmpeg executable", "ffmpeg.exe"), ("Executable files", "*.exe"), ("All files", "*.*")],
        )
        if selected:
            self.external_ffmpeg_path.set(selected)
            self._remember_state()

    def load_ollama_models(self) -> None:
        ollama_port = self._ollama_port_or_warn()
        if ollama_port is None:
            return
        try:
            models = sort_ollama_models_for_translation(
                get_ollama_models(
                    TranslationOptions(
                        model=self.model.get() or DEFAULT_TRANSLATION_MODEL,
                        ollama_host=self.ollama_host.get().strip() or "localhost",
                        ollama_port=ollama_port,
                    )
                )
            )
        except OllamaUnavailableError as exc:
            messagebox.showwarning("Ollama 연결 실패", str(exc))
            return
        if not models:
            messagebox.showinfo("Ollama 모델", "다운로드된 Ollama 모델이 없습니다.")
            return
        ModelDialog(self.root, models, self.model)

    def check_ollama(self) -> None:
        ollama_port = self._ollama_port_or_warn()
        if ollama_port is None:
            return
        try:
            models = get_ollama_models(
                TranslationOptions(
                    model=self.model.get() or DEFAULT_TRANSLATION_MODEL,
                    ollama_host=self.ollama_host.get().strip() or "localhost",
                    ollama_port=ollama_port,
                )
            )
        except OllamaUnavailableError as exc:
            messagebox.showwarning("Ollama 연결 실패", str(exc))
            return
        sorted_models = sort_ollama_models_for_translation(models)
        recommended = [model for model in sorted_models if format_ollama_model_choice(model).startswith("[번역 추천]")]
        discouraged = [model for model in sorted_models if format_ollama_model_choice(model).startswith("[실험용/비추천]")]
        lines = [
            f"Ollama 연결 성공",
            f"설치된 모델: {len(models)}개",
            f"번역 추천 모델: {len(recommended)}개",
            f"실험용/비추천 모델: {len(discouraged)}개",
        ]
        if recommended:
            lines.extend(["", "추천 모델:", *recommended[:5]])
        else:
            lines.extend(["", "추천 모델이 없습니다.", "README의 Hy-MT2 7B 또는 30B 모델 설치 안내를 참고하세요."])
        messagebox.showinfo("Ollama 확인", "\n".join(lines))

    def change_ollama_server(self) -> None:
        OllamaServerDialog(self.root, self.ollama_host, self.ollama_port)

    def start(self) -> None:
        input_text = self.input_path.get().strip()
        if not input_text:
            messagebox.showwarning("입력 필요", "처리할 영상 파일 또는 폴더를 선택하세요.")
            return
        validation_error = validate_input_path(Path(input_text))
        if validation_error is not None:
            messagebox.showwarning("입력 확인", validation_error)
            return
        ollama_port = self._ollama_port_or_warn()
        if ollama_port is None:
            return
        asr_backend = asr_backend_from_label(self.asr_engine.get())
        if asr_backend == "qwen3" and not qwen_environment_python().exists():
            messagebox.showwarning(
                "Qwen3-ASR 설치 필요",
                "Qwen3-ASR을 사용하려면 먼저 standalone\\install-qwen3.bat를 실행해 주세요.",
            )
            self._refresh_qwen_status()
            return
        options = LauncherOptions(
            input_path=Path(input_text),
            output_dir=Path(self.output_dir.get().strip() or self.app_root / "tmp-output"),
            translate=self.translate.get(),
            translation_model=self.model.get(),
            korean_style=self.korean_style.get(),
            model_device=self.model_device.get().strip() or "cuda:0",
            asr_backend=asr_backend,
            ollama_host=self.ollama_host.get().strip() or "localhost",
            ollama_port=ollama_port,
        )
        command = build_process_command(options)
        self._remember_state()
        self.pending_translation_copy = None
        self._clear_log()
        self.progress_summary.set("작업을 시작합니다.")
        self.progress_detail.set(Path(input_text).name)
        self._append_log("> " + " ".join(command) + "\n\n")
        self._set_running_buttons()
        self.started_at = time.monotonic()
        self.status.set("처리 중 0초 경과")
        self._start_progress_bar()
        self._update_elapsed_status()
        threading.Thread(target=self._run_commands, args=([command],), daemon=True).start()

    def translate_existing(self) -> None:
        input_text = self.input_path.get().strip()
        if not input_text:
            messagebox.showwarning("입력 필요", "처리한 영상 파일 또는 폴더를 선택하세요.")
            return
        input_path = Path(input_text)
        output_dir = Path(self.output_dir.get().strip() or self.app_root / "tmp-output")
        validation_error = validate_input_path(input_path)
        if validation_error is not None:
            messagebox.showwarning("입력 확인", validation_error)
            return
        srt_paths = find_existing_japanese_subtitles(input_path, output_dir)
        if not srt_paths:
            messagebox.showwarning("번역할 자막 없음", "작업 폴더에서 일본어 자막(*.ja.srt)을 찾지 못했습니다.")
            return
        ollama_port = self._ollama_port_or_warn()
        if ollama_port is None:
            return
        existing_outputs = existing_korean_subtitles(srt_paths, output_dir)
        if existing_outputs:
            names = "\n".join(path.name for path in existing_outputs[:5])
            if len(existing_outputs) > 5:
                names += f"\n... 외 {len(existing_outputs) - 5}개"
            if not messagebox.askyesno("번역 자막 덮어쓰기", f"이미 한국어 자막이 있습니다. 덮어쓸까요?\n\n{names}"):
                return
        options = LauncherTranslationOptions(
            input_path=input_path,
            output_dir=output_dir,
            translation_model=self.model.get(),
            korean_style=self.korean_style.get(),
            ollama_host=self.ollama_host.get().strip() or "localhost",
            ollama_port=ollama_port,
        )
        commands = [build_translate_command(srt_path, options) for srt_path in srt_paths]
        self._remember_state()
        self.pending_translation_copy = (input_path, output_dir, srt_paths)
        self._clear_log()
        self.progress_summary.set("번역을 시작합니다.")
        self.progress_detail.set(f"대상 자막 {len(commands)}개")
        for command in commands:
            self._append_log("> " + " ".join(command) + "\n")
        self._append_log("\n")
        self._set_running_buttons()
        self.started_at = time.monotonic()
        self.status.set("번역 중 0초 경과")
        self._start_progress_bar()
        self._update_elapsed_status("번역 중")
        threading.Thread(target=self._run_commands, args=(commands,), daemon=True).start()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.started_at = None
            self._stop_progress_bar()
            self.progress_summary.set("중지를 요청했습니다.")
            self.progress_detail.set("현재 작업이 정리되는 중입니다.")
            self.process.terminate()
            self.status.set("중지 요청됨")

    def _run_commands(self, commands: list[list[str]]) -> None:
        try:
            for index, command in enumerate(commands, 1):
                if len(commands) > 1:
                    self.events.put(("log", f"\n[{index}/{len(commands)}] {' '.join(command)}\n"))
                exit_code = self._run_command(command)
                if exit_code != 0:
                    self.events.put(("done", str(exit_code)))
                    return
            self.events.put(("done", "0"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _run_command(self, command: list[str]) -> int:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        ffmpeg_path = self.external_ffmpeg_path.get().strip()
        if ffmpeg_path:
            env[FFMPEG_PATH_ENV] = ffmpeg_path
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=Path(__file__).resolve().parents[2],
            bufsize=1,
            env=env,
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.events.put(("log", line))
        return self.process.wait()

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log" and payload is not None:
                self._append_log(payload)
            elif kind == "done":
                self.process = None
                self.started_at = None
                self._stop_progress_bar()
                self._set_idle_buttons()
                if payload == "0":
                    self.status.set("대기 중")
                    self.progress_summary.set("완료되었습니다.")
                    self.progress_detail.set("결과 파일을 확인할 수 있습니다.")
                    copied_paths = self._copy_pending_translations()
                    if copied_paths:
                        copied_names = ", ".join(path.name for path in copied_paths[:3])
                        if len(copied_paths) > 3:
                            copied_names += f" 외 {len(copied_paths) - 3}개"
                        self.progress_detail.set(f"번역 자막을 원본 위치로 복사했습니다: {copied_names}")
                        self._append_log(
                            "\nCopied Korean subtitle(s) to input location:\n"
                            + "".join(f"  {path}\n" for path in copied_paths)
                        )
                    self._append_log("\n완료되었습니다.\n")
                    self._refresh_derived_status()
                    self._refresh_recent_work_time_status()
                    self._refresh_result_paths()
                    messagebox.showinfo("완료", "작업이 완료되었습니다.")
                else:
                    self.status.set("대기 중")
                    self.progress_summary.set("실패했습니다.")
                    self.progress_detail.set("자세한 로그를 확인하세요.")
                    self._append_log(f"\n실패했습니다. 종료 코드: {payload}\n")
                    self._set_log_visible(True)
                    self._refresh_derived_status()
                self.pending_translation_copy = None
            elif kind == "error" and payload is not None:
                self.process = None
                self.started_at = None
                self._stop_progress_bar()
                self._set_idle_buttons()
                self.status.set("대기 중")
                self.progress_summary.set("오류가 발생했습니다.")
                self.progress_detail.set("자세한 로그를 확인하세요.")
                self._append_log(f"\n오류: {payload}\n")
                self._set_log_visible(True)
                self.pending_translation_copy = None
        self.root.after(100, self._drain_events)

    def _start_progress_bar(self) -> None:
        self.progress_bar.configure(mode="indeterminate", value=0)
        self.progress_bar.start(12)

    def _stop_progress_bar(self) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=0)

    def _copy_pending_translations(self) -> list[Path]:
        if self.pending_translation_copy is None:
            return []
        input_path, output_dir, srt_paths = self.pending_translation_copy
        return copy_korean_subtitles_to_input_location(input_path, output_dir, srt_paths)

    def _update_elapsed_status(self, label: str = "처리 중") -> None:
        if self.started_at is None:
            return
        elapsed = time.monotonic() - self.started_at
        self.status.set(f"{label} {format_elapsed_korean(elapsed)} 경과")
        self.root.after(1000, lambda: self._update_elapsed_status(label))

    def _append_log(self, text: str) -> None:
        self.log_buffer += text
        if self.log_widget is not None:
            self.log_widget.insert("end", text)
            self.log_widget.see("end")
        summary = summarize_progress_line(text)
        if summary:
            self.progress_summary.set(summary)
            clean_detail = text.strip().replace("\r", " ")
            if clean_detail:
                self.progress_detail.set(clean_detail[:160])

    def toggle_log(self) -> None:
        self._open_log_window()

    def _set_log_visible(self, visible: bool) -> None:
        if visible:
            self._open_log_window()
        else:
            self._close_log_window()

    def _open_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            self.log_window.focus_force()
            return
        window = Toplevel(self.root)
        window.title("자세한 로그")
        window.geometry("900x460")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_log_window)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_widget = ScrolledText(frame, height=18, wrap="word")
        self.log_widget.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.log_widget.insert("end", self.log_buffer)
        self.log_widget.see("end")

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="로그 복사", command=self.copy_log).pack(side="left", padx=4)
        ttk.Button(buttons, text="닫기", command=self._close_log_window).pack(side="left", padx=4)
        self.log_window = window

    def _close_log_window(self) -> None:
        window = self.log_window
        self.log_window = None
        self.log_widget = None
        if window is not None and window.winfo_exists():
            window.destroy()

    def _clear_log(self) -> None:
        self.log_buffer = ""
        if self.log_widget is not None:
            self.log_widget.delete("1.0", "end")

    def copy_log(self) -> None:
        text = self.log_buffer.strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _refresh_ffmpeg_status(self) -> None:
        self.ffmpeg_status.set(ffmpeg_status_text(self.external_ffmpeg_path.get()))
        self._remember_state()

    def _refresh_ollama_status(self) -> None:
        self.ollama_status.set(ollama_server_text(self.ollama_host.get(), self.ollama_port.get()))
        self._remember_state()

    def _refresh_qwen_status(self) -> None:
        self.qwen_status.set(qwen_environment_status_text())
        self._remember_state()

    def _ollama_port_or_warn(self) -> int | None:
        try:
            port = int(self.ollama_port.get().strip())
        except ValueError:
            messagebox.showwarning("Ollama 포트 확인", "Ollama 포트는 숫자로 입력해야 합니다.")
            return None
        if port <= 0 or port > 65535:
            messagebox.showwarning("Ollama 포트 확인", "Ollama 포트는 1부터 65535 사이여야 합니다.")
            return None
        return port

    def open_output_dir(self) -> None:
        output_dir = Path(self.output_dir.get().strip() or Path.cwd() / "tmp-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(output_dir)

    def open_latest_subtitle(self) -> None:
        self._refresh_result_paths()
        subtitle_paths = [path for path in self.last_result_paths if path.exists() and path.suffix.lower() == ".srt"]
        if not subtitle_paths:
            messagebox.showinfo("자막 열기", "열 수 있는 자막 파일이 아직 없습니다.")
            return
        latest = max(subtitle_paths, key=lambda path: path.stat().st_mtime)
        app = default_app_for_extension(latest.suffix.lower())
        if not messagebox.askyesno(
            "자막 열기 확인",
            "아래 자막 파일을 Windows 기본 연결 앱으로 엽니다.\n\n"
            f"파일: {latest}\n"
            f"연결 앱: {app}\n\n"
            "연결 앱이 동영상 플레이어라면 소리가 재생될 수 있습니다.\n"
            "계속 열까요?",
        ):
            return
        os.startfile(latest)

    def _refresh_derived_status(self) -> None:
        self._remember_state()
        input_text = self.input_path.get().strip()
        output_text = self.output_dir.get().strip()
        if not output_text:
            self.translation_status.set("입력/작업 폴더 필요")
            self.estimate_status.set("이력 부족")
            self._update_translate_button_state()
            return
        if not input_text:
            output_dir = Path(output_text)
            self.translation_status.set("입력/작업 폴더 필요")
            self.estimate_status.set(estimate_history_text(output_dir))
            self._update_translate_button_state()
            return
        input_path = Path(input_text)
        output_dir = Path(output_text)
        has_subtitles = bool(pending_translation_subtitles(input_path, output_dir))
        self.translation_status.set(summarize_existing_translation(input_path, output_dir))
        self.estimate_status.set(
            estimate_work_text(
                input_path,
                output_dir,
                self.translate.get() or has_subtitles,
                self.external_ffmpeg_path.get(),
            )
        )
        self._refresh_result_paths()
        self._update_translate_button_state()

    def _refresh_recent_work_time_status(self) -> None:
        input_text = self.input_path.get().strip()
        output_text = self.output_dir.get().strip()
        if not input_text or not output_text:
            return
        input_path = Path(input_text)
        output_dir = Path(output_text)
        has_translation = self.translate.get() or bool(existing_korean_subtitles(find_existing_japanese_subtitles(input_path, output_dir), output_dir))
        recent_text = recent_work_time_text(input_path, output_dir, has_translation)
        if recent_text:
            self.estimate_status.set(recent_text)

    def _refresh_result_paths(self) -> None:
        input_text = self.input_path.get().strip()
        output_text = self.output_dir.get().strip()
        if not input_text or not output_text:
            self.last_result_paths = []
            return
        self.last_result_paths = expected_output_paths(Path(input_text), Path(output_text))

    def _remember_state(self) -> None:
        if not hasattr(self, "input_path"):
            return
        save_launcher_state(
            launcher_state_from_values(
                self.output_dir.get().strip(),
                self.model_device.get().strip() or "cuda:0",
                self.external_ffmpeg_path.get().strip(),
                self.ollama_host.get().strip() or "localhost",
                self._state_ollama_port(),
                asr_backend_from_label(self.asr_engine.get()),
                self.app_root,
            )
        )

    def _state_ollama_port(self) -> int:
        try:
            port = int(self.ollama_port.get().strip())
        except ValueError:
            return 11434
        return port if 0 < port <= 65535 else 11434

    def _set_running_buttons(self) -> None:
        self.run_button.configure(state="disabled")
        self.translate_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def _set_idle_buttons(self) -> None:
        self.run_button.configure(state="normal")
        self._update_translate_button_state()
        self.stop_button.configure(state="disabled")

    def _update_translate_button_state(self) -> None:
        if not hasattr(self, "translate_button"):
            return
        if self.process is not None or self.started_at is not None:
            self.translate_button.configure(state="disabled")
            return
        input_text = self.input_path.get().strip()
        output_text = self.output_dir.get().strip()
        has_pending_subtitles = bool(
            input_text and output_text and pending_translation_subtitles(Path(input_text), Path(output_text))
        )
        if self.translate.get() or not has_pending_subtitles:
            self.translate_button.configure(state="disabled", text="번역")
        else:
            self.translate_button.configure(state="normal", text="번역 가능")


class ModelDialog:
    def __init__(self, root: Tk, models: list[str], target: StringVar) -> None:
        self.target = target
        self.models = models
        self.window = Toplevel(root)
        dialog = self.window
        dialog.title("Ollama 모델 선택")
        dialog.transient(root)
        dialog.grab_set()
        dialog.geometry("720x360")

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame,
            text="번역 추천 모델을 먼저 표시합니다.\n[실험용/비추천] 또는 [번역 미확인] 모델은 선택할 수 있지만 품질이나 형식 준수는 보장되지 않습니다.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.listbox = Listbox(frame)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        for model in models:
            self.listbox.insert("end", format_ollama_model_choice(model))
        if models:
            self.listbox.selection_set(0)
        self.listbox.bind("<Double-Button-1>", lambda _event: self.select())

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="선택", command=self.select).pack(side="left", padx=4)
        ttk.Button(buttons, text="닫기", command=dialog.destroy).pack(side="left", padx=4)

    def select(self) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.target.set(self.models[selection[0]])
        self.window.winfo_toplevel().destroy()


class OllamaServerDialog:
    def __init__(self, root: Tk, host: StringVar, port: StringVar) -> None:
        self.host = host
        self.port = port
        self.window = Toplevel(root)
        dialog = self.window
        dialog.title("Ollama 주소/포트 변경")
        dialog.transient(root)
        dialog.grab_set()
        dialog.geometry("420x150")

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        self.host_value = StringVar(value=host.get().strip() or "localhost")
        self.port_value = StringVar(value=port.get().strip() or "11434")

        ttk.Label(frame, text="주소").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Entry(frame, textvariable=self.host_value).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="포트").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Entry(frame, textvariable=self.port_value, width=12).grid(row=1, column=1, sticky="w", pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="저장", command=self.save).pack(side="left", padx=4)
        ttk.Button(buttons, text="취소", command=dialog.destroy).pack(side="left", padx=4)

    def save(self) -> None:
        host = self.host_value.get().strip() or "localhost"
        try:
            port = int(self.port_value.get().strip())
        except ValueError:
            messagebox.showwarning("Ollama 포트 확인", "Ollama 포트는 숫자로 입력해야 합니다.")
            return
        if port <= 0 or port > 65535:
            messagebox.showwarning("Ollama 포트 확인", "Ollama 포트는 1부터 65535 사이여야 합니다.")
            return
        self.host.set(host)
        self.port.set(str(port))
        self.window.winfo_toplevel().destroy()


def main() -> int:
    root = Tk()
    configure_theme(root)
    KotobaLauncher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
