from __future__ import annotations

import queue
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

from kotoba_standalone.media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, is_supported_media
from kotoba_standalone.settings import DEFAULT_TRANSLATION_MODEL, load_saved_translation_model
from kotoba_standalone.translate.ollama import (
    OllamaUnavailableError,
    TranslationOptions,
    default_output_srt,
    get_ollama_models,
)


@dataclass(frozen=True)
class LauncherOptions:
    input_path: Path
    output_dir: Path
    translate: bool = False
    translation_model: str = ""
    korean_style: str = "polite"
    model_device: str = "cuda:0"


@dataclass(frozen=True)
class LauncherTranslationOptions:
    input_path: Path
    output_dir: Path
    translation_model: str = ""
    korean_style: str = "polite"


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
    if options.translate:
        command.append("--translate")
        if options.translation_model.strip():
            command.extend(["--translation-model", options.translation_model.strip()])
        command.extend(["--korean-style", options.korean_style])
    return command


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
    command.extend(["--korean-style", options.korean_style])
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


class KotobaLauncher:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Kotoba Standalone")
        self.root.geometry("820x620")
        self.events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: float | None = None

        self.input_path = StringVar()
        self.output_dir = StringVar(value=str(Path.cwd() / "tmp-output"))
        self.translate = BooleanVar(value=False)
        self.model = StringVar(value=load_saved_translation_model() or DEFAULT_TRANSLATION_MODEL)
        self.korean_style = StringVar(value="polite")
        self.model_device = StringVar(value="cuda:0")
        self.status = StringVar(value="대기 중")

        self._build_ui()
        self.translate.trace_add("write", lambda *_args: self._update_translate_button_state())
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(7, weight=1)

        ttk.Label(outer, text="입력 영상 또는 폴더").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="파일 선택", command=self.choose_file).grid(row=0, column=2, padx=3)
        ttk.Button(outer, text="폴더 선택", command=self.choose_folder).grid(row=0, column=3, padx=3)

        ttk.Label(outer, text="결과 폴더").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="선택", command=self.choose_output_dir).grid(row=1, column=2, columnspan=2, sticky="ew", padx=3)

        ttk.Label(outer, text="번역 모델").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.model).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="Ollama 모델", command=self.load_ollama_models).grid(
            row=2, column=2, columnspan=2, sticky="ew", padx=3
        )

        ttk.Label(outer, text="한국어 말투").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(
            outer,
            textvariable=self.korean_style,
            values=("polite", "banmal", "strict-banmal"),
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(outer, text="처리 장치").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(
            outer,
            textvariable=self.model_device,
            values=("cuda:0", "cpu"),
            width=18,
        ).grid(row=4, column=1, sticky="w", padx=8)

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        buttons.columnconfigure(0, weight=1)
        ttk.Checkbutton(buttons, text="한국어 번역까지 실행", variable=self.translate).grid(row=0, column=1, padx=(0, 12))
        self.run_button = ttk.Button(buttons, text="시작", command=self.start)
        self.run_button.grid(row=0, column=2, padx=4)
        self.translate_button = ttk.Button(buttons, text="번역", command=self.translate_existing)
        self.translate_button.grid(row=0, column=3, padx=4)
        self.stop_button = ttk.Button(buttons, text="중지", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=4, padx=4)
        ttk.Label(buttons, textvariable=self.status).grid(row=0, column=0, sticky="w")

        self.log = ScrolledText(outer, height=18, wrap="word")
        self.log.grid(row=7, column=0, columnspan=4, sticky="nsew")

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="처리할 동영상 또는 음성 파일을 선택하세요",
            filetypes=media_filetypes(),
        )
        if selected:
            self.input_path.set(selected)

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="처리할 영상 폴더를 선택하세요")
        if selected:
            self.input_path.set(selected)

    def choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="결과를 저장할 폴더를 선택하세요")
        if selected:
            self.output_dir.set(selected)

    def load_ollama_models(self) -> None:
        try:
            models = get_ollama_models(TranslationOptions(model=self.model.get() or DEFAULT_TRANSLATION_MODEL))
        except OllamaUnavailableError as exc:
            messagebox.showwarning("Ollama 연결 실패", str(exc))
            return
        if not models:
            messagebox.showinfo("Ollama 모델", "다운로드된 Ollama 모델이 없습니다.")
            return
        ModelDialog(self.root, models, self.model)

    def start(self) -> None:
        input_text = self.input_path.get().strip()
        if not input_text:
            messagebox.showwarning("입력 필요", "처리할 영상 파일 또는 폴더를 선택하세요.")
            return
        validation_error = validate_input_path(Path(input_text))
        if validation_error is not None:
            messagebox.showwarning("입력 확인", validation_error)
            return
        options = LauncherOptions(
            input_path=Path(input_text),
            output_dir=Path(self.output_dir.get().strip() or Path.cwd() / "tmp-output"),
            translate=self.translate.get(),
            translation_model=self.model.get(),
            korean_style=self.korean_style.get(),
            model_device=self.model_device.get().strip() or "cuda:0",
        )
        command = build_process_command(options)
        self.log.delete("1.0", "end")
        self._append_log("> " + " ".join(command) + "\n\n")
        self._set_running_buttons()
        self.started_at = time.monotonic()
        self.status.set("처리 중 0초 경과")
        self._update_elapsed_status()
        threading.Thread(target=self._run_commands, args=([command],), daemon=True).start()

    def translate_existing(self) -> None:
        input_text = self.input_path.get().strip()
        if not input_text:
            messagebox.showwarning("입력 필요", "처리한 영상 파일 또는 폴더를 선택하세요.")
            return
        input_path = Path(input_text)
        output_dir = Path(self.output_dir.get().strip() or Path.cwd() / "tmp-output")
        validation_error = validate_input_path(input_path)
        if validation_error is not None:
            messagebox.showwarning("입력 확인", validation_error)
            return
        srt_paths = find_existing_japanese_subtitles(input_path, output_dir)
        if not srt_paths:
            messagebox.showwarning("번역할 자막 없음", "결과 폴더에서 일본어 자막(*.ja.srt)을 찾지 못했습니다.")
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
        )
        commands = [build_translate_command(srt_path, options) for srt_path in srt_paths]
        self.log.delete("1.0", "end")
        for command in commands:
            self._append_log("> " + " ".join(command) + "\n")
        self._append_log("\n")
        self._set_running_buttons()
        self.started_at = time.monotonic()
        self.status.set("번역 중 0초 경과")
        self._update_elapsed_status("번역 중")
        threading.Thread(target=self._run_commands, args=(commands,), daemon=True).start()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.started_at = None
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
                self._set_idle_buttons()
                if payload == "0":
                    self.status.set("완료")
                    self._append_log("\n완료되었습니다.\n")
                else:
                    self.status.set("실패")
                    self._append_log(f"\n실패했습니다. 종료 코드: {payload}\n")
            elif kind == "error" and payload is not None:
                self.process = None
                self.started_at = None
                self._set_idle_buttons()
                self.status.set("오류")
                self._append_log(f"\n오류: {payload}\n")
        self.root.after(100, self._drain_events)

    def _update_elapsed_status(self, label: str = "처리 중") -> None:
        if self.started_at is None:
            return
        elapsed = time.monotonic() - self.started_at
        self.status.set(f"{label} {format_elapsed_korean(elapsed)} 경과")
        self.root.after(1000, lambda: self._update_elapsed_status(label))

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

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
        self.translate_button.configure(state="disabled" if self.translate.get() else "normal")


class ModelDialog:
    def __init__(self, root: Tk, models: list[str], target: StringVar) -> None:
        self.target = target
        self.window = Toplevel(root)
        dialog = self.window
        dialog.title("Ollama 모델 선택")
        dialog.transient(root)
        dialog.grab_set()
        dialog.geometry("640x320")

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.listbox = Listbox(frame)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        for model in models:
            self.listbox.insert("end", model)
        if models:
            self.listbox.selection_set(0)

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="선택", command=self.select).pack(side="left", padx=4)
        ttk.Button(buttons, text="닫기", command=dialog.destroy).pack(side="left", padx=4)

    def select(self) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.target.set(self.listbox.get(selection[0]))
        self.window.winfo_toplevel().destroy()


def main() -> int:
    root = Tk()
    KotobaLauncher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
