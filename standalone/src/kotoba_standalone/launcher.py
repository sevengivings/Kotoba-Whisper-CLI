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

from kotoba_standalone.settings import DEFAULT_TRANSLATION_MODEL, load_saved_translation_model
from kotoba_standalone.translate.ollama import OllamaUnavailableError, TranslationOptions, get_ollama_models


@dataclass(frozen=True)
class LauncherOptions:
    input_path: Path
    output_dir: Path
    translate: bool = False
    translation_model: str = ""
    korean_style: str = "polite"
    model_device: str = "cuda:0"


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

        ttk.Checkbutton(outer, text="한국어 번역까지 실행", variable=self.translate).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Label(outer, text="번역 모델").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.model).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="Ollama 모델", command=self.load_ollama_models).grid(
            row=3, column=2, columnspan=2, sticky="ew", padx=3
        )

        ttk.Label(outer, text="한국어 말투").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(
            outer,
            textvariable=self.korean_style,
            values=("polite", "banmal", "strict-banmal"),
            state="readonly",
            width=18,
        ).grid(row=4, column=1, sticky="w", padx=8)

        ttk.Label(outer, text="처리 장치").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            outer,
            textvariable=self.model_device,
            values=("cuda:0", "cpu"),
            width=18,
        ).grid(row=5, column=1, sticky="w", padx=8)

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        buttons.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(buttons, text="시작", command=self.start)
        self.run_button.grid(row=0, column=1, padx=4)
        self.stop_button = ttk.Button(buttons, text="중지", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=4)
        ttk.Label(buttons, textvariable=self.status).grid(row=0, column=0, sticky="w")

        self.log = ScrolledText(outer, height=18, wrap="word")
        self.log.grid(row=7, column=0, columnspan=4, sticky="nsew")

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(title="처리할 영상 또는 오디오 파일을 선택하세요")
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
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.started_at = time.monotonic()
        self.status.set("처리 중 0초 경과")
        self._update_elapsed_status()
        threading.Thread(target=self._run_command, args=(command,), daemon=True).start()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.started_at = None
            self.process.terminate()
            self.status.set("중지 요청됨")

    def _run_command(self, command: list[str]) -> None:
        try:
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
            exit_code = self.process.wait()
            self.events.put(("done", str(exit_code)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

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
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if payload == "0":
                    self.status.set("완료")
                    self._append_log("\n완료되었습니다.\n")
                else:
                    self.status.set("실패")
                    self._append_log(f"\n실패했습니다. 종료 코드: {payload}\n")
            elif kind == "error" and payload is not None:
                self.process = None
                self.started_at = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status.set("오류")
                self._append_log(f"\n오류: {payload}\n")
        self.root.after(100, self._drain_events)

    def _update_elapsed_status(self) -> None:
        if self.started_at is None:
            return
        elapsed = time.monotonic() - self.started_at
        self.status.set(f"처리 중 {format_elapsed_korean(elapsed)} 경과")
        self.root.after(1000, self._update_elapsed_status)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")


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
