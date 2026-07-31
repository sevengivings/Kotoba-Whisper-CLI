from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


KoreanStyle = Literal["polite", "banmal", "strict-banmal"]


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ProcessOptions:
    output_dir: Path | None = None
    language: str = "japanese"
    silence_threshold_db: str = "-42dB"
    min_silence_duration_s: float = 0.5
    auto_silence_threshold: bool = False
    vad_pre_split: bool = True
    vad_max_segment_duration_s: float = 30.0
    vad_min_speech_duration_s: float = 0.25
    vad_padding_s: float = 0.4
    vad_merge_gap_s: float = 0.0
    batch_size: int = 8
    chunk_length_s: int = 15
    model_name: str = "kotoba-tech/kotoba-whisper-v2.2"
    model_device: str = "cuda:0"
    model_dtype: str = "float16"
    translate: bool = False
    translation_model: str = "hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M"
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    korean_style: KoreanStyle = "polite"


@dataclass(frozen=True)
class ProcessResult:
    input_path: Path
    output_dir: Path
    wav_path: Path | None
    ja_srt_path: Path | None
    ko_srt_path: Path | None
    copied_ko_srt_path: Path | None
    status: str
    message: str


@dataclass(frozen=True)
class TranslationOptions:
    model: str
    output: Path | None = None
    source: str = "japanese"
    target: str = "korean"
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    batch_translate: bool = True
    batch_size: int = 50
    text_split_size: int = 0
    timeout_seconds: int = 600
    korean_style: KoreanStyle = "polite"


@dataclass(frozen=True)
class TranslationResult:
    input_srt: Path
    output_srt: Path
    metadata_path: Path
    subtitle_count: int
    processing_seconds: float
