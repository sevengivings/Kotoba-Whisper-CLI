from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kotoba_standalone.progress import ProgressCallback
from kotoba_standalone.types import ProgressEvent, TranslationOptions, TranslationResult


OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 8192,
}

RECOMMENDED_TRANSLATION_MODEL_KEYWORDS = (
    "hy-mt2-30b",
    "hy-mt2-7b",
    "translategemma",
)
DISCOURAGED_TRANSLATION_MODEL_KEYWORDS = (
    "hy-mt2-1.8b",
)

SUBTITLE_STYLE_PROMPT = (
    "Many subtitles are short everyday Japanese, casual reactions, interjections, or adult video dialogue. "
    "Translate them into natural Korean spoken subtitle style. "
    "Keep short exclamations, moans, reactions, and backchannels short in Korean too. "
    "Do not over-explain unclear short utterances or add alternatives such as '(or ...)'. "
    "Do not sanitize adult context unnecessarily. "
    "Preserve concrete nouns, body parts, actions, and who is speaking when they are present in the source. "
    "If ASR text is unclear or noisy, choose one concise plausible subtitle translation without explaining uncertainty. "
    "Do not make the line more explicit than the source. "
)

KOREAN_POLITE_STYLE_PROMPT = (
    "When the target language is Korean, always use polite Korean speech style. "
    "Do not use banmal or casual plain endings. "
    "Keep very short reactions natural, but do not turn them into rude or informal Korean. "
)

KOREAN_BANMAL_STYLE_PROMPT = (
    "When the target language is Korean, use natural informal Korean speech style, also known as informal speech. "
    "Avoid polite Korean endings unless the source explicitly requires formality. "
    "Keep very short reactions natural and casual. "
)

KOREAN_STRICT_BANMAL_STYLE_PROMPT = (
    "When the target language is Korean, use strict informal Korean speech style. "
    "The Korean output must be non-polite and casual, even if the Japanese source uses polite expressions. "
    "Convert polite Korean expressions into informal Korean. "
    "Do not use Korean polite sentence endings or formal thanks. "
)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaModelError(RuntimeError):
    pass


def translate_srt(
    input_srt: Path,
    options: TranslationOptions,
    progress: ProgressCallback | None = None,
) -> TranslationResult:
    started = time.time()
    input_srt = input_srt.expanduser().resolve()
    if not input_srt.exists():
        raise FileNotFoundError(input_srt)

    assert_ollama_model_available(options)
    output_srt = resolve_output_srt(input_srt, options.output)
    entries = parse_srt(input_srt)
    translated_texts = translate_entries(
        entries,
        options=options,
        progress=progress,
        started=started,
    )

    part_path = output_srt.with_suffix(output_srt.suffix + ".part")
    write_srt(part_path, entries, translated_texts)
    validate_srt(part_path, len(entries))
    part_path.replace(output_srt)

    processing_seconds = round(time.time() - started, 3)
    metadata = {
        "status": "success",
        "input_srt": str(input_srt),
        "output_srt": str(output_srt),
        "provider": "ollama",
        "model": options.model,
        "source": options.source,
        "target": options.target,
        "subtitle_count": len(entries),
        "batch_translate": options.batch_translate,
        "batch_size": options.batch_size if options.batch_translate else None,
        "korean_style": options.korean_style,
        "processing_seconds": processing_seconds,
    }
    metadata_path = output_srt.with_suffix(".translation.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TranslationResult(input_srt, output_srt, metadata_path, len(entries), processing_seconds)


def check_ollama_available(options: TranslationOptions) -> None:
    get_ollama_models(options)


def get_ollama_models(options: TranslationOptions) -> list[str]:
    url = f"http://{options.ollama_host}:{options.ollama_port}/api/tags"
    request = urllib.request.Request(url, method="GET")
    timeout = min(5, max(1, options.timeout_seconds))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {url}. Start Ollama first, then retry."
        ) from exc
    models = [str(item.get("name", "")).strip() for item in data.get("models", []) if isinstance(item, dict)]
    models = [model for model in models if model]
    if not models:
        raise OllamaModelError(f"No Ollama models found at {url}. Run 'ollama pull <model>' first, then retry.")
    return models


def is_likely_translation_model(model: str) -> bool:
    return translation_model_label(model) == "번역 추천"


def translation_model_label(model: str) -> str:
    normalized = model.strip().lower()
    if any(keyword in normalized for keyword in DISCOURAGED_TRANSLATION_MODEL_KEYWORDS):
        return "실험용/비추천"
    if any(keyword in normalized for keyword in RECOMMENDED_TRANSLATION_MODEL_KEYWORDS):
        return "번역 추천"
    return "번역 미확인"


def sort_ollama_models_for_translation(models: list[str]) -> list[str]:
    rank = {"번역 추천": 0, "실험용/비추천": 1, "번역 미확인": 2}
    return sorted(models, key=lambda model: (rank[translation_model_label(model)], model.lower()))


def format_ollama_model_choice(model: str) -> str:
    return f"[{translation_model_label(model)}] {model}"


def assert_ollama_model_available(options: TranslationOptions) -> None:
    models = get_ollama_models(options)
    if options.model not in models:
        available = ", ".join(models)
        raise OllamaModelError(
            f"Ollama model '{options.model}' was not found at http://{options.ollama_host}:{options.ollama_port}. "
            f"Available models: {available}."
        )


def default_output_srt(input_srt: Path) -> Path:
    if input_srt.name.endswith(".ja.srt"):
        return input_srt.with_name(input_srt.name.removesuffix(".ja.srt") + ".ko.srt")
    return input_srt.with_name(f"{input_srt.stem}.ko.srt")


def resolve_output_srt(input_srt: Path, output: Path | None) -> Path:
    if output is None:
        return default_output_srt(input_srt)
    output = output.expanduser()
    if output.exists() and output.is_dir():
        return output / default_output_srt(input_srt).name
    if output.suffix:
        return output
    return output / default_output_srt(input_srt).name


def parse_srt(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    entries: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        entries.append({"timecode": lines[1].strip(), "text": " ".join(line.strip() for line in lines[2:]).strip()})
    if not entries:
        raise RuntimeError(f"No subtitle entries found: {path}")
    return entries


def translate_entries(
    entries: list[dict[str, str]],
    options: TranslationOptions,
    progress: ProgressCallback | None,
    started: float,
) -> list[str]:
    texts = [entry["text"] for entry in entries]
    translated = [""] * len(texts)
    if options.batch_translate:
        if options.batch_size <= 0:
            raise RuntimeError("--batch-size must be greater than 0")
        if options.text_split_size < 0:
            raise RuntimeError("--text-split-size must be greater than or equal to 0")
        batches = make_batches(texts, options.batch_size, options.text_split_size)
        for batch_no, batch in enumerate(batches, 1):
            first_subtitle = batch[0] + 1
            last_subtitle = batch[-1] + 1
            _emit(progress, started, "translate", f"Translating subtitles {first_subtitle}-{last_subtitle}", batch_no - 1, len(batches))
            numbered = [f"[{local_no}] {texts[index]}" for local_no, index in enumerate(batch, 1)]
            result = translate_text_ollama(options, "\n".join(numbered))
            parsed = parse_batch_translation(result, batch)
            for index, translated_text in parsed.items():
                translated[index] = translated_text
            _emit(progress, started, "translate", f"Translated subtitles {first_subtitle}-{last_subtitle}", batch_no, len(batches))

        missing = [index for index, value in enumerate(translated) if not value.strip()]
        if missing:
            for retry_no, index in enumerate(missing, 1):
                _emit(progress, started, "retry", f"Retrying subtitle {index + 1}", retry_no, len(missing))
                translated[index] = translate_text_ollama(options, texts[index])
    else:
        for index, text in enumerate(texts, 1):
            _emit(progress, started, "translate", f"Translating subtitle {index}", index, len(texts))
            translated[index - 1] = translate_text_ollama(options, text)

    validate_translations(translated)
    return translated


def make_batches(texts: list[str], batch_size: int, text_split_size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_length = 0
    for index, text in enumerate(texts):
        next_length = current_length + len(text) + 1
        exceeds_text_limit = text_split_size > 0 and next_length >= text_split_size
        if current and (len(current) >= batch_size or exceeds_text_limit):
            batches.append(current)
            current = []
            current_length = 0
        current.append(index)
        current_length += len(text) + 1
    if current:
        batches.append(current)
    return batches


def translate_text_ollama(options: TranslationOptions, text: str) -> str:
    url = f"http://{options.ollama_host}:{options.ollama_port}/api/chat"
    batch_mode = "\n" in text or re.match(r"^\[\d+\]\s+", text.strip())
    payload = {
        "model": options.model,
        "messages": [
            {"role": "system", "content": build_system_prompt(options.source, options.target, batch_mode, options.korean_style)},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=options.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama translation failed: {exc}") from exc
    return str(result.get("message", {}).get("content", "")).strip()


def build_system_prompt(source_lang: str, target_lang: str, batch_mode: bool, korean_style: str = "polite") -> str:
    korean_style_prompt = _korean_style_prompt(target_lang, korean_style)
    if batch_mode:
        return (
            "You are a professional video subtitle translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"{SUBTITLE_STYLE_PROMPT}"
            f"{korean_style_prompt}"
            "The input contains lines numbered [N]. "
            "Translate each line separately and prefix the output with the same [N]. "
            "Do not merge lines. Do not renumber lines. Output only the translated text."
        )
    return (
        "You are a professional video subtitle translator. "
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"{SUBTITLE_STYLE_PROMPT}"
        f"{korean_style_prompt}"
        "Ensure the translation is natural and conversational. "
        "Do not include any introductory, concluding remarks, or notes. Output only the translated text."
    )


def parse_batch_translation(result: str, batch: list[int]) -> dict[int, str]:
    local_to_original = {local_no: original_index for local_no, original_index in enumerate(batch, 1)}
    parsed: dict[int, str] = {}
    current_original_index = -1
    for line in _iter_batch_translation_lines(result):
        line = line.strip()
        if not line:
            continue
        strict_match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        fallback_match = re.match(r"^(\d+)[\.\)]\s+(.*)$", line) if not strict_match else None
        match = strict_match or fallback_match
        if match:
            local_no = int(match.group(1))
            if local_no not in local_to_original:
                current_original_index = -1
                continue
            current_original_index = local_to_original[local_no]
            parsed[current_original_index] = clean_translated_subtitle_text(match.group(2))
        elif current_original_index != -1:
            parsed[current_original_index] = clean_translated_subtitle_text(
                f"{parsed[current_original_index]} {line}".strip()
            )
    return parsed


def _iter_batch_translation_lines(result: str) -> list[str]:
    lines: list[str] = []
    for raw_line in result.strip().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        if re.match(r"^\d+[\.\)]\s+\[\d+\]\s+", raw_line):
            lines.append(raw_line)
        else:
            lines.extend(re.split(r"\s+(?=\[\d+\]\s+)", raw_line))
    return lines


def validate_translations(translated_texts: list[str]) -> None:
    for index, text in enumerate(translated_texts, 1):
        if not text.strip():
            raise RuntimeError(f"Empty translation remains at subtitle {index}")


def write_srt(path: Path, entries: list[dict[str, str]], translated_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, (entry, translated_text) in enumerate(zip(entries, translated_texts), 1):
        blocks.append(f"{index}\n{entry['timecode']}\n{clean_translated_subtitle_text(translated_text)}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def clean_translated_subtitle_text(text: str) -> str:
    cleaned = text.strip()
    while True:
        next_cleaned = re.sub(r"^\[\d{1,4}\]\s*", "", cleaned).strip()
        if next_cleaned == cleaned:
            return cleaned
        cleaned = next_cleaned


def validate_srt(path: Path, expected_count: int) -> None:
    blocks = [block for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()) if block.strip()]
    if len(blocks) != expected_count:
        raise RuntimeError(f"Output SRT count mismatch: expected {expected_count}, got {len(blocks)}")


def _korean_style_prompt(target_lang: str, korean_style: str) -> str:
    if target_lang.strip().lower() not in {"ko", "kor", "korean", "kr"}:
        return ""
    if korean_style == "polite":
        return KOREAN_POLITE_STYLE_PROMPT
    if korean_style == "banmal":
        return KOREAN_BANMAL_STYLE_PROMPT
    if korean_style == "strict-banmal":
        return KOREAN_STRICT_BANMAL_STYLE_PROMPT
    raise RuntimeError("--korean-style must be polite, banmal, or strict-banmal")


def _emit(
    callback: ProgressCallback | None,
    started: float,
    stage: str,
    message: str,
    current: int,
    total: int,
) -> None:
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=(current / total) * 100 if total else None,
            elapsed_seconds=time.time() - started,
        )
    )
