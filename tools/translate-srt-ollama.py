from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 8192,
}


class ProgressPrinter:
    def __init__(self) -> None:
        self._last_width = 0

    def update(self, message: str) -> None:
        width = max(self._last_width, len(message))
        sys.stdout.write("\r" + message.ljust(width))
        sys.stdout.flush()
        self._last_width = len(message)

    def finish(self) -> None:
        if self._last_width:
            sys.stdout.write("\r" + (" " * self._last_width) + "\r")
            sys.stdout.flush()
            self._last_width = 0

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
    "Prefer endings such as -요, -예요, -이에요, -세요, -습니다, and -입니다 when the sentence needs an ending. "
    "Keep very short reactions natural, but do not turn them into rude or informal Korean. "
)

KOREAN_BANMAL_STYLE_PROMPT = (
    "When the target language is Korean, use natural informal Korean speech style, also known as banmal. "
    "Avoid polite Korean endings such as -요, -예요, -이에요, -세요, -습니다, or -입니다 unless the source explicitly requires formality. "
    "Prefer concise informal spoken endings such as -해, -야, -네, -지, -잖아, and -거든 when natural. "
    "Keep very short reactions natural and casual. "
)

KOREAN_STRICT_BANMAL_STYLE_PROMPT = (
    "When the target language is Korean, use strict informal Korean speech style. "
    "The Korean output must be non-polite and casual, even if the Japanese source uses polite expressions such as です, ます, ください, or お願いします. "
    "Convert polite Korean expressions into informal Korean. "
    "Do not use Korean polite sentence endings such as -요, -예요, -이에요, -세요, -습니다, -입니다, or 감사합니다. "
    "Use informal forms such as 고마워, 기다려, 해, 야, 네, 지, 잖아, and 거든 when natural. "
    "For example, translate ありがとうございます as 고마워, not 감사합니다; translate 기다려 주세요 as 기다려. "
)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_srt: Path = args.input_srt
    if not input_srt.exists():
        raise SystemExit(f"Input SRT does not exist: {input_srt}")

    output_srt = args.output or input_srt.with_name(input_srt.name.replace(".ja.srt", ".ko.srt"))
    if output_srt == input_srt:
        output_srt = input_srt.with_name(f"{input_srt.stem}.ko.srt")

    entries = parse_srt(input_srt)
    print(f"[Info] Translating {len(entries)} subtitle(s) with Ollama model: {args.model}")
    started = time.time()

    translated_texts = translate_entries(
        entries,
        host=args.ollama_host,
        port=str(args.ollama_port),
        model=args.model,
        source_lang=args.source,
        target_lang=args.target,
        batch_translate=args.batch_translate,
        batch_size=args.batch_size,
        text_split_size=args.text_split_size,
        timeout_seconds=args.timeout_seconds,
        korean_style=args.korean_style,
    )

    part_path = output_srt.with_suffix(output_srt.suffix + ".part")
    write_srt(part_path, entries, translated_texts)
    validate_srt(part_path, len(entries))
    part_path.replace(output_srt)

    metadata = {
        "status": "success",
        "input_srt": str(input_srt),
        "output_srt": str(output_srt),
        "provider": "ollama",
        "model": args.model,
        "source": args.source,
        "target": args.target,
        "subtitle_count": len(entries),
        "batch_translate": args.batch_translate,
        "batch_size": args.batch_size if args.batch_translate else None,
        "korean_style": args.korean_style,
        "processing_seconds": round(time.time() - started, 3),
    }
    metadata_path = output_srt.with_suffix(".translation.json")
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    metadata_path.write_text(metadata_text, encoding="utf-8")
    print(f"[Info] Translation saved: {output_srt}")
    print("[Info] Translation metadata:")
    print(metadata_text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate SRT subtitles with Ollama.")
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", default="japanese")
    parser.add_argument("--target", default="korean")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", default="11434")
    parser.add_argument("--model", required=True)
    parser.set_defaults(batch_translate=True)
    parser.add_argument("--batch-translate", dest="batch_translate", action="store_true")
    parser.add_argument("--no-batch-translate", dest="batch_translate", action="store_false")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--text-split-size", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--korean-style", choices=("polite", "banmal", "strict-banmal"), default="polite")
    return parser


def parse_srt(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    entries: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        if "-->" not in lines[1]:
            continue
        entries.append({"timecode": lines[1].strip(), "text": " ".join(line.strip() for line in lines[2:]).strip()})
    if not entries:
        raise RuntimeError(f"No subtitle entries found: {path}")
    return entries


def translate_entries(
    entries: list[dict[str, str]],
    host: str,
    port: str,
    model: str,
    source_lang: str,
    target_lang: str,
    batch_translate: bool,
    batch_size: int,
    text_split_size: int,
    timeout_seconds: int,
    korean_style: str,
) -> list[str]:
    texts = [entry["text"] for entry in entries]
    translated = [""] * len(texts)
    progress = ProgressPrinter()
    started = time.time()
    if batch_translate:
        if batch_size <= 0:
            raise RuntimeError("--batch-size must be greater than 0")
        if text_split_size < 0:
            raise RuntimeError("--text-split-size must be greater than or equal to 0")
        batches = make_batches(texts, batch_size, text_split_size)
        for batch_no, batch in enumerate(batches, 1):
            first_subtitle = batch[0] + 1
            last_subtitle = batch[-1] + 1
            progress.update(
                f"[Info] Translating batch {batch_no}/{len(batches)}: "
                f"subtitles {first_subtitle}-{last_subtitle}"
            )
            numbered = [f"[{local_no}] {texts[index]}" for local_no, index in enumerate(batch, 1)]
            result = translate_text_ollama(
                host,
                port,
                model,
                source_lang,
                target_lang,
                "\n".join(numbered),
                timeout_seconds,
                korean_style,
            )
            parsed = parse_batch_translation(result, batch)
            for index, translated_text in parsed.items():
                translated[index] = translated_text
            progress.update(
                progress_message(
                    prefix="[Info] Batch translated",
                    current=batch_no,
                    total=len(batches),
                    item=f"subtitles {first_subtitle}-{last_subtitle}",
                    started=started,
                )
            )

        missing = [index for index, value in enumerate(translated) if not value.strip()]
        if missing:
            progress.finish()
            print(f"[Warning] Retrying {len(missing)} missing subtitle(s) one by one")
            retry_started = time.time()
            for retry_no, index in enumerate(missing, 1):
                progress.update(
                    progress_message(
                        prefix="[Info] Retrying missing subtitle",
                        current=retry_no,
                        total=len(missing),
                        item=f"subtitle {index + 1}",
                        started=retry_started,
                    )
                )
                translated[index] = translate_text_ollama(
                    host, port, model, source_lang, target_lang, texts[index], timeout_seconds, korean_style
                )
    else:
        for index, text in enumerate(texts, 1):
            progress.update(
                progress_message(
                    prefix="[Info] Translating line",
                    current=index,
                    total=len(texts),
                    item=f"subtitle {index}",
                    started=started,
                )
            )
            translated[index - 1] = translate_text_ollama(
                host, port, model, source_lang, target_lang, text, timeout_seconds, korean_style
            )

    progress.finish()
    validate_translations(translated)
    return translated


def progress_message(prefix: str, current: int, total: int, item: str, started: float) -> str:
    elapsed = time.time() - started
    percent = (current / total) * 100 if total else 0.0
    eta = (elapsed / current) * (total - current) if current else 0.0
    return (
        f"{prefix} {current}/{total} ({percent:.1f}%) | "
        f"{item} | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
    )


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


def translate_text_ollama(
    host: str,
    port: str,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    timeout_seconds: int,
    korean_style: str = "polite",
) -> str:
    url = f"http://{host}:{port}/api/chat"
    batch_mode = "\n" in text or re.match(r"^\[\d+\]\s+", text.strip())
    system_prompt = build_system_prompt(source_lang, target_lang, batch_mode, korean_style)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
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
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama translation failed: {exc}") from exc
    return str(result.get("message", {}).get("content", "")).strip()


def build_system_prompt(
    source_lang: str,
    target_lang: str,
    batch_mode: bool,
    korean_style: str = "polite",
) -> str:
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


def _is_korean_target(target_lang: str) -> bool:
    normalized = target_lang.strip().lower()
    return normalized in {"ko", "kor", "korean", "kr", "한국어", "조선말"}


def _korean_style_prompt(target_lang: str, korean_style: str) -> str:
    if not _is_korean_target(target_lang):
        return ""
    if korean_style == "polite":
        return KOREAN_POLITE_STYLE_PROMPT
    if korean_style == "banmal":
        return KOREAN_BANMAL_STYLE_PROMPT
    if korean_style == "strict-banmal":
        return KOREAN_STRICT_BANMAL_STYLE_PROMPT
    raise RuntimeError("--korean-style must be polite, banmal, or strict-banmal")


def parse_batch_translation(result: str, batch: list[int]) -> dict[int, str]:
    local_to_original = {local_no: original_index for local_no, original_index in enumerate(batch, 1)}
    parsed: dict[int, str] = {}
    current_original_index = -1
    for line in result.splitlines():
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
            parsed[current_original_index] = match.group(2).strip()
        elif current_original_index != -1:
            parsed[current_original_index] = f"{parsed[current_original_index]} {line}".strip()
    return parsed


def validate_translations(translated_texts: list[str]) -> None:
    for index, text in enumerate(translated_texts, 1):
        if not text.strip():
            raise RuntimeError(f"Empty translation remains at subtitle {index}")


def write_srt(path: Path, entries: list[dict[str, str]], translated_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, (entry, translated_text) in enumerate(zip(entries, translated_texts), 1):
        blocks.append(f"{index}\n{entry['timecode']}\n{translated_text.strip()}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def validate_srt(path: Path, expected_count: int) -> None:
    blocks = [block for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()) if block.strip()]
    if len(blocks) != expected_count:
        raise RuntimeError(f"Output SRT count mismatch: expected {expected_count}, got {len(blocks)}")


if __name__ == "__main__":
    raise SystemExit(main())
