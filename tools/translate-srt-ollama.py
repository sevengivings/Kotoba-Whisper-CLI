from __future__ import annotations

import argparse
import json
import re
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate SRT subtitles with Ollama.")
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", default="japanese")
    parser.add_argument("--target", default="korean")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", default="11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-translate", action="store_true")
    parser.add_argument("--text-split-size", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=int, default=600)
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
        text_split_size=args.text_split_size,
        timeout_seconds=args.timeout_seconds,
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
        "processing_seconds": round(time.time() - started, 3),
    }
    metadata_path = output_srt.with_suffix(".translation.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Info] Translation saved: {output_srt}")
    return 0


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
    text_split_size: int,
    timeout_seconds: int,
) -> list[str]:
    texts = [entry["text"] for entry in entries]
    translated = [""] * len(texts)
    if batch_translate:
        for batch in make_batches(texts, text_split_size):
            numbered = [f"[{local_no}] {texts[index]}" for local_no, index in enumerate(batch, 1)]
            result = translate_text_ollama(
                host,
                port,
                model,
                source_lang,
                target_lang,
                "\n".join(numbered),
                timeout_seconds,
            )
            parsed = parse_batch_translation(result, batch)
            for index, translated_text in parsed.items():
                translated[index] = translated_text

        missing = [index for index, value in enumerate(translated) if not value.strip()]
        if missing:
            print(f"[Warning] Retrying {len(missing)} missing subtitle(s) one by one")
            for index in missing:
                translated[index] = translate_text_ollama(
                    host, port, model, source_lang, target_lang, texts[index], timeout_seconds
                )
    else:
        for index, text in enumerate(texts, 1):
            print(f"[Info] Translating line {index}/{len(texts)}")
            translated[index - 1] = translate_text_ollama(
                host, port, model, source_lang, target_lang, text, timeout_seconds
            )

    validate_translations(translated)
    return translated


def make_batches(texts: list[str], text_split_size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_length = 0
    for index, text in enumerate(texts):
        if current and current_length + len(text) >= text_split_size:
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
) -> str:
    url = f"http://{host}:{port}/api/chat"
    batch_mode = "\n" in text or re.match(r"^\[\d+\]\s+", text.strip())
    if batch_mode:
        system_prompt = (
            "You are a professional video subtitle translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"{SUBTITLE_STYLE_PROMPT}"
            "The input contains lines numbered [N]. "
            "Translate each line separately and prefix the output with the same [N]. "
            "Do not merge lines. Do not renumber lines. Output only the translated text."
        )
    else:
        system_prompt = (
            "You are a professional video subtitle translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"{SUBTITLE_STYLE_PROMPT}"
            "Ensure the translation is natural and conversational. "
            "Do not include any introductory, concluding remarks, or notes. Output only the translated text."
        )
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
