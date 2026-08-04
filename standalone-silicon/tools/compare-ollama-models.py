from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


DEFAULT_MODELS = [
    "hf.co/mradermacher/Hy-MT2-30B-A3B-GGUF:Q4_K_M",
    "hf.co/mradermacher/Hy-MT2-7B-GGUF:Q4_K_M",
    "hf.co/mradermacher/Hy-MT2-1.8B-GGUF:Q4_K_M",
]

DEFAULT_TEXT = "\n".join(
    [
        "[1] フロントの手違いで予約したホテルが相部屋なんて本当に驚きましたね。",
        "[2] 今から別のホテルを探すんですか？この時間では難しいと思います。",
        "[3] 先輩だからって遠慮しないでください。",
    ]
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Ollama subtitle translation models.")
    parser.add_argument("--model", action="append", dest="models", help="Model to test. Can be passed more than once.")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    print(f"Testing {len(models)} model(s) with timeout {args.timeout_seconds}s")
    print("")

    for model in models:
        started = time.time()
        print(f"Model: {model}")
        try:
            output = translate_once(args.ollama_host, args.ollama_port, model, args.text, args.timeout_seconds)
        except Exception as exc:
            elapsed = time.time() - started
            print(f"  Failed after {elapsed:.3f}s")
            print(f"  {exc}")
            print("")
            continue
        elapsed = time.time() - started
        print(f"  Succeeded in {elapsed:.3f}s")
        for line in output.splitlines()[:8]:
            print(f"  {line}")
        print("")
    return 0


def translate_once(host: str, port: int, model: str, text: str, timeout_seconds: int) -> str:
    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional video subtitle translator. "
                    "Translate Japanese to natural polite Korean. "
                    "The input contains lines numbered [N]. "
                    "Translate each line separately and prefix the output with the same [N]. "
                    "Output only the translated text."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 256,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return str(data.get("message", {}).get("content", "")).strip()


if __name__ == "__main__":
    raise SystemExit(main())
