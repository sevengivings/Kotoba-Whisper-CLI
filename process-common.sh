#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
  echo "Error: $*" >&2
  exit 1
}

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    return 1
  fi
}

path_stem() {
  local name="$1"
  echo "${name%.*}"
}

path_ext() {
  local name="$1"
  if [[ "$name" == *.* ]]; then
    echo ".${name##*.}"
  else
    echo ""
  fi
}

ensure_docker_watcher() {
  command -v docker >/dev/null 2>&1 || die "docker command was not found. Install Docker Engine or Docker Desktop."
  docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start Docker, then try again."

  local status
  status="$(docker ps --filter name=kotoba-folder-watcher --format '{{.Status}}' 2>/dev/null || true)"
  if [[ -n "$status" ]]; then
    return
  fi

  echo "Kotoba folder watcher is not running. Starting it first..."
  (cd "$SCRIPT_DIR" && docker compose up -d --build)
  status="$(docker ps --filter name=kotoba-folder-watcher --format '{{.Status}}' 2>/dev/null || true)"
  [[ -n "$status" ]] || die "Kotoba folder watcher did not start. Check 'docker logs kotoba-folder-watcher'."
}

submission_name_available() {
  local target_name="$1"
  local target_path="$2"
  local input_dir="$3"
  local output_dir="$4"
  local failed_dir="$5"
  local archive_dir="$6"
  local processing_dir="$7"
  local base_name
  base_name="$(path_stem "$target_name")"
  local options_name="$target_name.options.json"

  local paths=(
    "$target_path"
    "$target_path.part"
    "$input_dir/$target_name.part"
    "$input_dir/$options_name"
    "$input_dir/$options_name.part"
    "$processing_dir/$options_name"
    "$processing_dir/$target_name"
    "$archive_dir/$target_name"
    "$failed_dir/$target_name"
    "$output_dir/$base_name.ja.srt"
    "$output_dir/$base_name.process.json"
  )

  local candidate
  for candidate in "${paths[@]}"; do
    [[ ! -e "$candidate" ]] || return 1
  done
  return 0
}

write_job_options() {
  local options_path="$1"
  local silence_threshold_db="$2"
  local auto_silence_threshold="$3"
  local min_silence_duration_s="$4"
  local delete_source_on_success="$5"
  local py
  py="$(python_cmd)" || die "Python was not found. Install python3."

  if [[ -n "$silence_threshold_db" && ! "$silence_threshold_db" =~ ^-?[0-9]+([.][0-9]+)?dB$ ]]; then
    die "silence threshold must look like -35dB"
  fi

  local part_path="$options_path.part"
  rm -f -- "$part_path"
  "$py" - "$part_path" "$silence_threshold_db" "$auto_silence_threshold" "$min_silence_duration_s" "$delete_source_on_success" <<'PY'
import json
import sys

path, threshold, auto_threshold, min_silence, delete_source = sys.argv[1:6]
options = {"delete_source_on_success": delete_source == "true"}
if auto_threshold == "true":
    options["auto_silence_threshold"] = True
if threshold:
    options["silence_threshold_db"] = threshold
if min_silence and float(min_silence) > 0:
    options["min_silence_duration_s"] = float(min_silence)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(options, handle, ensure_ascii=False, indent=2)
PY
  mv -f -- "$part_path" "$options_path"
}

ollama_base_uri() {
  echo "http://$OLLAMA_HOST:$OLLAMA_PORT"
}

get_ollama_models() {
  local uri
  uri="$(ollama_base_uri)/api/tags"
  local py
  py="$(python_cmd)" || die "Python was not found. Install python3."
  curl -fsS "$uri" | "$py" -c 'import json,sys; print("\n".join(m.get("name","") for m in json.load(sys.stdin).get("models", []) if m.get("name","")))'
}

assert_ollama_model_available() {
  local model="$1"
  local models
  models="$(get_ollama_models)" || die "Ollama server is not reachable at $(ollama_base_uri)/api/tags. Start Ollama, or pass --ollama-host/--ollama-port."
  [[ -n "$models" ]] || die "No Ollama models found. Run 'ollama pull <model>' first."
  if ! grep -Fxq -- "$model" <<<"$models"; then
    die "Ollama model '$model' was not found at $(ollama_base_uri). Available models: $(tr '\n' ',' <<<"$models" | sed 's/,$//')"
  fi
}

translation_defaults_path() {
  echo "$SCRIPT_DIR/config/translation-defaults.json"
}

get_saved_translation_model() {
  local path
  path="$(translation_defaults_path)"
  [[ -f "$path" ]] || return 0
  local py
  py="$(python_cmd)" || return 0
  "$py" - "$path" <<'PY' || true
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8-sig") as handle:
        print(json.load(handle).get("ollama_model", ""))
except Exception:
    pass
PY
}

save_translation_model() {
  local model="$1"
  local path
  path="$(translation_defaults_path)"
  local py
  py="$(python_cmd)" || die "Python was not found. Install python3."
  mkdir -p -- "$(dirname "$path")"
  "$py" - "$path" "$model" <<'PY'
import json
import sys
from datetime import datetime

path, model = sys.argv[1:3]
payload = {
    "provider": "ollama",
    "ollama_model": model,
    "updated_at": datetime.now().replace(microsecond=0).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
}

select_ollama_model() {
  local models
  models="$(get_ollama_models)" || die "Ollama server is not reachable at $(ollama_base_uri)/api/tags. Start Ollama, or pass --ollama-host/--ollama-port."
  [[ -n "$models" ]] || die "No Ollama models found. Run 'ollama pull <model>' first."

  echo "Available Ollama models:" >&2
  local index=1
  local model
  while IFS= read -r model; do
    echo "  [$index] $model" >&2
    index=$((index + 1))
  done <<<"$models"

  local choice
  read -r -p "Choose translation model number: " choice
  [[ "$choice" =~ ^[0-9]+$ ]] || die "Invalid model choice: $choice"
  local selected
  selected="$(sed -n "${choice}p" <<<"$models")"
  [[ -n "$selected" ]] || die "Invalid model choice: $choice"
  echo "$selected"
}

resolve_translation_model() {
  local requested_model="$1"
  local choose_model="$2"
  local model
  if [[ -n "$requested_model" ]]; then
    assert_ollama_model_available "$requested_model"
    echo "$requested_model"
    return
  fi
  if [[ "$choose_model" == "true" ]]; then
    select_ollama_model
    return
  fi
  model="$(get_saved_translation_model)"
  if [[ -n "$model" ]]; then
    assert_ollama_model_available "$model"
    echo "$model"
    return
  fi
  die "No translation model configured. Use --translation-model <model> or --translate-model-choice."
}

invoke_srt_translation() {
  local input_srt="$1"
  local model="$2"
  local py
  py="$(python_cmd)" || die "Python was not found. Install python3."

  local output_srt
  if [[ "$input_srt" == *.ja.srt ]]; then
    output_srt="${input_srt%.ja.srt}.ko.srt"
  else
    output_srt="${input_srt%.*}.ko.srt"
  fi

  echo
  echo "Translating SRT:"
  echo "  Input:  $input_srt"
  echo "  Output: $output_srt"
  echo "  Model:  $model"

  local args=(
    "$SCRIPT_DIR/tools/translate-srt-ollama.py"
    "$input_srt"
    "--output" "$output_srt"
    "--ollama-host" "$OLLAMA_HOST"
    "--ollama-port" "$OLLAMA_PORT"
    "--model" "$model"
    "--batch-size" "$BATCH_SIZE"
    "--text-split-size" "$TEXT_SPLIT_SIZE"
    "--korean-style" "$KOREAN_STYLE"
  )
  if [[ "$BATCH_TRANSLATE" == "true" ]]; then
    args+=("--batch-translate")
  fi
  if [[ "$NO_BATCH_TRANSLATE" == "true" ]]; then
    args+=("--no-batch-translate")
  fi

  "$py" "${args[@]}"
  save_translation_model "$model"
  echo "Translation completed:"
  echo "  $output_srt"
}

validate_common_options() {
  if [[ "$TRANSLATE" == "true" && "$NO_WAIT" == "true" ]]; then
    die "--translate requires the default wait mode. Remove --no-wait."
  fi
  if [[ "$AUTO_SILENCE_THRESHOLD" == "true" && -n "$SILENCE_THRESHOLD_DB" ]]; then
    die "Use either --auto-silence-threshold or --silence-threshold-db, not both."
  fi
  if [[ "$BATCH_TRANSLATE" == "true" && "$NO_BATCH_TRANSLATE" == "true" ]]; then
    die "Use either --batch-translate or --no-batch-translate, not both."
  fi
  case "$KOREAN_STYLE" in
    polite|banmal|strict-banmal) ;;
    *) die "--korean-style must be polite, banmal, or strict-banmal" ;;
  esac
}
