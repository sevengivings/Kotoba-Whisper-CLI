#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/process-common.sh"

usage() {
  cat <<'EOF'
Usage: ./process-file.sh [options] <media-file>

Options:
  --no-wait
  --keep-staged-copy
  --silence-threshold-db -35dB
  --auto-silence-threshold
  --min-silence-duration-seconds 0.4
  --translate
  --translate-model-choice
  --translation-model MODEL
  --ollama-host HOST
  --ollama-port PORT
  --batch-translate
  --no-batch-translate
  --batch-size COUNT
  --text-split-size CHARS
  --korean-style polite|banmal|strict-banmal
  --timeout-minutes MINUTES
EOF
}

NO_WAIT=false
KEEP_STAGED_COPY=false
SILENCE_THRESHOLD_DB=""
AUTO_SILENCE_THRESHOLD=false
MIN_SILENCE_DURATION_SECONDS=0
TRANSLATE=false
TRANSLATE_MODEL_CHOICE=false
TRANSLATION_MODEL=""
OLLAMA_HOST="localhost"
OLLAMA_PORT=11434
BATCH_TRANSLATE=false
NO_BATCH_TRANSLATE=false
BATCH_SIZE=50
TEXT_SPLIT_SIZE=0
KOREAN_STYLE="polite"
TIMEOUT_MINUTES=240
MEDIA_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait) NO_WAIT=true; shift ;;
    --wait) shift ;;
    --keep-staged-copy) KEEP_STAGED_COPY=true; shift ;;
    --silence-threshold-db) SILENCE_THRESHOLD_DB="${2:-}"; shift 2 ;;
    --auto-silence-threshold) AUTO_SILENCE_THRESHOLD=true; shift ;;
    --min-silence-duration-seconds) MIN_SILENCE_DURATION_SECONDS="${2:-0}"; shift 2 ;;
    --translate) TRANSLATE=true; shift ;;
    --translate-model-choice) TRANSLATE_MODEL_CHOICE=true; shift ;;
    --translation-model) TRANSLATION_MODEL="${2:-}"; shift 2 ;;
    --ollama-host) OLLAMA_HOST="${2:-localhost}"; shift 2 ;;
    --ollama-port) OLLAMA_PORT="${2:-11434}"; shift 2 ;;
    --batch-translate) BATCH_TRANSLATE=true; shift ;;
    --no-batch-translate) NO_BATCH_TRANSLATE=true; shift ;;
    --batch-size) BATCH_SIZE="${2:-50}"; shift 2 ;;
    --text-split-size) TEXT_SPLIT_SIZE="${2:-0}"; shift 2 ;;
    --korean-style) KOREAN_STYLE="${2:-polite}"; shift 2 ;;
    --timeout-minutes) TIMEOUT_MINUTES="${2:-240}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) die "Unknown option: $1" ;;
    *) MEDIA_PATH="$1"; shift ;;
  esac
done

[[ -n "$MEDIA_PATH" ]] || { usage; exit 1; }
[[ -f "$MEDIA_PATH" ]] || die "File not found: $MEDIA_PATH"

validate_common_options

translation_model=""
if [[ "$TRANSLATE" == "true" ]]; then
  translation_model="$(resolve_translation_model "$TRANSLATION_MODEL" "$TRANSLATE_MODEL_CHOICE")"
fi

ensure_docker_watcher

input_dir="$SCRIPT_DIR/input"
output_dir="$SCRIPT_DIR/output"
failed_dir="$SCRIPT_DIR/failed"
archive_dir="$SCRIPT_DIR/archive"
processing_dir="$SCRIPT_DIR/processing"
mkdir -p -- "$input_dir"

source_name="$(basename "$MEDIA_PATH")"
base_name="$(path_stem "$source_name")"
extension="$(path_ext "$source_name")"
target_name="$source_name"
target_path="$input_dir/$target_name"
index=1

while ! submission_name_available "$target_name" "$target_path" "$input_dir" "$output_dir" "$failed_dir" "$archive_dir" "$processing_dir"; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  if [[ "$index" -eq 1 ]]; then
    target_name="${base_name}_${stamp}${extension}"
  else
    target_name="${base_name}_${stamp}_${index}${extension}"
  fi
  target_path="$input_dir/$target_name"
  index=$((index + 1))
done

part_path="$target_path.part"
options_path="$input_dir/$target_name.options.json"
rm -f -- "$part_path"

echo "Submitting file:"
echo "  Source: $(realpath "$MEDIA_PATH")"
echo "  Staged: $target_path"
[[ -z "$SILENCE_THRESHOLD_DB" ]] || echo "  Silence threshold override: $SILENCE_THRESHOLD_DB"
[[ "$AUTO_SILENCE_THRESHOLD" != "true" ]] || echo "  Silence threshold override: auto"
awk "BEGIN { exit !($MIN_SILENCE_DURATION_SECONDS > 0) }" && echo "  Min silence duration override: $MIN_SILENCE_DURATION_SECONDS" || true
if [[ "$KEEP_STAGED_COPY" == "true" ]]; then
  echo "  Staged copy: keep after success"
else
  echo "  Staged copy: delete after success"
fi
echo

delete_source_on_success=true
[[ "$KEEP_STAGED_COPY" == "false" ]] || delete_source_on_success=false
write_job_options "$options_path" "$SILENCE_THRESHOLD_DB" "$AUTO_SILENCE_THRESHOLD" "$MIN_SILENCE_DURATION_SECONDS" "$delete_source_on_success"
cp -- "$MEDIA_PATH" "$part_path"
mv -- "$part_path" "$target_path"

echo "Submitted. The watcher will process it after the file stability check."

if [[ "$NO_WAIT" == "true" ]]; then
  echo
  echo "Watch progress:"
  echo "  docker logs -f kotoba-folder-watcher"
  echo
  echo "Check status:"
  echo "  docker compose ps"
  exit 0
fi

submitted_base_name="$(path_stem "$target_name")"
deadline=$((SECONDS + TIMEOUT_MINUTES * 60))
output_srt="$output_dir/$submitted_base_name.ja.srt"
process_json="$output_dir/$submitted_base_name.process.json"
failed_file="$failed_dir/$target_name"

echo "Waiting for completion. Timeout: $TIMEOUT_MINUTES minutes"
while [[ "$SECONDS" -lt "$deadline" ]]; do
  if [[ -f "$process_json" ]]; then
    echo
    echo "Completed:"
    cat "$process_json"
    if [[ -f "$output_srt" ]]; then
      echo
      echo "SRT:"
      echo "  $output_srt"
      if [[ "$TRANSLATE" == "true" ]]; then
        invoke_srt_translation "$output_srt" "$translation_model"
      fi
    fi
    exit 0
  fi

  if [[ -f "$failed_file" ]]; then
    echo
    echo "Failed. Source moved to:"
    echo "  $failed_file"
    failure_json="$failed_dir/$submitted_base_name.failure.json"
    [[ ! -f "$failure_json" ]] || cat "$failure_json"
    exit 1
  fi

  sleep 10
done

die "Timed out while waiting for completion: $target_name"
