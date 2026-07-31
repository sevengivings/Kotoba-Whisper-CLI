#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/process-common.sh"

usage() {
  cat <<'EOF'
Usage: ./process-dir.sh [options] <directory>

Options:
  --recurse
  --no-wait
  --keep-staged-copy
  --silence-threshold-db -35dB
  --auto-silence-threshold
  --min-silence-duration-seconds 0.4
  --timeout-minutes MINUTES
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
EOF
}

RECURSE=false
NO_WAIT=false
KEEP_STAGED_COPY=false
SILENCE_THRESHOLD_DB=""
AUTO_SILENCE_THRESHOLD=false
MIN_SILENCE_DURATION_SECONDS=0
TIMEOUT_MINUTES=1440
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
DIR_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recurse) RECURSE=true; shift ;;
    --no-wait) NO_WAIT=true; shift ;;
    --wait) shift ;;
    --keep-staged-copy) KEEP_STAGED_COPY=true; shift ;;
    --silence-threshold-db) SILENCE_THRESHOLD_DB="${2:-}"; shift 2 ;;
    --auto-silence-threshold) AUTO_SILENCE_THRESHOLD=true; shift ;;
    --min-silence-duration-seconds) MIN_SILENCE_DURATION_SECONDS="${2:-0}"; shift 2 ;;
    --timeout-minutes) TIMEOUT_MINUTES="${2:-1440}"; shift 2 ;;
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
    -h|--help) usage; exit 0 ;;
    --*) die "Unknown option: $1" ;;
    *) DIR_PATH="$1"; shift ;;
  esac
done

[[ -n "$DIR_PATH" ]] || { usage; exit 1; }
[[ -d "$DIR_PATH" ]] || die "Directory not found: $DIR_PATH"

validate_common_options

translation_model=""
if [[ "$TRANSLATE" == "true" ]]; then
  translation_model="$(resolve_translation_model "$TRANSLATION_MODEL" "$TRANSLATE_MODEL_CHOICE")"
fi

mapfile -d '' sources < <(
  if [[ "$RECURSE" == "true" ]]; then
    find "$DIR_PATH" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' -o -iname '*.ts' -o -iname '*.m2ts' -o -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.flac' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.wma' \) -print0
  else
    find "$DIR_PATH" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' -o -iname '*.ts' -o -iname '*.m2ts' -o -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.flac' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.wma' \) -print0
  fi | sort -z
)

if [[ "${#sources[@]}" -eq 0 ]]; then
  echo "No supported media files found:"
  echo "  $DIR_PATH"
  exit 0
fi

ensure_docker_watcher

input_dir="$SCRIPT_DIR/input"
output_dir="$SCRIPT_DIR/output"
failed_dir="$SCRIPT_DIR/failed"
archive_dir="$SCRIPT_DIR/archive"
processing_dir="$SCRIPT_DIR/processing"
mkdir -p -- "$input_dir"

echo "Submitting directory:"
echo "  Source: $(realpath "$DIR_PATH")"
echo "  Files:  ${#sources[@]}"
echo "  Recurse: $RECURSE"
[[ -z "$SILENCE_THRESHOLD_DB" ]] || echo "  Silence threshold override: $SILENCE_THRESHOLD_DB"
[[ "$AUTO_SILENCE_THRESHOLD" != "true" ]] || echo "  Silence threshold override: auto"
awk "BEGIN { exit !($MIN_SILENCE_DURATION_SECONDS > 0) }" && echo "  Min silence duration override: $MIN_SILENCE_DURATION_SECONDS" || true
if [[ "$KEEP_STAGED_COPY" == "true" ]]; then
  echo "  Staged copy: keep after success"
else
  echo "  Staged copy: delete after success"
fi
echo

submitted_target_names=()
submitted_base_names=()
submitted_source_paths=()
submitted_output_srts=()
submitted_process_jsons=()
submitted_progress_jsons=()
submitted_failed_files=()

for source_path in "${sources[@]}"; do
  source_name="$(basename "$source_path")"
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
  echo "  Source: $source_path"
  echo "  Staged: $target_path"

  delete_source_on_success=true
  [[ "$KEEP_STAGED_COPY" == "false" ]] || delete_source_on_success=false
  write_job_options "$options_path" "$SILENCE_THRESHOLD_DB" "$AUTO_SILENCE_THRESHOLD" "$MIN_SILENCE_DURATION_SECONDS" "$delete_source_on_success"
  cp -- "$source_path" "$part_path"
  mv -- "$part_path" "$target_path"

  submitted_base_name="$(path_stem "$target_name")"
  submitted_target_names+=("$target_name")
  submitted_base_names+=("$submitted_base_name")
  submitted_source_paths+=("$source_path")
  submitted_output_srts+=("$output_dir/$submitted_base_name.ja.srt")
  submitted_process_jsons+=("$output_dir/$submitted_base_name.process.json")
  submitted_progress_jsons+=("$processing_dir/$submitted_base_name.progress.json")
  submitted_failed_files+=("$failed_dir/$target_name")
done

echo
echo "Submitted ${#submitted_target_names[@]} file(s). The watcher will process them after the file stability check."

if [[ "$NO_WAIT" == "true" ]]; then
  echo
  echo "Watch progress:"
  echo "  docker logs -f kotoba-folder-watcher"
  echo
  echo "Check status:"
  echo "  docker compose ps"
  exit 0
fi

deadline=$((SECONDS + TIMEOUT_MINUTES * 60))
completed=()
for _ in "${submitted_target_names[@]}"; do
  completed+=("false")
done

echo "Waiting for completion. Timeout: $TIMEOUT_MINUTES minutes"
while [[ "$SECONDS" -lt "$deadline" ]]; do
  pending=0
  for i in "${!submitted_target_names[@]}"; do
    if [[ "${completed[$i]}" == "true" ]]; then
      continue
    fi

    if [[ -f "${submitted_process_jsons[$i]}" ]]; then
      finish_progress_line
      echo
      echo "Completed: ${submitted_target_names[$i]}"
      if [[ -f "${submitted_output_srts[$i]}" ]]; then
        echo "  SRT: ${submitted_output_srts[$i]}"
        if [[ "$TRANSLATE" == "true" ]]; then
          invoke_srt_translation "${submitted_output_srts[$i]}" "$translation_model" "${submitted_source_paths[$i]}"
        fi
      fi
      completed[$i]="true"
      continue
    fi

    if [[ -f "${submitted_failed_files[$i]}" ]]; then
      finish_progress_line
      echo
      echo "Failed: ${submitted_target_names[$i]}"
      echo "  Source moved to: ${submitted_failed_files[$i]}"
      failure_json="$failed_dir/${submitted_base_names[$i]}.failure.json"
      [[ ! -f "$failure_json" ]] || cat "$failure_json"
      completed[$i]="true"
      continue
    fi

    pending=$((pending + 1))
  done

  if [[ "$pending" -eq 0 ]]; then
    finish_progress_line
    echo
    echo "All submitted files completed."
    exit 0
  fi

  if [[ "$pending" -eq 1 ]]; then
    for i in "${!submitted_target_names[@]}"; do
      if [[ "${completed[$i]}" == "true" ]]; then
        continue
      fi
      if summary="$(progress_summary "${submitted_progress_jsons[$i]}")"; then
        print_progress_line "Still waiting: ${submitted_target_names[$i]} | $summary"
      else
        print_progress_line "Still waiting: ${submitted_target_names[$i]} | queued or waiting for worker"
      fi
    done
  else
    finish_progress_line
    echo "Still waiting: $pending file(s)"
    for i in "${!submitted_target_names[@]}"; do
      if [[ "${completed[$i]}" == "true" ]]; then
        continue
      fi
      if summary="$(progress_summary "${submitted_progress_jsons[$i]}")"; then
        echo "  ${submitted_target_names[$i]}: $summary"
      else
        echo "  ${submitted_target_names[$i]}: queued or waiting for worker"
      fi
    done
  fi
  sleep 10
done

finish_progress_line
die "Timed out while waiting for completion."
