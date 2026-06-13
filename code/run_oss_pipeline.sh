#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash code/run_oss_pipeline.sh --model <qwen2.5-32b|qwen2.5-7b|gemma-3-27b|gemma-3-4b|llama-3.1-8b|llama-3.1-70b|gptoss|gpt-oss> --dataset <in-domain|in-domain/bin0|data/in-domain/bin0|./data/in-domain/bin0>

Examples:
  bash code/run_oss_pipeline.sh --model llama-3.1-8b --dataset in-domain
  bash code/run_oss_pipeline.sh --model llama-3.1-70b --dataset in-domain/bin0
  bash code/run_oss_pipeline.sh --model gpt-oss --dataset in-domain/bin0
  bash code/run_oss_pipeline.sh -m qwen2.5-32b -d ./data/in-domain/bin3
EOF
}

MODEL=""
DATASET_INPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="${2:-}"
      shift 2
      ;;
    -d|--dataset)
      DATASET_INPUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$DATASET_INPUT" ]]; then
  echo "Both --model and --dataset are required." >&2
  usage
  exit 1
fi

case "$MODEL" in
  gpt-oss) MODEL="gptoss" ;;
  qwen) MODEL="qwen2.5-32b" ;;
  gemma) MODEL="gemma-3-27b" ;;
  llama) MODEL="llama-3.1-8b" ;;
esac

case "$MODEL" in
  qwen2.5-32b|qwen2.5-7b|gemma-3-27b|gemma-3-4b|llama-3.1-8b|llama-3.1-70b|gptoss) ;;
  *)
    echo "Unsupported model: ${MODEL}. Use one of: qwen2.5-32b, qwen2.5-7b, gemma-3-27b, gemma-3-4b, llama-3.1-8b, llama-3.1-70b, gptoss (or gpt-oss)." >&2
    exit 1
    ;;
esac

normalize_dataset_dir() {
  local input="$1"
  local normalized="${input%/}"
  if [[ "$normalized" == ./data/* || "$normalized" == data/* ]]; then
    echo "./${normalized#./}"
  else
    echo "./data/${normalized#./}"
  fi
}

sanitize_model_label() {
  local input="$1"
  printf '%s' "$input" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

resolve_dataset_dirs() {
  local normalized_input="$1"
  local -a targets=()
  local dataset_basename
  dataset_basename="$(basename "$normalized_input")"

  # Single dataset mode: ./data/in-domain/bin0
  if [[ "$dataset_basename" =~ ^bin[0-9]+$ ]]; then
    if [[ ! -d "$normalized_input" ]]; then
      echo "Dataset directory not found: ${normalized_input}" >&2
      return 1
    fi
    targets+=("$normalized_input")
  else
    # Batch mode: ./data/in-domain -> run all bin* datasets found there.
    if [[ ! -d "$normalized_input" ]]; then
      echo "Dataset directory not found: ${normalized_input}" >&2
      return 1
    fi

    local candidate
    for candidate in "$normalized_input"/bin*; do
      if [[ -d "$candidate" ]]; then
        local candidate_base
        candidate_base="$(basename "$candidate")"
        if [[ "$candidate_base" =~ ^bin[0-9]+$ ]]; then
          targets+=("$candidate")
        fi
      fi
    done
  fi

  if [[ "${#targets[@]}" -eq 0 ]]; then
    echo "No bin datasets found under: ${normalized_input}" >&2
    return 1
  fi

  printf '%s\n' "${targets[@]}" | sort -V
}

DATASET_DIR_INPUT="$(normalize_dataset_dir "$DATASET_INPUT")"
mapfile -t DATASET_DIRS < <(resolve_dataset_dirs "$DATASET_DIR_INPUT")

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")"
RUN_GROUP_ID="$(date +%Y%m%d_%H%M%S)_${GIT_SHA}"

TRAIN_LOG_DIR="logs/train/${MODEL}"
INFER_LOG_DIR="logs/infer/${MODEL}"
EVAL_LOG_DIR="logs/eval/${MODEL}"
mkdir -p "$TRAIN_LOG_DIR" "$INFER_LOG_DIR" "$EVAL_LOG_DIR"

TRAIN_SCRIPT="code/fine-tuning/${MODEL}/train.py"
EXTRACT_SCRIPT="code/single-node/${MODEL}/extraction.py"
EVAL_SCRIPT="code/single-node/evaluation.py"

echo "Target datasets (${#DATASET_DIRS[@]}):"
printf ' - %s\n' "${DATASET_DIRS[@]}"
echo "model: ${MODEL}"
echo "run_group_id: ${RUN_GROUP_ID}"
echo

for DATASET_DIR in "${DATASET_DIRS[@]}"; do
  DATASET_BIN="$(basename "$DATASET_DIR")"
  DATASET_ARG="${DATASET_DIR}/"
  RUN_ID="${RUN_GROUP_ID}_${DATASET_BIN}"
  OUTPUT_SUFFIX="${RUN_ID}"
  MODEL_LABEL="$(sanitize_model_label "$MODEL")"

  TRAIN_LOG_PATH="${TRAIN_LOG_DIR}/${RUN_ID}.log"
  INFER_LOG_PATH="${INFER_LOG_DIR}/${RUN_ID}.log"
  EVAL_LOG_PATH="${EVAL_LOG_DIR}/${RUN_ID}.log"
  ADAPTER_DIR="artifacts/lora/${MODEL}/${DATASET_BIN}/${RUN_ID}/final"
  OUTPUT_FILE="test-set-${MODEL_LABEL}-singleagent-${OUTPUT_SUFFIX}.json"
  RESULT_FILE="${DATASET_DIR}/evaluation_result/test-set-${MODEL_LABEL}-singleagent-${OUTPUT_SUFFIX}-result.txt"

  echo "========================================"
  echo "Dataset: ${DATASET_DIR}"
  echo "run_id: ${RUN_ID}"
  echo "========================================"

  echo "=== [1/4] Prepare SFT dataset (${DATASET_BIN}) ==="
  python code/fine-tuning/oss_shared/prepare_sft_dataset.py -d "$DATASET_ARG"

  echo "=== [2/4] Fine-tuning (${MODEL}, ${DATASET_BIN}) ==="
  python "$TRAIN_SCRIPT" -d "$DATASET_ARG" --run-id "$RUN_ID" 2>&1 | tee "$TRAIN_LOG_PATH"

  if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "Adapter directory was not created: ${ADAPTER_DIR}" >&2
    exit 1
  fi

  echo "=== [3/4] Inference (${MODEL}, ${DATASET_BIN}) ==="
  python "$EXTRACT_SCRIPT" -d "$DATASET_ARG" --adapter_dir "$ADAPTER_DIR" --output-suffix "$OUTPUT_SUFFIX" 2>&1 | tee "$INFER_LOG_PATH"

  echo "=== [4/4] Evaluation (${MODEL}, ${DATASET_BIN}) ==="
  python "$EVAL_SCRIPT" -d "$DATASET_ARG" -f "$OUTPUT_FILE" --output-suffix "$OUTPUT_SUFFIX" 2>&1 | tee "$EVAL_LOG_PATH"

  echo
  echo "Completed: ${DATASET_DIR}"
  echo "train_log: ${TRAIN_LOG_PATH}"
  echo "infer_log: ${INFER_LOG_PATH}"
  echo "eval_log: ${EVAL_LOG_PATH}"
  echo "result_file: ${RESULT_FILE}"
  echo
done

echo "All dataset pipelines finished successfully."
