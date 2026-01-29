#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <predictions-file> [retriever]"
  echo
  echo "If you don’t pass [retriever], it will be inferred from the parent folder of the predictions file."
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

PRED_FILE="$1"

# If user passed a retriever name, use it; otherwise infer from path
if [[ $# -eq 2 ]]; then
  RETRIEVER="$2"
else
  RETRIEVER="$(basename "$(dirname "$PRED_FILE")")"
fi

BASE_LOG_DIR="LOGS/Processing_Datasets/QUEST"
LOG_DIR="${BASE_LOG_DIR}/${RETRIEVER}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
LOG_PATH="${LOG_DIR}/${TIMESTAMP}.log"

echo "↪ Running preprocess for retriever=$RETRIEVER"
echo "  predictions → $PRED_FILE"
echo "  logging    → $LOG_PATH"
echo

python helpers/preprocess_data_for_experiments.py \
    --predictions "$PRED_FILE" \
    > "$LOG_PATH" 2>&1

echo
echo "✓ Done. Log written to $LOG_PATH"