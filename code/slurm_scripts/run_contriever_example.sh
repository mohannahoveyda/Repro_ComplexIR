#!/bin/bash
# Usage: bash run_contriever_example.sh [--quest_plus]
#   --quest_plus  Use QUEST withVariants data and distinct cache/output.
# Runs Contriever (facebook/contriever) on Snellius (SLURM). Submit from repo root.

QUEST_PLUS_MODE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quest_plus|--quest-plus)
            QUEST_PLUS_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--quest_plus]" >&2
            exit 1
            ;;
    esac
done

if [ "$QUEST_PLUS_MODE" = "1" ]; then
    export QUERY_FILE="./data/QUEST_w_Variants/data/quest_test_withVarients_converted.jsonl"
    export CORPUS_FILE="./data/QUEST_w_Variants/data/quest_text_w_id_withVarients.jsonl"
    export OUTPUT_FILE="./outputs/runs/contriever/results_withVariants_$(date +%Y%m%d_%H%M%S).jsonl"
    export CACHE_DIR="./outputs/runs/contriever/cache_withVariants"
    export INDEX_NAME="./outputs/runs/contriever/Contriever-QUEST_withVariants"
    export TASK_SUFFIX="_withVariants"
    export QUERY_FORMAT="quest"
    export CORPUS_FORMAT="quest_plus"
else
    export QUERY_FILE="./data/QUEST/test_id_added.jsonl"
    export CORPUS_FILE="./data/QUEST/documents.jsonl"
    export OUTPUT_FILE="./outputs/runs/contriever/results_$(date +%Y%m%d_%H%M%S).jsonl"
fi

export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_LENGTH="${MAX_LENGTH:-512}"

# Optional: subset for debugging
# export MAX_DOCS=1000
# export MAX_QUERIES=100

# Optional: use MS MARCO finetuned model
# export MODEL_NAME="facebook/contriever-msmarco"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
sbatch --export=ALL "$SCRIPT_DIR/contriever_gpu.sh"

echo "=========================================="
echo "Job submitted with:"
echo "  QUERY_FILE=$QUERY_FILE"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  OUTPUT_FILE=$OUTPUT_FILE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  MAX_LENGTH=$MAX_LENGTH"
[ "$QUEST_PLUS_MODE" = "1" ] && echo "  (quest_plus: CACHE_DIR, INDEX_NAME, TASK_SUFFIX, QUERY_FORMAT, CORPUS_FORMAT set)"
if [ -n "${MAX_DOCS}" ] || [ -n "${MAX_QUERIES}" ]; then
    echo "  MAX_DOCS=${MAX_DOCS:-'(not set)'}"
    echo "  MAX_QUERIES=${MAX_QUERIES:-'(not set)'}"
fi
echo "=========================================="
