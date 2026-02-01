#!/bin/bash
# Usage: bash run_promptriever_example.sh [--quest_plus]
#   --quest_plus  Use QUEST withVariants data and distinct cache/output (no overwrite of default QUEST).
# Runs Promptriever-7B on Snellius (SLURM). Submit from repo root.

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
    export OUTPUT_FILE="./outputs/runs/promptriever-7b/results_withVariants_$(date +%Y%m%d_%H%M%S).jsonl"
    export CACHE_DIR="./outputs/runs/promptriever-7b/cache_withVariants"
    export INDEX_NAME="./outputs/runs/promptriever-7b/Promptriever-7B-QUEST_withVariants"
    export TASK_SUFFIX="_withVariants"
    export QUERY_FORMAT="quest"
    export CORPUS_FORMAT="quest_plus"
else
    export QUERY_FILE="./data/QUEST/test_id_added.jsonl"
    export CORPUS_FILE="./data/QUEST/documents.jsonl"
    export OUTPUT_FILE="./outputs/runs/promptriever-7b/results_$(date +%Y%m%d_%H%M%S).jsonl"
fi

export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_LENGTH="${MAX_LENGTH:-512}"

# Optional: subset for debugging
# export MAX_DOCS=100
# export MAX_QUERIES=10

# Optional: caching (content-keyed cache)
# export CACHE_DIR="./outputs/runs/promptriever-7b/cache"
# export USE_CACHE=0   # disable cache

# Optional: multi-GPU (request more GPUs in promptriever_7b_gpu.sh and set USE_MULTIGPU=1)
# export USE_MULTIGPU=1

# If repo is not in $HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR on Snellius:
# export PROJECT_ROOT=/path/to/SIGIR26_Repro_ComplexIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
sbatch --export=ALL "$SCRIPT_DIR/promptriever_7b_gpu.sh"

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
