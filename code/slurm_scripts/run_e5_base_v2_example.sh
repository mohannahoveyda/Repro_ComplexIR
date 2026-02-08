#!/bin/bash
# Usage: bash run_e5_base_v2_example.sh [--quest_plus]
#   --quest_plus  Use QUEST withVariants data and distinct cache/output (no overwrite of default QUEST).
# Runs e5-base-v2_main.py on Snellius (SLURM). Submit from repo root.

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
    export OUTPUT_FILE="./outputs/runs/e5-base-v2/results_withVariants_$(date +%Y%m%d_%H%M%S).jsonl"
    export CACHE_DIR="./outputs/runs/e5-base-v2/cache_withVariants"
    export INDEX_NAME="./outputs/runs/e5-base-v2/E5-base-v2-QUEST_withVariants"
    export TASK_SUFFIX="_withVariants"
    export QUERY_FORMAT="quest"
    export CORPUS_FORMAT="quest_plus"
else
    export QUERY_FILE="./data/QUEST/test_id_added.jsonl"
    export CORPUS_FILE="./data/QUEST/documents.jsonl"
    export OUTPUT_FILE="./outputs/runs/e5-base-v2/results_$(date +%Y%m%d_%H%M%S).jsonl"
    export INDEX_NAME="./outputs/runs/e5-base-v2/E5-base-v2-QUEST"
fi

export BATCH_SIZE="${BATCH_SIZE:-32}"

# Optional: subset for debugging
# export MAX_DOCS=1000
# export MAX_QUERIES=100

# Optional: caching (e5-base-v2_main.py; content-keyed cache)
# export CACHE_DIR="./cache"
# export NO_CACHE=1   # disable cache

# Optional: multi-GPU (request more GPUs in e5_base_v2_gpu.sh and set MULTIGPU=1)
# export MULTIGPU=1

# Optional: batch size tuning
# export DOC_BATCH_SIZE=16  # Smaller batch size for documents if needed
# export MAX_LENGTH=512  # Maximum sequence length (default: 512)

# Optional: performance optimizations
# export AUTO_BATCH=1  # Auto-tune batch sizes
# export NO_FAISS=1   # Use direct matrix multiplication instead of FAISS

# If repo is not in $HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR on Snellius:
# export PROJECT_ROOT=/path/to/SIGIR26_Repro_ComplexIR

# Repo root = parent of code/ (so logs go to repo/outputs/logs regardless of where you run from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
sbatch --export=ALL "$SCRIPT_DIR/e5_base_v2_gpu.sh"

echo "=========================================="
echo "Job submitted with:"
echo "  QUERY_FILE=$QUERY_FILE"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  OUTPUT_FILE=$OUTPUT_FILE"
echo "  BATCH_SIZE=$BATCH_SIZE"
[ "$QUEST_PLUS_MODE" = "1" ] && echo "  (quest_plus: CACHE_DIR, INDEX_NAME, TASK_SUFFIX, QUERY_FORMAT, CORPUS_FORMAT set)"
if [ -n "${MAX_DOCS}" ] || [ -n "${MAX_QUERIES}" ]; then
    echo "  MAX_DOCS=${MAX_DOCS:-'(not set)'}"
    echo "  MAX_QUERIES=${MAX_QUERIES:-'(not set)'}"
fi
echo "=========================================="
