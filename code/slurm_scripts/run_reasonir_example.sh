#!/bin/bash
# Usage: bash run_reasonir_example.sh [--quest_plus]
#   --quest_plus  Use QUEST withVariants data and distinct cache/output (no overwrite of default QUEST).
# Runs ReasonIR-8B on Snellius (SLURM). Submit from repo root.

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
    export OUTPUT_FILE="./outputs/runs/reasonir-8b/results_withVariants_$(date +%Y%m%d_%H%M%S).jsonl"
    export CACHE_DIR="./outputs/runs/reasonir-8b/cache_withVariants"
    export INDEX_NAME="./outputs/runs/reasonir-8b/ReasonIR-8B-QUEST_withVariants"
    export TASK_SUFFIX="_withVariants"
    export QUERY_FORMAT="quest"
    export CORPUS_FORMAT="quest_plus"
else
    export QUERY_FILE="./data/QUEST/test_id_added.jsonl"
    export CORPUS_FILE="./data/QUEST/documents.jsonl"
    export OUTPUT_FILE="./outputs/runs/reasonir-8b/results_$(date +%Y%m%d_%H%M%S).jsonl"
fi

export BATCH_SIZE="${BATCH_SIZE:-16}"
export DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-32}"
export INSPECT_BATCHES="${INSPECT_BATCHES:-0}"

# Optional: set quest_plus mode (legacy; prefer --quest_plus flag above)
# export QUEST_PLUS=1

# Optional: override model or index name
# export MODEL_NAME="reasonir/ReasonIR-8B"
# export INDEX_NAME="ReasonIR-8B-QUEST"

# Optional: Caching options (enabled by default)
# export USE_CACHE=1  # Set to 0 to disable caching
# export CACHE_DIR="./cache"  # Custom cache directory

# Optional: Similarity computation
# export USE_FAISS=1  # Use FAISS (default, good for large corpora)
# export USE_FAISS=0  # Use direct matrix multiplication (faster for smaller corpora, matches retrievers.py in reasonir paper)

# Optional: Maximum sequence length
# export MAX_LENGTH=32768  # Default: 32768

# explicitly set hard limits (preferred for debugging)
# export MAX_DOCS=1000
# export MAX_QUERIES=100

# Optional: Performance optimizations
export AUTO_BATCH="${AUTO_BATCH:-0}"
export USE_MULTIGPU="${USE_MULTIGPU:-1}"

# Repo root = parent of code/ (so paths and logs work regardless of where you run from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
sbatch --export=ALL "$SCRIPT_DIR/reasonir_8b_gpu.sh"

echo "=========================================="
echo "Job submitted with environment variables:"
echo "  QUERY_FILE=$QUERY_FILE"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  OUTPUT_FILE=$OUTPUT_FILE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  DOC_BATCH_SIZE=$DOC_BATCH_SIZE"
echo "  INSPECT_BATCHES=$INSPECT_BATCHES"
[ "$QUEST_PLUS_MODE" = "1" ] && echo "  (quest_plus: CACHE_DIR, INDEX_NAME, TASK_SUFFIX, QUERY_FORMAT, CORPUS_FORMAT set)"
echo "  USE_CACHE=${USE_CACHE:-1} (default: enabled)"
echo "  USE_FAISS=${USE_FAISS:-1} (default: enabled)"
echo "  QUICK_RUN=${QUICK_RUN:-0} (default: disabled)"
if [ "${QUICK_RUN:-0}" = "1" ]; then
    echo "  QUICK_DOCS=${QUICK_DOCS:-100}"
    echo "  QUICK_QUERIES=${QUICK_QUERIES:-10}"
fi
if [ "${AUTO_BATCH:-0}" = "1" ]; then
    echo "  AUTO_BATCH: Enabled (will auto-tune batch sizes)"
fi
if [ "${USE_MULTIGPU:-0}" = "1" ]; then
    echo "  USE_MULTIGPU: Enabled (will use all available GPUs)"
fi
if [ -n "${MAX_DOCS}" ] || [ -n "${MAX_QUERIES}" ]; then
    echo "  MAX_DOCS=${MAX_DOCS:-'(not set)'}"
    echo "  MAX_QUERIES=${MAX_QUERIES:-'(not set)'}"
fi
echo "=========================================="
echo ""
# First run will encode embeddings. Subsequent runs will be much faster with caching
