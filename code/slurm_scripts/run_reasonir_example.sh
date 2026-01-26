#!/bin/bash
# Usage: bash run_reasonir_example.sh


# Set environment variables
export QUERY_FILE="./data/QUEST/test_id_added.jsonl"
export CORPUS_FILE="./data/QUEST/documents.jsonl"
export OUTPUT_FILE="./outputs/runs/reasonir-8b/results_$(date +%Y%m%d_%H%M%S).jsonl"
export BATCH_SIZE=16  
export DOC_BATCH_SIZE=32  # Separate batch size for documents (can be smaller for large corpora)
export INSPECT_BATCHES=1  # Set to 1 to inspect batches

# Optional: set quest_plus mode
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

# Optional: Quick run mode (for sanity checking - uses first 100 docs and 10 queries)
# export QUICK_RUN=1  # Enable quick run mode
# export QUICK_DOCS=100  # Number of documents (default: 100)
# export QUICK_QUERIES=10  # Number of queries (default: 10)

# Optional: Performance optimizations
export AUTO_BATCH=0  # Auto-tune batch sizes based on GPU memory (recommended!)
export USE_MULTIGPU=1  # Use multiple GPUs if available (requires --gres=gpu:2 in SLURM)

# --export=ALL passes all environment variables
sbatch --export=ALL code/slurm_scripts/reasonir_8b_gpu.sh

echo "=========================================="
echo "Job submitted with environment variables:"
echo "  QUERY_FILE=$QUERY_FILE"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  OUTPUT_FILE=$OUTPUT_FILE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  DOC_BATCH_SIZE=$DOC_BATCH_SIZE"
echo "  INSPECT_BATCHES=$INSPECT_BATCHES"
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
echo "=========================================="
echo ""
# First run will encode embeddings. Subsequent runs will be much faster with caching
