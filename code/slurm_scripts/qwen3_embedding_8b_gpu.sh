#!/bin/bash
#SBATCH --job-name=qwen3-emb-8b
#SBATCH --output=outputs/logs/qwen3-emb-8b-%j.out
#SBATCH --error=outputs/logs/qwen3-emb-8b-%j.err
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2  
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables for GPU optimization
# For multi-GPU, don't set CUDA_VISIBLE_DEVICES (let SLURM handle it)
# PyTorch will automatically detect all GPUs available to the job
# Count actual GPUs available via nvidia-smi (most reliable method)
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

# Also try to get from SLURM variables for logging
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    SLURM_NUM=$(echo $SLURM_GPUS_ON_NODE | wc -w)
    echo "SLURM_GPUS_ON_NODE reports: $SLURM_NUM GPU(s)"
elif [ -n "$SLURM_GPUS" ]; then
    SLURM_NUM=$SLURM_GPUS
    echo "SLURM_GPUS reports: $SLURM_NUM GPU(s)"
else
    echo "SLURM GPU variables not set, using nvidia-smi count"
fi

echo "nvidia-smi detected: $NUM_GPUS GPU(s)"
echo "Using $NUM_GPUS GPU(s) for this job"

# Only restrict CUDA_VISIBLE_DEVICES for single GPU
# For multi-GPU, let PyTorch see all GPUs (don't set CUDA_VISIBLE_DEVICES)
if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Setting CUDA_VISIBLE_DEVICES=0 for single GPU"
else
    # Don't set CUDA_VISIBLE_DEVICES - let PyTorch see all GPUs
    unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
    echo "Multi-GPU mode: Not setting CUDA_VISIBLE_DEVICES (PyTorch will see all $NUM_GPUS GPUs)"
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Conda: try common locations so it works on login and compute nodes
for CONDA_ROOT in "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        break
    fi
done
CONDA_ENV_NAME="${CONDA_ENV:-qwen3emb}"
conda activate "$CONDA_ENV_NAME" || { echo "ERROR: conda activate $CONDA_ENV_NAME failed. List envs: conda info --envs"; exit 1; }

echo "GPU Information:"
nvidia-smi

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
# Verify transformers version >= 4.51.0 (required by Qwen3-Embedding)
python -c 'import transformers; v=transformers.__version__; print(f"transformers version: {v}"); assert tuple(int(x) for x in v.split(".")[:2]) >= (4, 51), f"Qwen3-Embedding requires transformers>=4.51.0, got {v}"' || { echo "ERROR: transformers version too old. Run: pip install --upgrade transformers"; exit 1; }

# Project root (change if your repo is elsewhere on Snellius)
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR}"
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

# Set default paths (modify as needed)
QUERY_FILE="${QUERY_FILE:-./data/QUEST/test_id_added.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-./data/QUEST/documents.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/runs/qwen3-emb-8b/results_$(date +%Y%m%d_%H%M%S).jsonl}"
BATCH_SIZE="${BATCH_SIZE:-16}"            # Batch size for queries (conservative for 8B model)
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-16}"    # Batch size for documents

USE_CACHE="${USE_CACHE:-1}"               # Enable caching by default (set to 0 to disable)
CACHE_DIR="${CACHE_DIR:-}"                # Cache directory (default: auto-generated)
USE_FAISS="${USE_FAISS:-1}"               # Use FAISS by default (set to 0 for direct computation)
MAX_LENGTH="${MAX_LENGTH:-8192}"           # Maximum sequence length (Qwen3-Embedding supports up to 32768)

# explicit limits for docs/queries (preferred over QUICK_RUN)
MAX_DOCS="${MAX_DOCS:-}"                  # If set, limit to first N documents
MAX_QUERIES="${MAX_QUERIES:-}"            # If set, limit to first N queries

# QUEST withVariants: distinct cache/output (set by run_qwen3_embedding_example.sh --quest_plus)
TASK_SUFFIX="${TASK_SUFFIX:-}"
INDEX_NAME="${INDEX_NAME:-}"
QUERY_FORMAT="${QUERY_FORMAT:-}"
CORPUS_FORMAT="${CORPUS_FORMAT:-}"

# Performance optimization options
AUTO_BATCH="${AUTO_BATCH:-0}"             # Auto-tune batch sizes (default: disabled)
USE_MULTIGPU="${USE_MULTIGPU:-0}"         # Use multiple GPUs if available (default: disabled)

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p outputs/logs

# Run Qwen3-Embedding-8B with GPU optimizations
echo "=========================================="
echo "Starting Qwen3-Embedding-8B inference"
echo "Query file: $QUERY_FILE"
echo "Corpus file: $CORPUS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Query batch size: $BATCH_SIZE"
echo "Document batch size: $DOC_BATCH_SIZE"
echo "Caching: $([ "$USE_CACHE" = "0" ] && echo "disabled" || echo "enabled")"
echo "FAISS: $([ "$USE_FAISS" = "0" ] && echo "disabled (direct computation)" || echo "enabled")"
echo "Max length: $MAX_LENGTH"
if [ -n "$MAX_DOCS" ] || [ -n "$MAX_QUERIES" ]; then
    echo "SUBSET MODE: max_docs=${MAX_DOCS:-ALL}, max_queries=${MAX_QUERIES:-ALL}"
fi
if [ "$AUTO_BATCH" = "1" ]; then
    echo "AUTO BATCH TUNING: Enabled"
fi
if [ "$USE_MULTIGPU" = "1" ]; then
    echo "MULTI-GPU MODE: Enabled (using all available GPUs)"
fi
echo "INSPECT_BATCHES: ${INSPECT_BATCHES:-not set}"
[ -n "$TASK_SUFFIX" ] && echo "Task suffix: $TASK_SUFFIX"
[ -n "$INDEX_NAME" ] && echo "Index name: $INDEX_NAME"
[ -n "$QUERY_FORMAT" ] && echo "Query format: $QUERY_FORMAT"
[ -n "$CORPUS_FORMAT" ] && echo "Corpus format: $CORPUS_FORMAT"
echo "=========================================="

# Build command with optional arguments
CMD="python code/baselines/qwen3_embedding/qwen3-embedding-8b.py \
    --query-file \"$QUERY_FILE\" \
    --corpus-file \"$CORPUS_FILE\" \
    --output-file \"$OUTPUT_FILE\" \
    --device cuda \
    --batch-size \"$BATCH_SIZE\" \
    --doc-batch-size \"$DOC_BATCH_SIZE\" \
    --max-length \"$MAX_LENGTH\""

# Add optional arguments
[ -n "$QUEST_PLUS" ] && [ "$QUEST_PLUS" = "1" ] && CMD="$CMD --quest-plus"
[ -n "$MODEL_NAME" ] && CMD="$CMD --model-name \"$MODEL_NAME\""
[ -n "$INDEX_NAME" ] && CMD="$CMD --index-name \"$INDEX_NAME\""
[ "$USE_CACHE" = "0" ] && CMD="$CMD --no-cache"
[ -n "$CACHE_DIR" ] && CMD="$CMD --cache-dir \"$CACHE_DIR\""
[ -n "$TASK_SUFFIX" ] && CMD="$CMD --task-suffix \"$TASK_SUFFIX\""
[ -n "$QUERY_FORMAT" ] && CMD="$CMD --query-format \"$QUERY_FORMAT\""
[ -n "$CORPUS_FORMAT" ] && CMD="$CMD --corpus-format \"$CORPUS_FORMAT\""
[ "$USE_FAISS" = "0" ] && CMD="$CMD --no-faiss"
# Map subset controls to Python arguments
if [ -n "$MAX_DOCS" ]; then
    CMD="$CMD --max-docs \"$MAX_DOCS\""
elif [ "$QUICK_RUN" = "1" ]; then
    CMD="$CMD --max-docs \"$QUICK_DOCS\""
fi
if [ -n "$MAX_QUERIES" ]; then
    CMD="$CMD --max-queries \"$MAX_QUERIES\""
elif [ "$QUICK_RUN" = "1" ]; then
    CMD="$CMD --max-queries \"$QUICK_QUERIES\""
fi
[ "$AUTO_BATCH" = "1" ] && CMD="$CMD --auto-batch"
[ "$USE_MULTIGPU" = "1" ] && CMD="$CMD --multigpu"

# Execute command
eval $CMD

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully!"
    echo "Output saved to: $OUTPUT_FILE"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Job failed with exit code: $EXIT_CODE"
    echo "End Time: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi
