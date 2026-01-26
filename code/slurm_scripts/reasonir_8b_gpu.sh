#!/bin/bash
#SBATCH --job-name=reasonir-8b
#SBATCH --output=outputs/logs/reasonir-8b-%j.out
#SBATCH --error=outputs/logs/reasonir-8b-%j.err
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

# Load modules (adjust based on Snellius setup)
# Uncomment and modify based on your Snellius environment
# module load Python/3.10.4-GCCcore-11.3.0
# module load CUDA/11.7.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0

# Set environment variables for GPU optimization
# For multi-GPU, don't set CUDA_VISIBLE_DEVICES (let SLURch handle it)
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

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate reasonir

# Workaround for tokenizer loading issue: update tokenizers if needed
# The error "data did not match any variant of untagged enum ModelWrapper" 
# can occur with incompatible tokenizers library versions
# Uncomment the following line if you encounter tokenizer errors:
# pip install --upgrade tokenizers --quiet

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print Python and package versions
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Navigate to project root
cd $HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR

# Set default paths (modify as needed)
QUERY_FILE="${QUERY_FILE:-./data/QUEST/test_id_added.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-./data/QUEST/documents.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/runs/reasonir-8b/results_$(date +%Y%m%d_%H%M%S).jsonl}"
BATCH_SIZE="${BATCH_SIZE:-64}"  # Batch size for queries (optimized: 32-128 for A100)
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-32}"  # Batch size for documents (can be smaller for large corpora)

# New optimization options
USE_CACHE="${USE_CACHE:-1}"  # Enable caching by default (set to 0 to disable)
CACHE_DIR="${CACHE_DIR:-}"  # Cache directory (default: auto-generated)
USE_FAISS="${USE_FAISS:-1}"  # Use FAISS by default (set to 0 for direct computation)
MAX_LENGTH="${MAX_LENGTH:-4096}"  # Maximum sequence length

# Quick run options (for sanity checking)
QUICK_RUN="${QUICK_RUN:-0}"  # Enable quick run mode (default: disabled)
QUICK_DOCS="${QUICK_DOCS:-100}"  # Number of documents in quick run (default: 100)
QUICK_QUERIES="${QUICK_QUERIES:-10}"  # Number of queries in quick run (default: 10)

# Performance optimization options
AUTO_BATCH="${AUTO_BATCH:-0}"  # Auto-tune batch sizes (default: disabled)
USE_MULTIGPU="${USE_MULTIGPU:-0}"  # Use multiple GPUs if available (default: disabled)

# Create output directory if it doesn't exist
mkdir -p $(dirname "$OUTPUT_FILE")

# Run ReasonIR-8B with GPU optimizations
echo "=========================================="
echo "Starting ReasonIR-8B inference (Optimized)"
echo "Query file: $QUERY_FILE"
echo "Corpus file: $CORPUS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Query batch size: $BATCH_SIZE"
echo "Document batch size: $DOC_BATCH_SIZE"
echo "Caching: $([ "$USE_CACHE" = "0" ] && echo "disabled" || echo "enabled")"
echo "FAISS: $([ "$USE_FAISS" = "0" ] && echo "disabled (direct computation)" || echo "enabled")"
echo "Max length: $MAX_LENGTH"
if [ "$QUICK_RUN" = "1" ]; then
    echo "QUICK RUN MODE: Enabled (${QUICK_DOCS} docs, ${QUICK_QUERIES} queries)"
fi
if [ "$AUTO_BATCH" = "1" ]; then
    echo "AUTO BATCH TUNING: Enabled"
fi
if [ "$USE_MULTIGPU" = "1" ]; then
    echo "MULTI-GPU MODE: Enabled (using all available GPUs)"
fi
echo "INSPECT_BATCHES: ${INSPECT_BATCHES:-not set}"
echo "=========================================="

# Build command with optional arguments
CMD="python code/baselines/reasonir/reasonir-8b.py \
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
[ "$USE_FAISS" = "0" ] && CMD="$CMD --no-faiss"
[ "$QUICK_RUN" = "1" ] && CMD="$CMD --quick-run --quick-docs \"$QUICK_DOCS\" --quick-queries \"$QUICK_QUERIES\""
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
