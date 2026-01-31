#!/bin/bash
#SBATCH --job-name=gritlm-7b
#SBATCH --output=outputs/logs/gritlm-7b-%j.out
#SBATCH --error=outputs/logs/gritlm-7b-%j.err
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "nvidia-smi detected: $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Conda: try common locations so it works on login and compute nodes
for CONDA_ROOT in "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        break
    fi
done
CONDA_ENV_NAME="${CONDA_ENV:-reasonir}"
conda activate "$CONDA_ENV_NAME" || { echo "ERROR: conda activate $CONDA_ENV_NAME failed. List envs: conda info --envs"; exit 1; }

echo "GPU Information:"
nvidia-smi
echo "Python: $(which python) $(python --version)"
python -c 'import torch; print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())' || { echo "ERROR: No module 'torch' in env $CONDA_ENV_NAME. Install with: pip install torch"; exit 1; }

# Project root (change if your repo is elsewhere on Snellius)
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR}"
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

QUERY_FILE="${QUERY_FILE:-./data/QUEST/test_id_added.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-./data/QUEST/documents.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/runs/gritlm-7b/results_$(date +%Y%m%d_%H%M%S).jsonl}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_DOCS="${MAX_DOCS:-}"
MAX_QUERIES="${MAX_QUERIES:-}"
CACHE_DIR="${CACHE_DIR:-}"
TASK_SUFFIX="${TASK_SUFFIX:-}"
INDEX_NAME="${INDEX_NAME:-}"
QUERY_FORMAT="${QUERY_FORMAT:-}"
CORPUS_FORMAT="${CORPUS_FORMAT:-}"
# QUEST withVariants: set these to use distinct cache/output (no overwrite of previous QUEST):
#   QUERY_FILE=./data/QUEST_w_Variants/data/quest_test_withVarients_converted.jsonl
#   CORPUS_FILE=./data/QUEST_w_Variants/data/quest_text_w_id_withVarients.jsonl
#   OUTPUT_FILE=./outputs/runs/gritlm-7b/results_withVariants_$(date +%Y%m%d_%H%M%S).jsonl
#   CACHE_DIR=./outputs/runs/gritlm-7b/cache_withVariants
#   INDEX_NAME=./outputs/runs/gritlm-7b/GritLM-7B-QUEST_withVariants
#   TASK_SUFFIX=_withVariants
#   QUERY_FORMAT=quest
#   CORPUS_FORMAT=quest_plus

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p outputs/logs

echo "=========================================="
echo "Starting GritLM-7B inference (gritlm-7b_main.py)"
echo "Query file: $QUERY_FILE"
echo "Corpus file: $CORPUS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Batch size: $BATCH_SIZE"
if [ -n "$MAX_DOCS" ] || [ -n "$MAX_QUERIES" ]; then
    echo "SUBSET: max_docs=${MAX_DOCS:-ALL}, max_queries=${MAX_QUERIES:-ALL}"
fi
[ -n "$CACHE_DIR" ] && echo "Cache dir: $CACHE_DIR"
[ -n "$TASK_SUFFIX" ] && echo "Task suffix: $TASK_SUFFIX"
[ -n "$INDEX_NAME" ] && echo "Index name: $INDEX_NAME"
[ -n "$QUERY_FORMAT" ] && echo "Query format: $QUERY_FORMAT"
[ -n "$CORPUS_FORMAT" ] && echo "Corpus format: $CORPUS_FORMAT"
echo "=========================================="

CMD="python code/baselines/gritlm/gritlm-7b_main.py \
    --query-file \"$QUERY_FILE\" \
    --corpus-file \"$CORPUS_FILE\" \
    --output-file \"$OUTPUT_FILE\" \
    --device cuda \
    --batch-size \"$BATCH_SIZE\""

[ -n "$QUEST_PLUS" ] && [ "$QUEST_PLUS" = "1" ] && CMD="$CMD --quest-plus"
[ -n "$MODEL_NAME" ] && CMD="$CMD --model-name \"$MODEL_NAME\""
[ -n "$INDEX_NAME" ] && CMD="$CMD --index-name \"$INDEX_NAME\""
[ -n "$MAX_DOCS" ] && CMD="$CMD --max-docs \"$MAX_DOCS\""
[ -n "$MAX_QUERIES" ] && CMD="$CMD --max-queries \"$MAX_QUERIES\""
[ -n "$CACHE_DIR" ] && CMD="$CMD --cache-dir \"$CACHE_DIR\""
[ -n "$TASK_SUFFIX" ] && CMD="$CMD --task-suffix \"$TASK_SUFFIX\""
[ -n "$QUERY_FORMAT" ] && CMD="$CMD --query-format \"$QUERY_FORMAT\""
[ -n "$CORPUS_FORMAT" ] && CMD="$CMD --corpus-format \"$CORPUS_FORMAT\""
[ -n "$NO_CACHE" ] && [ "$NO_CACHE" = "1" ] && CMD="$CMD --no-cache"
[ -n "$MULTIGPU" ] && [ "$MULTIGPU" = "1" ] && CMD="$CMD --multigpu"
[ -n "$NO_FAISS" ] && [ "$NO_FAISS" = "1" ] && CMD="$CMD --no-faiss"
[ -n "$AUTO_BATCH" ] && [ "$AUTO_BATCH" = "1" ] && CMD="$CMD --auto-batch"

# Execute command
eval $CMD
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
