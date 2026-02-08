#!/bin/bash
#SBATCH --job-name=contriever
#SBATCH --output=outputs/logs/contriever-%j.out
#SBATCH --error=outputs/logs/contriever-%j.err
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

for CONDA_ROOT in "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        break
    fi
done
CONDA_ENV_NAME="${CONDA_ENV:-reasonir}"
conda activate "$CONDA_ENV_NAME" || { echo "ERROR: conda activate $CONDA_ENV_NAME failed."; exit 1; }

echo "GPU Information:"
nvidia-smi
echo "Python: $(which python) $(python --version)"
python -c 'import torch; print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())' || exit 1

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/SIGIR_2026/SIGIR26_Repro_ComplexIR}"
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

QUERY_FILE="${QUERY_FILE:-./data/QUEST/test_id_added.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-./data/QUEST/documents.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/runs/contriever/results_$(date +%Y%m%d_%H%M%S).jsonl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-}"
MAX_LENGTH="${MAX_LENGTH:-512}"
USE_CACHE="${USE_CACHE:-1}"
CACHE_DIR="${CACHE_DIR:-}"
USE_FAISS="${USE_FAISS:-1}"
MAX_DOCS="${MAX_DOCS:-}"
MAX_QUERIES="${MAX_QUERIES:-}"
TASK_SUFFIX="${TASK_SUFFIX:-}"
INDEX_NAME="${INDEX_NAME:-}"
QUERY_FORMAT="${QUERY_FORMAT:-}"
CORPUS_FORMAT="${CORPUS_FORMAT:-}"
AUTO_BATCH="${AUTO_BATCH:-0}"
USE_MULTIGPU="${USE_MULTIGPU:-0}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p outputs/logs

echo "=========================================="
echo "Starting Contriever inference"
echo "Query file: $QUERY_FILE"
echo "Corpus file: $CORPUS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Caching: $([ "$USE_CACHE" = "0" ] && echo "disabled" || echo "enabled")"
echo "FAISS: $([ "$USE_FAISS" = "0" ] && echo "disabled" || echo "enabled")"
if [ -n "$MAX_DOCS" ] || [ -n "$MAX_QUERIES" ]; then
    echo "SUBSET: max_docs=${MAX_DOCS:-ALL}, max_queries=${MAX_QUERIES:-ALL}"
fi
[ -n "$CACHE_DIR" ] && echo "Cache dir: $CACHE_DIR"
[ -n "$TASK_SUFFIX" ] && echo "Task suffix: $TASK_SUFFIX"
[ -n "$INDEX_NAME" ] && echo "Index name: $INDEX_NAME"
[ -n "$QUERY_FORMAT" ] && echo "Query format: $QUERY_FORMAT"
[ -n "$CORPUS_FORMAT" ] && echo "Corpus format: $CORPUS_FORMAT"
[ "$AUTO_BATCH" = "1" ] && echo "AUTO BATCH: Enabled"
[ "$USE_MULTIGPU" = "1" ] && echo "MULTI-GPU: Enabled"
echo "=========================================="

CMD="python code/baselines/contriever/contriever.py \
    --query-file \"$QUERY_FILE\" \
    --corpus-file \"$CORPUS_FILE\" \
    --output-file \"$OUTPUT_FILE\" \
    --device cuda \
    --batch-size \"$BATCH_SIZE\" \
    --max-length \"$MAX_LENGTH\""

[ -n "$QUEST_PLUS" ] && [ "$QUEST_PLUS" = "1" ] && CMD="$CMD --quest-plus"
[ -n "$MODEL_NAME" ] && CMD="$CMD --model-name \"$MODEL_NAME\""
[ -n "$INDEX_NAME" ] && CMD="$CMD --index-name \"$INDEX_NAME\""
[ -n "$DOC_BATCH_SIZE" ] && CMD="$CMD --doc-batch-size \"$DOC_BATCH_SIZE\""
[ "$USE_CACHE" = "0" ] && CMD="$CMD --no-cache"
[ -n "$CACHE_DIR" ] && CMD="$CMD --cache-dir \"$CACHE_DIR\""
[ "$USE_FAISS" = "0" ] && CMD="$CMD --no-faiss"
[ -n "$MAX_DOCS" ] && CMD="$CMD --max-docs \"$MAX_DOCS\""
[ -n "$MAX_QUERIES" ] && CMD="$CMD --max-queries \"$MAX_QUERIES\""
[ -n "$TASK_SUFFIX" ] && CMD="$CMD --task-suffix \"$TASK_SUFFIX\""
[ -n "$QUERY_FORMAT" ] && CMD="$CMD --query-format \"$QUERY_FORMAT\""
[ -n "$CORPUS_FORMAT" ] && CMD="$CMD --corpus-format \"$CORPUS_FORMAT\""
[ "$AUTO_BATCH" = "1" ] && CMD="$CMD --auto-batch"
[ "$USE_MULTIGPU" = "1" ] && CMD="$CMD --multigpu"

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
