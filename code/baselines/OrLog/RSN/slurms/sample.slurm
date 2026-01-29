#!/usr/bin/env bash
#
# ─── SLURM DIRECTIVES ────────────────────────────────────────────────────────
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl
#SBATCH --output=/home/mhoveyda1/REASON/LOGS/slurm_logs/slurm-%j.log
#SBATCH --error=/home/mhoveyda1/REASON/LOGS/slurm_logs/slurm-%j.log

# ─── USAGE & REQUIRED ARGS ───────────────────────────────────────────────────
if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch $0 <RETRIEVER> <MODEL_NAME>"
  echo "  RETRIEVER   one of BM25 or E5"
  echo "  MODEL_NAME  e.g. meta-llama/Meta-Llama-3-8B-Instruct"
  exit 1
fi

RETRIEVER="$1"
MODEL_NAME="$2"
MODEL_TAG="${MODEL_NAME##*/}"    # strip path → e.g. Meta-Llama-3-8B-Instruct
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TOTAL_SAMPLES=10
CHUNK_SIZE=10

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT=/home/mhoveyda1/REASON
JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
RUNDIR=$ROOT/runs/${RETRIEVER}/${MODEL_TAG}_${JOB_ID}_${TIMESTAMP}
LOGDIR=$RUNDIR/logs
OUTDIR=$RUNDIR/results 

mkdir -p "$LOGDIR" "$OUTDIR" 

# ─── CHUNK INDICES ────────────────────────────────────────────────────────────
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

# ─── ENVIRONMENT ─────────────────────────────────────────────────────────────
source /home/mhoveyda1/anaconda3/etc/profile.d/conda.sh
conda activate eap-ig
cd "$ROOT"

# ─── RUN ─────────────────────────────────────────────────────────────────────
echo "[$(date)] Chunk $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_MAX for $RETRIEVER on $MODEL_NAME"
nvidia-smi

python src/main.py \
    --retriever   "$RETRIEVER" \
    --model-name  "$MODEL_NAME" \
    --start       "$START" \
    --end         "$END" \
    --outdir      "$OUTDIR/chunk_${SLURM_ARRAY_TASK_ID}" \
    --log_dir     "$LOGDIR" \
    > "$LOGDIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

# ── MERGE (last task) ────────────────────────────────────────────────────────
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
  echo "[$(date)] All chunks done, merging details…"
  python src/merge_details.py "$RUNDIR"
  echo "[$(date)] Merge complete."
fi
