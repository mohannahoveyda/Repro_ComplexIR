#!/bin/bash
# ----------  job resources  ----------
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-17
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl
#SBATCH --output=/home/mhoveyda1/REASON/runs/slurm_logs/slurm-%j.out
#SBATCH --error=/home/mhoveyda1/REASON/runs/slurm_logs/slurm-%j.err

# ----------  variables you edit per model  ----------
TIMESTAMP="${TIMESTAMP:-unknown_time}"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_TAG="Meta-Llama-3-8B-Instruct-${TIMESTAMP}"

TOTAL_SAMPLES=1700
CHUNK_SIZE=100
# NUM_CHUNKS=$(( (TOTAL_SAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
# LAST_IDX=$(( NUM_CHUNKS - 1 ))



# ----------  derived paths  ----------
ROOT=/home/mhoveyda1/REASON
RUNDIR=$ROOT/runs/${MODEL_TAG}
LOGDIR=$RUNDIR/logs
OUTDIR=$RUNDIR/results
RETRIEVER="BM25"

mkdir -p $LOGDIR $OUTDIR

# ----------  chunk indices  ----------
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

# ----------  environment  ----------
source /home/mhoveyda1/anaconda3/etc/profile.d/conda.sh
conda activate eap-ig
cd $ROOT

# ----------  run  ----------
nvidia-smi
python src/main.py \
    -m "$MODEL_NAME" \
    -q 4bit \
    -M tf \
    -n $TOTAL_SAMPLES \
    --start $START --end $END \
    --retriever $RETRIEVER \
    --outdir "$OUTDIR/chunk_${SLURM_ARRAY_TASK_ID}" \
    --log_dir "$LOGDIR" \
    > "$LOGDIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

# ── only the “last” array task does the merge ────────
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
    echo "[$(date)] All chunks done, running merge_details.py…"  
    python /home/mhoveyda1/REASON/src/merge_details.py "$RUNDIR"
    echo "[$(date)] merge_details.py complete."
fi