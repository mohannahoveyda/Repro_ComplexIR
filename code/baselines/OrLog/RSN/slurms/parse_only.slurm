#!/usr/bin/env bash
#
# ─── SLURM DIRECTIVES ────────────────────────────────────────────────────────
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl
#SBATCH --job-name=parse_only
#SBATCH --output=%x/logs/parse_only_%j.out   # %x → job name
#SBATCH --error=%x/logs/parse_only_%j.err

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
# choose BM25 or E5
RETRIEVER=${RETRIEVER:-BM25}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT=/home/mhoveyda1/REASON
LOGDIR=$ROOT/runs/parse_only_${RETRIEVER}/logs
mkdir -p "$LOGDIR"
cd "$ROOT"

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────
source /home/mhoveyda1/anaconda3/etc/profile.d/conda.sh
conda activate eap-ig

# ─── RUN PARSE-ONLY ───────────────────────────────────────────────────────────
echo "[$(date)] Parsing all queries for retriever $RETRIEVER …"
python src/main.py \
    --retriever  "$RETRIEVER" \
    --model-name "$MODEL_NAME" \
    --parse-only \
    --log_dir    "$LOGDIR"

echo "[$(date)] Done parsing; cache populated. Exiting."