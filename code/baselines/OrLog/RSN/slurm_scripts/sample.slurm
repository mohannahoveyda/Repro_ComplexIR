#!/usr/bin/env bash
#
# ─── SLURM DIRECTIVES ────────────────────────────────────────────────────────
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl
#SBATCH --output=%x/logs/slurm-%A_%a.out     # %x expands to the job name (see below)
#SBATCH --error=%x/logs/slurm-%A_%a.err

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
MODEL_TAG="Meta-Llama-3-8B-Instruct-${TIMESTAMP}"

# you only ever need to change RETRIEVER to BM25 or E5
RETRIEVER="BM25"

# how many examples per SLURM task
CHUNK_SIZE=10

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT=/home/mhoveyda1/REASON
RUNDIR=$ROOT/runs/$MODEL_TAG
LOGDIR=$RUNDIR/logs
OUTROOT=$RUNDIR/results

mkdir -p $LOGDIR $OUTROOT

# ─── DETERMINE HOW MANY TASKS WE NEED ─────────────────────────────────────────
# count the total number of lines (examples) in the selected JSONL
DATA_PATH=/home/mhoveyda1/REASON/predictions/BM25/test_top20_sample0_2025-07-07_14-43_Filtered_Augmented_with_wikidata_and_wikipedia_metadata.jsonl
TOTAL_SAMPLES=$(wc -l < "$DATA_PATH")
NUM_CHUNKS=$(( (TOTAL_SAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# now re‐launch this script *with* a properly computed array
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Submitting array of 0..$((NUM_CHUNKS-1))"
  sbatch --job-name="$MODEL_TAG" \
         --array=0-$((NUM_CHUNKS-1)) \
         $0
  exit 0
fi

# ─── CHUNK INDICES ────────────────────────────────────────────────────────────
TASK_ID=$SLURM_ARRAY_TASK_ID
START=$(( TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

OUTDIR=$OUTROOT/chunk_${TASK_ID}
mkdir -p "$OUTDIR"

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────
source /home/mhoveyda1/anaconda3/etc/profile.d/conda.sh
conda activate eap-ig
cd $ROOT

# ─── RUN ─────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting chunk $TASK_ID: examples $START .. $END of $TOTAL_SAMPLES"

nvidia-smi

python src/main.py \
  --retriever "$RETRIEVER" \
  --model-name "$MODEL_NAME" \
  --quantize 8bit \
  --method   tf \
  --start    $START \
  --end      $END \
  --outdir   "$OUTDIR" \
  --log_dir  "$LOGDIR"

# ─── MERGE (only on last task) ────────────────────────────────────────────────
if [ "$TASK_ID" -eq $((NUM_CHUNKS-1)) ]; then
  echo "[$(date)] All chunks done; merging details…"
  python src/merge_details.py "$RUNDIR"
  echo "[$(date)] merge complete."
fi

echo "[$(date)] Chunk $TASK_ID finished."