#!/usr/bin/env bash
set -euo pipefail

# ─── ARGS & USAGE ───────────────────────────────────────────────────────────
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <run_dir> [poll_interval_seconds]"
  exit 1
fi

RUN_DIR="$1"
POLL_INTERVAL="${2:-60}"

# ─── CONFIG ─────────────────────────────────────────────────────────────────
JOB_NAME="test.slurm"                            # SLURM script name to watch
PYTHON_ENV_PROFILE="/home/mhoveyda1/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="eap-ig"
MERGE_SCRIPT="/home/mhoveyda1/REASON/src/merge_details.py"

# ─── PRECHECKS ───────────────────────────────────────────────────────────────
if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: run_dir '$RUN_DIR' does not exist."
  exit 2
fi
if [ ! -f "$MERGE_SCRIPT" ]; then
  echo "ERROR: merge script '$MERGE_SCRIPT' not found."
  exit 3
fi

# ─── POLL LOOP ───────────────────────────────────────────────────────────────
echo "[$(date)] Waiting for all SLURM jobs named '$JOB_NAME' to finish..."
while true; do
  remaining=$( squeue -u "$USER" -h -o "%j" | grep -cw "^${JOB_NAME}$" || true )
  if [ "$remaining" -eq 0 ]; then
    echo "[$(date)] No more '$JOB_NAME' jobs. Proceeding to merge."
    break
  fi
  echo "[$(date)] $remaining job(s) still in queue; sleeping ${POLL_INTERVAL}s..."
  sleep "$POLL_INTERVAL"
done

# ─── RUN MERGE ───────────────────────────────────────────────────────────────
echo "[$(date)] Activating conda env '$CONDA_ENV' and running merge..."
source "$PYTHON_ENV_PROFILE"
conda activate "$CONDA_ENV"
python "$MERGE_SCRIPT" "$RUN_DIR" \
  && echo "[$(date)] Merge completed successfully." \
  || echo "[$(date)] Merge encountered errors."

