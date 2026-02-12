#!/bin/bash
# Evaluate a LogiCoL run file using the ComplexIR evaluation code.
#
# Usage: ./eval_logicol_run.sh <LOGICOL_RUN_FILE> [QUEST_TEST_FILE] [MODEL_NAME]
#
# Example:
#   ./eval_logicol_run.sh /path/to/run_quest_LogiCol_xxx.jsonl
#   ./eval_logicol_run.sh /path/to/run_xxx.jsonl "" "LogiCoL e5-base-v2"
#
# Uses evaluate.py (TREC format) and prints LaTeX table row.
# If QUEST_TEST_FILE is not provided, uses LogiCoL's quest_test_withVarients.jsonl.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGICOL_RUN="${1:?Usage: $0 <logicol_run_file> [quest_test_file] [model_name]}"
QUEST_TEST="${2:-}"
MODEL_NAME="${3:-LogiCoL}"

# Resolve paths
if [ ! -f "$LOGICOL_RUN" ]; then
    echo "ERROR: Run file not found: $LOGICOL_RUN"
    exit 1
fi

# LogiCoL root: run file is at output/results/, so go up 2 levels
LOGICOL_DIR="$(cd "$(dirname "$LOGICOL_RUN")/../.." && pwd)"
if [ -z "$QUEST_TEST" ]; then
    # Auto-detect: run file with "_quest_" in name -> quest_test.jsonl, else quest_test_withVarients.jsonl
    RUN_BASENAME="$(basename "$LOGICOL_RUN")"
    if [[ "$RUN_BASENAME" == *"_quest_"* ]]; then
        QUEST_FILE="quest_test.jsonl"
    else
        QUEST_FILE="quest_test_withVarients.jsonl"
    fi
    if [ -f "$LOGICOL_DIR/data/data/$QUEST_FILE" ]; then
        QUEST_TEST="$LOGICOL_DIR/data/data/$QUEST_FILE"
    elif [ -f "$LOGICOL_DIR/data/$QUEST_FILE" ]; then
        QUEST_TEST="$LOGICOL_DIR/data/$QUEST_FILE"
    else
        echo "ERROR: Quest test file not found ($QUEST_FILE). Please provide as second argument."
        exit 1
    fi
fi

if [ ! -f "$QUEST_TEST" ]; then
    echo "ERROR: Quest test file not found: $QUEST_TEST"
    exit 1
fi

# Output paths (in same dir as run file)
RUN_DIR="$(dirname "$LOGICOL_RUN")"
RUN_BASE="$(basename "$LOGICOL_RUN" .jsonl)"
QUEST_BASE="$(basename "$QUEST_TEST" .jsonl)"
RUN_JSONL_TREC="${RUN_DIR}/${RUN_BASE}_trec_input.jsonl"
RUN_TREC="${RUN_DIR}/${RUN_BASE}.trec"
QRELS_TREC="${RUN_DIR}/${QUEST_BASE}_qrels.trec"

echo "=========================================="
echo "Evaluate LogiCoL Run (TREC + LaTeX)"
echo "=========================================="
echo "Run file:       $LOGICOL_RUN"
echo "Quest test:     $QUEST_TEST"
echo "Model name:     $MODEL_NAME"
echo "TREC run:       $RUN_TREC"
echo "Qrels:          $QRELS_TREC"
echo "=========================================="

# Step 1: Create TREC qrels from quest test
echo ""
echo "[1/4] Creating TREC qrels from quest test..."
python3 "$SCRIPT_DIR/prepare_logicol_for_trec.py" qrels --input "$QUEST_TEST" --output "$QRELS_TREC"

# Step 2: Convert LogiCoL run to TREC-ready JSONL (id/docs/scores)
echo ""
echo "[2/4] Converting LogiCoL run to TREC format..."
python3 "$SCRIPT_DIR/prepare_logicol_for_trec.py" run --input "$LOGICOL_RUN" --output "$RUN_JSONL_TREC"

# Step 3: Run preprocess_run to get final TREC run file
echo ""
echo "[3/4] Running preprocess_run..."
python3 "$SCRIPT_DIR/preprocess_run.py" --input "$RUN_JSONL_TREC" --output "$RUN_TREC" --run_name "logicol"

# Step 4: Run evaluate.py with LaTeX output
echo ""
echo "[4/4] Running evaluation (evaluate.py with LaTeX)..."
python3 "$SCRIPT_DIR/evaluate.py" \
    --run "$RUN_TREC" \
    --qrels "$QRELS_TREC" \
    --cutoffs 5 20 100 \
    --dataset quest \
    --model-name "$MODEL_NAME" \
    --latex \
    --output "${RUN_DIR}/${RUN_BASE}_metrics.json"

echo ""
echo "=========================================="
echo "Done. Metrics: ${RUN_DIR}/${RUN_BASE}_metrics.json"
echo "=========================================="
