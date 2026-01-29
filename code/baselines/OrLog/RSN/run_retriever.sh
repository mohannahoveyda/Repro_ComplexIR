#!/bin/bash

if [ -z "$MODEL" ] || [ -z "$EXAMPLES" ] || [ -z "$DOCS" ] || [ -z "$SAMPLE" ] || [ -z "$TOPK" ]; then
  echo "Error: Please set all required environment variables: MODEL, EXAMPLES, DOCS, SAMPLE, and TOPK."
  exit 1
fi

CURRENT_DATETIME=$(date "+%Y-%m-%d_%H-%M")
OUTPUT="./predictions/$MODEL/test_top${TOPK}_sample${SAMPLE}_${CURRENT_DATETIME}.jsonl"
LOGFILE="./LOGS/retrieval/${MODEL}_run_top${TOPK}_sample${SAMPLE}_${CURRENT_DATETIME}.log"
mkdir -p $(dirname "$OUTPUT")
mkdir -p $(dirname "$LOGFILE")

# Check which model to use for retrieval
case "$MODEL" in
  BM25)
    echo "Running BM25…"
    python -m language.quest.bm25.run_bm25_retriever \
      --examples "$EXAMPLES" \
      --docs     "$DOCS"      \
      --output   "$OUTPUT"    \
      --sample   "$SAMPLE"    \
      --topk     "$TOPK"     >"$LOGFILE" 2>&1
    ;;
  E5)
    echo "Running Dense E5 retriever…"
    INDEX="language/quest/e5/quest_e5_base_v2.index" 
    python -m language.quest.e5.run_dense_retriever \
      --examples "$EXAMPLES" \
      --index    "$INDEX"    \
      --docs     "$DOCS"     \
      --output   "$OUTPUT"   \
      --sample   "$SAMPLE"   \
      --topk     "$TOPK"    >"$LOGFILE" 2>&1
    ;;
  *)
    echo "Unknown model: $MODEL"
    exit 1
    ;;
esac