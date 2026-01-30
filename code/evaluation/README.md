# Evaluation Pipeline

This directory contains scripts for preprocessing and evaluating retrieval results using standard TREC formats and metrics.

## Overview

The evaluation pipeline consists of three main steps:

1. **Preprocess QRELS**: Convert ground truth data (JSONL) to TREC qrels format (written based on QUEST data)
2. **Preprocess RUN**: Convert retrieval results to TREC run format 
3. **Evaluate**: Compute Recall@k and NDCG@k metrics (requires TREC formatted test and gold files)

## Scripts

### 1. `preprocess_qrels.py`

Converts QUEST test data (JSONL format) to TREC qrels (ground truth) format.

**Input Format:**
- JSONL file with each line containing:
  ```json
  {
    "id": "query_id",
    "docs": ["doc1", "doc2", ...],
    ...
  }
  ```
- If `id` field is missing, the script will automatically create an ID-augmented version.

**Output Format:**
- TREC qrels format: `qid 0 docid relevance`
  - `qid`: Query ID
  - `0`: Unused field (TREC standard)
  - `docid`: Relevant document title
  - `relevance`: Always 1 (all docs in gold file are relevant)

**Usage:**
```bash
python code/evaluation/preprocess_qrels.py \
  --input data/QUEST/test.jsonl \
  --output data/QUEST/test_qrels
```


### 2. `preprocess_run.py`

Converts retrieval results to TREC run format. Supports multiple input formats including JSONL and various TREC-like formats.

**Input Formats Supported:**
1. **JSONL format:**
   ```json
   {
     "id": "query_id",
     "docs": ["doc1", "doc2", ...],
     "scores": [score1, score2, ...],
     ...
   }
   ```

2. **TREC format:** `qid Q0 docid rank score run_name`
3. **Simplified format:** `qid docid score`
4. **Tab-separated format:** `qid\tdocid\tscore`

**Output Format:**
- TREC run format: `qid Q0 docid rank score run_name`
  - `qid`: Query ID
  - `Q0`: Standard TREC field
  - `docid`: Document ID (may contain spaces)
  - `rank`: Sequential rank starting from 1
  - `score`: Retrieval score
  - `run_name`: Run identifier

**Sorting Behavior:**
- **Negative scores**: Sorted in ascending order (most negative = rank 1) - as done in QUEST's implementation for BM25
- **Positive scores**: Sorted in descending order (highest = rank 1)

**Usage:**
```bash
python code/evaluation/preprocess_run.py \
  --input outputs/runs/my_run.jsonl \
  --run_name my_run_name
```

---

### 3. `evaluate.py`

Evaluates retrieval results using Recall@k and NDCG@k metrics.

**Input Requirements:**
- TREC run file (from `preprocess_run.py`)
- TREC qrels file (from `preprocess_qrels.py`)

**Metrics Computed:**
- **Recall@k**: Fraction of relevant documents retrieved in top-k results
- **NDCG@k**: Normalized Discounted Cumulative Gain at cutoff k

**Usage:**
```bash
python code/evaluation/evaluate.py \
  --run outputs/runs/my_run_trec \
  --qrels data/QUEST/test_id_added_qrels
```

**First N queries only:** Use `--max-queries N` to evaluate only on the first N queries (query IDs are sorted for a deterministic order):
```bash
python code/evaluation/evaluate.py \
  --run outputs/runs/my_run_trec \
  --qrels data/QUEST/test_id_added_qrels \
  --max-queries 100
```

**Compare two runs:** Use `--compare RUN1 RUN2` to evaluate and compare two model runs side-by-side. Combine with `--max-queries` to compare on the first N queries only. Optional `--name1` and `--name2` set labels in the output.
```bash
python code/evaluation/evaluate.py \
  --qrels data/QUEST/test_id_added_qrels \
  --compare outputs/runs/model_a_trec outputs/runs/model_b_trec \
  --name1 "Model A" --name2 "Model B" \
  --max-queries 100
```

**Output:**
The script prints a table with Recall@k and NDCG@k for each cutoff. In compare mode, both runs are printed side-by-side.

If `--output` is specified, results are also saved as JSON:


---

## Complete Workflow Example

To evaluate a model's performance; 1) Preprocess ground truth (if not already done), 2) Preprocess retrieval results, 3) Evaluate



---

## File Format Details

### TREC QRELS Format
```
qid 0 docid relevance
```
- Fields are space-separated
- Document IDs may contain spaces (everything between field 2 and the last field)
- Only documents with `relevance > 0` are considered relevant

### TREC RUN Format
```
qid Q0 docid rank score run_name
```
- Fields are space-separated
- Document IDs may contain spaces (everything between "Q0" and the rank field)
- Results are sorted by score (descending for positive scores, ascending for negative scores)
- Ranks are assigned sequentially starting from 1

---

## Notes

- **Query ID Matching**: The evaluation script only evaluates queries that appear in both the run file and qrels file. Query IDs must match exactly.
- **Negative Scores**: When all scores for a query are negative, they are sorted in ascending order (most negative = best rank).
- **Document ID Matching**: Document IDs are matched exactly as strings, so ensure consistent formatting between run and qrels files.
- **Missing Queries**: If a query appears in the run file but not in qrels, it is skipped and the other way around.

