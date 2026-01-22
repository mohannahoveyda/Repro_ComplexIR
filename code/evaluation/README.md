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

**Options:**
- `--input`: Path to input run file (required)
- `--output`: Path to output TREC run file (default: input path with `_trec` suffix)
- `--run_name`: Run name to use if not found in input file (default: "run")

**Examples:**
```bash
# Convert JSONL to TREC format
python code/evaluation/preprocess_run.py --input outputs/runs/bm25/test_top100_sample0_2026-01-22_16-45.jsonl --run_name bm25
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

**Options:**
- `--run`: Path to TREC run file (required)
- `--qrels`: Path to TREC qrels file (required)
- `--cutoffs`: List of cutoff values (k) for evaluation (default: `1 5 10 20 50 100`)
- `--output`: Optional path to save results as JSON

**Examples:**
```bash
# Save results to JSON 
python code/evaluation/evaluate.py \
  --run outputs/runs/bm25/test_top100_sample0_2026-01-22_16-45_trec \
  --qrels data/QUEST/test_id_added_qrels \
  --output outputs/runs/bm25_test/evaluation_results.json
```

**Output:**
The script prints a table with Recall@k and NDCG@k for each cutoff:
```
============================================================
Evaluation Results
============================================================

Cutoff     Recall@k        NDCG@k         
------------------------------------------------------------
1          0.1234          0.1234         
5          0.2345          0.2345         
10         0.3456          0.3456         
...
============================================================
```

If `--output` is specified, results are also saved as JSON:
```json
{
  "cutoffs": [1, 5, 10, 20, 50],
  "recall": {
    "1": 0.1234,
    "5": 0.2345,
    ...
  },
  "ndcg": {
    "1": 0.1234,
    "5": 0.2345,
    ...
  },
  "num_queries": 1727
}
```

---

## Complete Workflow Example

Here's a complete example of evaluating retrieval results:

```bash
# Step 1: Preprocess ground truth (if not already done)
python code/evaluation/preprocess_qrels.py \
  --input data/QUEST/test.jsonl \
  --output data/QUEST/test_id_added_qrels

# Step 2: Preprocess retrieval results
python code/evaluation/preprocess_run.py \
  --input outputs/runs/bm25_test/test_top50_sample0_2025-01-03_17-44.jsonl \
  --run_name bm25_temptest

# Step 3: Evaluate
python code/evaluation/evaluate.py \
  --run outputs/runs/bm25_test/test_top50_sample0_2025-01-03_17-44_trec \
  --qrels data/QUEST/test_id_added_qrels \
  --cutoffs 1 5 10 20 50 \
  --output outputs/runs/bm25_test/evaluation_results.json
```

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
- **Missing Queries**: If a query appears in the run file but not in qrels, it is skipped. If a query appears in qrels but not in run, it is also skipped.

---

## Troubleshooting

**Issue**: "No overlapping query IDs between ground truth and run file"
- **Solution**: Check that query IDs in both files match exactly (same format, no extra whitespace)

**Issue**: "Skipping malformed line" warnings in preprocess_run.py
- **Solution**: Check that your input file format matches one of the supported formats. For JSONL, ensure each line is valid JSON with `id`, `docs`, and `scores` fields.

**Issue**: Low recall/NDCG scores
- **Solution**: Verify that document IDs in your run file match exactly with document IDs in the qrels file (case-sensitive, including spaces and special characters)
