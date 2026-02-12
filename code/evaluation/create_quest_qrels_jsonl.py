#!/usr/bin/env python
"""
Create JSONL qrels from quest_test_withVarients.jsonl for evaluate_jsonl.py.

quest_test_withVarients format:
  {"queries": [...], "operators": [], "documents": ["quest_123", ...], "nl_query": "...", ...}

Qrels format (output for evaluate_jsonl):
  {"id": "0", "docs": ["quest_123", ...]}

Uses 0-based line index as query ID to match LogiCoL run file (query_id 0, 1, 2, ...).
"""

import argparse
import json
import sys


def create_qrels(input_path: str, output_path: str, use_nl_query: bool = True):
    """Create JSONL qrels from quest test file.
    
    Uses nl_query as id when use_nl_query=True (to match LogiCoL run files which
    may have different query order). Use index when use_nl_query=False.
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                docs = rec.get("documents", rec.get("docs", []))
                qid = str(rec.get("nl_query", idx)) if use_nl_query else str(idx)
                out = {"id": qid, "docs": [str(d) for d in docs if d]}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[create_qrels] Warning: Skipping line {idx}: {e}", file=sys.stderr)
                continue
    print(f"[create_qrels] Created qrels for {count} queries -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL qrels from quest_test_withVarients.jsonl"
    )
    parser.add_argument("--input", "-i", required=True, help="Quest test file (JSONL)")
    parser.add_argument("--output", "-o", default=None, help="Output qrels path (default: input dir + quest_test_withVarients_qrels.jsonl)")
    args = parser.parse_args()

    import os
    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.dirname(args.input)
        output = os.path.join(out_dir, f"{base}_qrels.jsonl")

    create_qrels(args.input, output)


if __name__ == "__main__":
    main()
