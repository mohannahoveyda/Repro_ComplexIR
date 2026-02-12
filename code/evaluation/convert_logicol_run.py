#!/usr/bin/env python
"""
Convert LogiCoL run file format to the format expected by evaluate_jsonl.py.

LogiCoL format (input):
  {"query_id": 0, "nl_query": "...", "documents": [{"doc_id": "quest_123", "score": 0.8}, ...]}

evaluate_jsonl format (output):
  {"id": "0", "docs": ["quest_123", ...], "scores": [0.8, ...]}
"""

import argparse
import json
import sys


def convert_logicol_to_eval_jsonl(input_path: str, output_path: str):
    """Convert LogiCoL run file to evaluate_jsonl format."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                # Use nl_query as id to match qrels (query order may differ between run and quest file)
                qid = rec.get("nl_query", rec.get("query_id", ""))
                documents = rec.get("documents", [])
                docs = [d["doc_id"] for d in documents]
                scores = [d["score"] for d in documents]
                out = {"id": str(qid), "docs": docs, "scores": scores}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[convert] Warning: Skipping line: {e}", file=sys.stderr)
                continue
    print(f"[convert] Converted {count} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LogiCoL run file to evaluate_jsonl format"
    )
    parser.add_argument("--input", "-i", required=True, help="LogiCoL run file (JSONL)")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: input with _eval.jsonl suffix)")
    args = parser.parse_args()

    output = args.output or args.input.replace(".jsonl", "_eval.jsonl").replace(".json", "_eval.jsonl")
    if output == args.input:
        output = args.input + "_eval.jsonl"

    convert_logicol_to_eval_jsonl(args.input, output)


if __name__ == "__main__":
    main()
