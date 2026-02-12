#!/usr/bin/env python
"""
Prepare LogiCoL run and quest test for TREC-format evaluation (evaluate.py).

Converts:
1. LogiCoL run (query_id, documents) -> JSONL with id/docs/scores (id = hash of nl_query)
2. Quest test (documents) -> TREC qrels (qid = hash of nl_query)

Uses MD5 hash of nl_query as qid so run and qrels match (query order may differ).
"""

import argparse
import hashlib
import json
import sys


def _qid_hash(nl_query: str) -> str:
    """Stable hash for nl_query (no spaces, for TREC format)."""
    return hashlib.md5(nl_query.encode("utf-8")).hexdigest()[:16]


def convert_run_to_trec_jsonl(input_path: str, output_path: str):
    """Convert LogiCoL run to id/docs/scores JSONL for preprocess_run."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                nl_query = rec.get("nl_query", "")
                documents = rec.get("documents", [])
                docs = [d["doc_id"] for d in documents]
                scores = [d["score"] for d in documents]
                qid = _qid_hash(nl_query)
                out = {"id": qid, "docs": docs, "scores": scores}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[prepare] Warning: Skipping run line: {e}", file=sys.stderr)
    print(f"[prepare] Converted {count} queries to {output_path}")
    return count


def create_qrels_trec(input_path: str, output_path: str):
    """Create TREC qrels from quest test (id = hash of nl_query)."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                nl_query = rec.get("nl_query", "")
                docs = rec.get("documents", rec.get("docs", []))
                qid = _qid_hash(nl_query)
                for doc in docs:
                    if doc:
                        fout.write(f"{qid} 0 {doc} 1\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[prepare] Warning: Skipping qrels line: {e}", file=sys.stderr)
    print(f"[prepare] Created TREC qrels for {count} queries -> {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LogiCoL run and quest test for TREC evaluation"
    )
    sub = parser.add_subparsers(dest="cmd", help="Command")
    p_run = sub.add_parser("run", help="Convert LogiCoL run to TREC-ready JSONL")
    p_run.add_argument("--input", "-i", required=True, help="LogiCoL run file")
    p_run.add_argument("--output", "-o", required=True, help="Output JSONL (id/docs/scores)")

    p_qrels = sub.add_parser("qrels", help="Create TREC qrels from quest test")
    p_qrels.add_argument("--input", "-i", required=True, help="Quest test JSONL")
    p_qrels.add_argument("--output", "-o", required=True, help="Output TREC qrels file")

    args = parser.parse_args()
    if args.cmd == "run":
        convert_run_to_trec_jsonl(args.input, args.output)
    elif args.cmd == "qrels":
        create_qrels_trec(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
