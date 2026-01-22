#!/usr/bin/env python
"""
Run E5-Mistral retrieval on QUEST test queries.

Reads:
    QUEST test queries from:
        /home/mhoveyda1/SIGIR_2026/SIGIR26_Repro_ComplexIR/data/QUEST/test_id_added.jsonl

Each line:
    {
        "id": int,
        "query": str,
        "docs": [title1, title2, ...],
        ...
    }

Index:
    Assumes build_quest_e5_index.py has produced in INDEX_DIR:
        doc_ids.txt
        doc_embeddings.npy
        faiss_index_ip.bin

Writes:
    TREC-style run file with:
        qid Q0 docid rank score run_name
"""

import argparse
import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Default paths relative to project root
TEST_PATH = "data/QUEST/test_id_added.jsonl"
INDEX_DIR = "outputs/quest_e5_mistral_index"
DEFAULT_RUN = "outputs/runs/e5_mistral_quest.trec"


def load_quest_test(path: str) -> Tuple[List[str], List[str]]:
    """
    Load QUEST test_id_added.jsonl and return query_ids and query_texts.

    qid   := str(rec["id"])
    qtext := rec["query"]
    """
    qids: List[str] = []
    queries: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec["id"])
            qtext = rec["query"]
            qids.append(qid)
            queries.append(qtext.strip())

    return qids, queries


def load_index(index_dir: str):
    # Load doc IDs
    doc_ids_path = os.path.join(index_dir, "doc_ids.txt")
    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = [line.strip() for line in f if line.strip()]

    # Load FAISS index
    faiss_path = os.path.join(index_dir, "faiss_index_ip.bin")
    index = faiss.read_index(faiss_path)

    # Optional sanity check with embeddings
    emb_path = os.path.join(index_dir, "doc_embeddings.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path, mmap_mode="r")
        if emb.shape[0] != len(doc_ids) or emb.shape[1] != index.d:
            raise ValueError(
                f"Mismatch: doc_ids={len(doc_ids)}, embeddings={emb.shape}, index_dim={index.d}"
            )

    return doc_ids, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default=TEST_PATH,
        help="Path to QUEST test_id_added.jsonl",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=INDEX_DIR,
        help="Directory containing doc_ids.txt and faiss_index_ip.bin",
    )
    parser.add_argument(
        "--output_run",
        type=str,
        default=DEFAULT_RUN,
        help="Where to write TREC run file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="How many docs to retrieve per query",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for E5 encoding of queries",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SentenceTransformer (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="e5_mistral_quest",
        help="Run name for last column in TREC file",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_run), exist_ok=True)

    print(f"[retrieval] Loading QUEST test queries from: {args.test}")
    qids, queries = load_quest_test(args.test)
    print(f"[retrieval] Loaded {len(qids)} queries.")

    print(f"[retrieval] Loading index from: {args.index_dir}")
    doc_ids, index = load_index(args.index_dir)
    print(f"[retrieval] Index has {index.ntotal} documents.")

    print("[retrieval] Loading E5-Mistral-7B-Instruct model for queries...")
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct", device=args.device)
    model.max_seq_length = 4096

    print("[retrieval] Encoding queries with E5 (web_search_query prompt)...")
    # E5 Instruct: use prompt_name="web_search_query" for retrieval queries
    query_embs = model.encode(
        queries,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt_name="web_search_query",
    ).astype("float32")

    print(f"[retrieval] Running FAISS search (top_k={args.top_k})...")
    scores, indices = index.search(query_embs, args.top_k)

    print(f"[retrieval] Writing TREC run to: {args.output_run}")
    with open(args.output_run, "w", encoding="utf-8") as out_f:
        for qi, qid in enumerate(tqdm(qids, desc="writing run")):
            row_scores = scores[qi]
            row_indices = indices[qi]

            # FAISS returns in sorted order, but we sort again defensively
            sorted_pairs = sorted(
                zip(row_indices, row_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            for rank, (doc_idx, score) in enumerate(sorted_pairs, start=1):
                if doc_idx < 0:
                    continue
                doc_id = doc_ids[doc_idx]
                out_f.write(
                    f"{qid} Q0 {doc_id} {rank} {float(score):.6f} {args.run_name}\n"
                )

    print("[retrieval] Done.")


if __name__ == "__main__":
    main()
