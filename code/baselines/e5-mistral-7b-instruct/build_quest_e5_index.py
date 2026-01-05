#!/usr/bin/env python
"""
Build an E5-Mistral index for QUEST documents.

Reads:
    /home/mhoveyda1/SIGIR_2026/SIGIR26_Repro_ComplexIR/data/QUEST/documents.jsonl

Each line:
    {
        "title": str,
        "text": str,
        ...
    }

Uses:
    doc_id   = title
    doc_text = f"{title}. {text}"

Writes to OUTPUT_DIR:
    doc_ids.txt          (one doc_id per line, aligned with embeddings)
    doc_embeddings.npy   (float32, [num_docs, dim])
    faiss_index_ip.bin   (FAISS IndexFlatIP over normalized embeddings)
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
CORPUS_PATH = "data/QUEST/documents.jsonl"
OUTPUT_DIR = "outputs/quest_e5_mistral_index"


def load_quest_corpus(path: str, max_docs: int = None) -> Tuple[List[str], List[str]]:
    """
    Load QUEST documents.jsonl and return doc_ids and texts.

    doc_id   := title
    doc_text := f"{title}. {text}"
    """
    doc_ids: List[str] = []
    texts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            title = rec["title"]
            text = rec["text"]
            doc_ids.append(str(title))
            # Title + text, similar to E5 BEIR script
            doc_text = f"{title}. {text}".strip()
            texts.append(doc_text)

            if max_docs is not None and len(doc_ids) >= max_docs:
                break

    return doc_ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS_PATH,
        help="Path to QUEST documents.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to store embeddings and FAISS index",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for E5 encoding",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional limit on number of documents (for debugging)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SentenceTransformer (e.g., 'cuda' or 'cpu')",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[build] Loading QUEST corpus from: {args.corpus}")
    doc_ids, texts = load_quest_corpus(args.corpus, max_docs=args.max_docs)
    print(f"[build] Loaded {len(doc_ids)} documents.")

    print("[build] Loading E5-Mistral-7B-Instruct model...")
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct", device=args.device)
    # As recommended for this model
    model.max_seq_length = 4096

    print("[build] Encoding documents with E5 (this may take a while)...")
    doc_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-compatible
    )

    doc_embeddings = doc_embeddings.astype("float32")
    dim = doc_embeddings.shape[1]
    print(f"[build] Embeddings shape: {doc_embeddings.shape} (dim={dim})")

    # Save doc IDs
    doc_ids_path = os.path.join(args.output_dir, "doc_ids.txt")
    with open(doc_ids_path, "w", encoding="utf-8") as f:
        for did in doc_ids:
            f.write(did + "\n")
    print(f"[build] Wrote doc IDs to {doc_ids_path}")

    # Save embeddings
    emb_path = os.path.join(args.output_dir, "doc_embeddings.npy")
    np.save(emb_path, doc_embeddings)
    print(f"[build] Wrote embeddings to {emb_path}")

    # Build FAISS index (inner product on normalized embeddings = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    print(f"[build] FAISS index built with {index.ntotal} vectors.")

    faiss_path = os.path.join(args.output_dir, "faiss_index_ip.bin")
    faiss.write_index(index, faiss_path)
    print(f"[build] Wrote FAISS index to {faiss_path}")

    print("[build] Done.")


if __name__ == "__main__":
    main()
