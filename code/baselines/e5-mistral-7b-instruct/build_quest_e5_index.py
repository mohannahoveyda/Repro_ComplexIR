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
    checkpoint_*.npy     (intermediate checkpoint files for resuming)
"""

import argparse
import json
import os
import glob
from typing import List, Tuple, Optional

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


def find_latest_checkpoint(output_dir: str) -> Optional[Tuple[int, np.ndarray]]:
    """Find the latest checkpoint and return (start_idx, partial_embeddings)."""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*.npy")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    
    # Extract indices and find latest
    checkpoint_indices = []
    for cp_path in checkpoints:
        try:
            idx = int(os.path.basename(cp_path).replace("checkpoint_", "").replace(".npy", ""))
            checkpoint_indices.append((idx, cp_path))
        except ValueError:
            continue
    
    if not checkpoint_indices:
        return None
    
    # Find the checkpoint with highest index
    latest_idx, latest_path = max(checkpoint_indices, key=lambda x: x[0])
    print(f"[build] Found checkpoint at index {latest_idx}: {latest_path}")
    partial_embeddings = np.load(latest_path)
    return latest_idx + 1, partial_embeddings  # Return next index to start from


def encode_with_checkpoints(
    model: SentenceTransformer,
    texts: List[str],
    doc_ids: List[str],
    output_dir: str,
    batch_size: int,
    checkpoint_interval: int = 1000,
) -> np.ndarray:
    """
    Encode texts with periodic checkpointing.
    
    Args:
        checkpoint_interval: Save checkpoint every N batches
    """
    total_docs = len(texts)
    
    # Check for existing checkpoint
    resume_info = find_latest_checkpoint(output_dir)
    if resume_info is not None:
        start_idx, partial_embeddings = resume_info
        print(f"[build] Resuming from checkpoint at document {start_idx}/{total_docs}")
        # Get embedding dimension from partial embeddings
        dim = partial_embeddings.shape[1]
        # Initialize full embeddings array
        doc_embeddings = np.zeros((total_docs, dim), dtype=np.float32)
        doc_embeddings[:start_idx] = partial_embeddings
    else:
        start_idx = 0
        doc_embeddings = None
    
    # Process in batches with checkpointing
    num_batches = (total_docs - start_idx + batch_size - 1) // batch_size
    
    for batch_num in tqdm(range(num_batches), desc="Encoding batches"):
        batch_start = start_idx + batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_docs)
        batch_texts = texts[batch_start:batch_end]
        
        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Initialize embeddings array on first batch if starting fresh
        if doc_embeddings is None:
            dim = batch_embeddings.shape[1]
            doc_embeddings = np.zeros((total_docs, dim), dtype=np.float32)
        
        # Store batch embeddings
        doc_embeddings[batch_start:batch_end] = batch_embeddings
        
        # Save checkpoint periodically
        if (batch_num + 1) % checkpoint_interval == 0 or batch_end == total_docs:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{batch_end - 1}.npy")
            np.save(checkpoint_path, doc_embeddings[:batch_end])
            print(f"[build] Saved checkpoint at document {batch_end}/{total_docs}")
    
    return doc_embeddings


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
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N batches (default: 1000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[build] Loading QUEST corpus from: {args.corpus}")
    doc_ids, texts = load_quest_corpus(args.corpus, max_docs=args.max_docs)
    print(f"[build] Loaded {len(doc_ids)} documents.")

    print("[build] Loading E5-Mistral-7B-Instruct model...")
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct", device=args.device)
    # As recommended for this model
    model.max_seq_length = 512

    print("[build] Encoding documents with E5 (this may take a while)...")
    if args.resume:
        doc_embeddings = encode_with_checkpoints(
            model, texts, doc_ids, args.output_dir, args.batch_size, args.checkpoint_interval
        )
    else:
        # Check if checkpoint exists and warn user
        resume_info = find_latest_checkpoint(args.output_dir)
        if resume_info is not None:
            print(f"[build] WARNING: Checkpoint found but --resume not specified.")
            print(f"[build] Use --resume to continue from checkpoint, or delete checkpoints to start fresh.")
            response = input("Continue from checkpoint? (y/n): ")
            if response.lower() == 'y':
                doc_embeddings = encode_with_checkpoints(
                    model, texts, doc_ids, args.output_dir, args.batch_size, args.checkpoint_interval
                )
            else:
                # Remove checkpoints and start fresh
                checkpoint_pattern = os.path.join(args.output_dir, "checkpoint_*.npy")
                for cp in glob.glob(checkpoint_pattern):
                    os.remove(cp)
                print("[build] Starting fresh encoding...")
                doc_embeddings = encode_with_checkpoints(
                    model, texts, doc_ids, args.output_dir, args.batch_size, args.checkpoint_interval
                )
        else:
            doc_embeddings = encode_with_checkpoints(
                model, texts, doc_ids, args.output_dir, args.batch_size, args.checkpoint_interval
            )

    dim = doc_embeddings.shape[1]
    print(f"[build] Embeddings shape: {doc_embeddings.shape} (dim={dim})")

    # Save doc IDs
    doc_ids_path = os.path.join(args.output_dir, "doc_ids.txt")
    with open(doc_ids_path, "w", encoding="utf-8") as f:
        for did in doc_ids:
            f.write(did + "\n")
    print(f"[build] Wrote doc IDs to {doc_ids_path}")

    # Save final embeddings
    emb_path = os.path.join(args.output_dir, "doc_embeddings.npy")
    np.save(emb_path, doc_embeddings)
    print(f"[build] Wrote embeddings to {emb_path}")

    # Clean up checkpoint files
    checkpoint_pattern = os.path.join(args.output_dir, "checkpoint_*.npy")
    for cp in glob.glob(checkpoint_pattern):
        os.remove(cp)
    print("[build] Cleaned up checkpoint files")

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
