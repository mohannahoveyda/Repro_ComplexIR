#!/usr/bin/env python
"""
Evaluate retrieval results using Recall@k and NDCG@k metrics.

Reads:
    - TREC run file: qid Q0 docid rank score run_name
    - TREC qrels file: qid 0 docid relevance

Computes:
    - Recall@k for each cutoff k
    - NDCG@k for each cutoff k
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


def load_qrels(qrels_path: str) -> Dict[str, Set[str]]:
    """
    Load ground truth from TREC qrels file.
    
    Format: qid 0 docid relevance
    
    Note: docid may contain spaces, so we parse carefully.
    We look for: qid, 0, then everything until the last field (relevance).
    
    Returns:
        Dict mapping query_id (str) to set of relevant document IDs
    """
    qrels = defaultdict(set)
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            line = line.strip()
            parts = line.split()
            
            if len(parts) < 4:
                continue
            
            qid = parts[0]
            # Second field should be "0"
            if parts[1] != "0":
                continue
            
            # Relevance is the last field
            try:
                relevance = int(parts[-1])
            except ValueError:
                continue
            
            # Docid is everything between parts[2] and parts[-1]
            # Join with spaces to handle docids with spaces
            docid = " ".join(parts[2:-1])
            
            # Only include documents with positive relevance
            if relevance > 0:
                qrels[qid].add(docid)
    
    return dict(qrels)


def load_trec_run(run_path: str) -> Dict[str, List[tuple]]:
    """
    Load TREC run file.
    
    Format: qid Q0 docid rank score run_name
    
    Note: docid and run_name may contain spaces, so we parse carefully.
    We look for: qid, Q0, then docid (until we find rank), rank, score, run_name.
    
    Returns:
        Dict mapping query_id to list of (doc_id, rank, score) tuples
        sorted by rank (ascending)
    """
    runs = defaultdict(list)
    
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            line = line.strip()
            parts = line.split()
            
            if len(parts) < 6:
                continue
            
            qid = parts[0]
            # Second field should be "Q0"
            if parts[1] != "Q0":
                continue
            
            # Rank is the first integer after Q0
            # Score is the first float after rank
            # Find rank and score positions
            rank_idx = None
            score_idx = None
            
            for i in range(2, len(parts)):
                # Try to parse as integer (rank)
                if rank_idx is None:
                    try:
                        int(parts[i])
                        rank_idx = i
                        continue
                    except ValueError:
                        pass
                
                # Try to parse as float (score)
                if rank_idx is not None and score_idx is None:
                    try:
                        float(parts[i])
                        score_idx = i
                        break
                    except ValueError:
                        pass
            
            if rank_idx is None or score_idx is None:
                continue
            
            # Docid is everything between parts[2] and parts[rank_idx]
            doc_id = " ".join(parts[2:rank_idx])
            rank = int(parts[rank_idx])
            score = float(parts[score_idx])
            
            runs[qid].append((doc_id, rank, score))
    
    # Sort by rank for each query
    for qid in runs:
        runs[qid].sort(key=lambda x: x[1])
    
    return dict(runs)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved: List of retrieved document IDs (up to k)
        relevant: Set of relevant document IDs
        k: Cutoff value
    
    Returns:
        Recall@k value
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    num_relevant_retrieved = len([doc for doc in retrieved_k if doc in relevant])
    
    return num_relevant_retrieved / len(relevant)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute DCG@k.
    
    Args:
        relevance_scores: List of relevance scores (1 for relevant, 0 for not)
        k: Cutoff value
    
    Returns:
        DCG@k value
    """
    relevance_scores = relevance_scores[:k]
    if len(relevance_scores) == 0:
        return 0.0
    
    # DCG = sum(rel_i / log2(i + 1)) for i in [1, k]
    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        dcg += rel / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute NDCG@k.
    
    Args:
        retrieved: List of retrieved document IDs (up to k)
        relevant: Set of relevant document IDs
        k: Cutoff value
    
    Returns:
        NDCG@k value
    """
    # Get relevance scores for retrieved documents
    relevance_scores = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    
    # Compute DCG@k
    dcg = dcg_at_k(relevance_scores, k)
    
    # Compute IDCG@k (ideal DCG)
    num_relevant = len(relevant)
    if num_relevant == 0:
        return 0.0
    
    # Ideal: all relevant docs at the top
    ideal_scores = [1.0] * min(num_relevant, k)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def evaluate(
    qrels: Dict[str, Set[str]],
    runs: Dict[str, List[tuple]],
    cutoffs: List[int]
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate retrieval results.
    
    Args:
        qrels: Ground truth (query_id -> set of relevant docs)
        runs: Retrieval results (query_id -> list of (doc_id, rank, score))
        cutoffs: List of cutoff values (k)
    
    Returns:
        Dict with keys 'recall' and 'ndcg', each mapping to
        Dict[cutoff -> average metric value]
    """
    # Get all query IDs that appear in both qrels and runs
    query_ids = set(qrels.keys()) & set(runs.keys())
    
    if len(query_ids) == 0:
        raise ValueError("No overlapping query IDs between ground truth and run file")
    
    # Initialize metrics
    recall_sums = {k: 0.0 for k in cutoffs}
    ndcg_sums = {k: 0.0 for k in cutoffs}
    
    # Evaluate each query
    for qid in query_ids:
        relevant = qrels[qid]
        retrieved_items = runs[qid]
        retrieved_docs = [doc_id for doc_id, _, _ in retrieved_items]
        
        for k in cutoffs:
            recall_sums[k] += recall_at_k(retrieved_docs, relevant, k)
            ndcg_sums[k] += ndcg_at_k(retrieved_docs, relevant, k)
    
    # Average over queries
    num_queries = len(query_ids)
    results = {
        "recall": {k: recall_sums[k] / num_queries for k in cutoffs},
        "ndcg": {k: ndcg_sums[k] / num_queries for k in cutoffs}
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results using Recall@k and NDCG@k"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to TREC run file",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to TREC qrels file (qid 0 docid relevance)",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50, 100],
        help="List of cutoff values (k) for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()
    
    print(f"[evaluation] Loading ground truth from: {args.qrels}")
    qrels = load_qrels(args.qrels)
    print(f"[evaluation] Loaded {len(qrels)} queries with ground truth")
    
    print(f"[evaluation] Loading run file from: {args.run}")
    runs = load_trec_run(args.run)
    print(f"[evaluation] Loaded {len(runs)} queries in run file")
    
    print(f"[evaluation] Evaluating with cutoffs: {args.cutoffs}")
    results = evaluate(qrels, runs, args.cutoffs)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\n{'Cutoff':<10} {'Recall@k':<15} {'NDCG@k':<15}")
    print("-" * 60)
    
    for k in args.cutoffs:
        recall = results["recall"][k]
        ndcg = results["ndcg"][k]
        print(f"{k:<10} {recall:<15.4f} {ndcg:<15.4f}")
    
    print("=" * 60)
    
    # Save results if output path is provided
    if args.output:
        output_data = {
            "cutoffs": args.cutoffs,
            "recall": results["recall"],
            "ndcg": results["ndcg"],
            "num_queries": len(set(qrels.keys()) & set(runs.keys()))
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[evaluation] Results saved to: {args.output}")


if __name__ == "__main__":
    main()

