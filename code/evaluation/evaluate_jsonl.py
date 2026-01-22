#!/usr/bin/env python
"""
Evaluate retrieval results directly from JSONL format (matching QUEST evaluation approach).

This script evaluates directly from JSONL files without converting to TREC format,
which may be closer to how QUEST evaluates results.

Reads:
    - Run file (JSONL): {"id": "qid", "docs": [...], "scores": [...]}
    - Qrels file (JSONL): {"id": "qid", "docs": [...]}

Computes:
    - Recall@k for each cutoff k
    - NDCG@k for each cutoff k
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


def load_qrels_jsonl(qrels_path: str) -> Dict[str, Set[str]]:
    """
    Load ground truth from JSONL file.
    
    Format: {"id": "qid", "docs": ["doc1", "doc2", ...], ...}
    
    Returns:
        Dict mapping query_id (str) to set of relevant document IDs
    """
    qrels = {}
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                qid = str(data.get("id", ""))
                docs = data.get("docs", [])
                
                if qid:
                    # Convert all docs to strings and create a set
                    qrels[qid] = {str(doc) for doc in docs if doc}
            except json.JSONDecodeError:
                continue
    
    return qrels


def load_run_jsonl(run_path: str) -> Dict[str, List[tuple]]:
    """
    Load run file from JSONL format.
    
    Format: {"id": "qid", "docs": [...], "scores": [...], ...}
    
    Returns:
        Dict mapping query_id to list of (doc_id, rank, score) tuples
        sorted by rank (ascending)
    """
    runs = {}
    
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                qid = str(data.get("id", ""))
                docs = data.get("docs", [])
                scores = data.get("scores", [])
                
                if not qid or len(docs) != len(scores):
                    continue
                
                # Create list of (doc, score) tuples
                doc_score_pairs = [(str(doc), float(score)) for doc, score in zip(docs, scores) if doc]
                
                # Sort by score
                # Check if all scores are negative
                all_negative = all(score < 0 for _, score in doc_score_pairs)
                
                if all_negative:
                    # For negative scores: sort ascending (most negative = rank 1)
                    doc_score_pairs.sort(key=lambda x: x[1], reverse=False)
                else:
                    # For positive scores: sort descending (highest = rank 1)
                    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Assign ranks and create tuples
                ranked_results = [(doc, rank, score) for rank, (doc, score) in enumerate(doc_score_pairs, start=1)]
                
                runs[qid] = ranked_results
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                continue
    
    return runs


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
    
    print(f"[evaluation] Evaluating {len(query_ids)} queries")
    print(f"[evaluation] Qrels has {len(qrels)} queries, runs has {len(runs)} queries")
    
    # Initialize metrics
    recall_sums = {k: 0.0 for k in cutoffs}
    ndcg_sums = {k: 0.0 for k in cutoffs}
    
    # Track document matching issues
    doc_match_issues = defaultdict(int)
    
    # Evaluate each query
    for qid in query_ids:
        relevant = qrels[qid]
        retrieved_items = runs[qid]
        retrieved_docs = [doc_id for doc_id, _, _ in retrieved_items]
        
        # Check document matching
        matched_docs = [doc for doc in retrieved_docs if doc in relevant]
        if len(matched_docs) == 0 and len(relevant) > 0:
            # Try case-insensitive matching
            relevant_lower = {doc.lower() for doc in relevant}
            retrieved_lower = [doc.lower() for doc in retrieved_docs]
            matched_lower = [doc for doc in retrieved_lower if doc in relevant_lower]
            if len(matched_lower) > 0:
                doc_match_issues["case_mismatch"] += 1
        
        for k in cutoffs:
            recall_sums[k] += recall_at_k(retrieved_docs, relevant, k)
            ndcg_sums[k] += ndcg_at_k(retrieved_docs, relevant, k)
    
    if doc_match_issues:
        print(f"[evaluation] Document matching issues: {dict(doc_match_issues)}")
    
    # Average over queries
    num_queries = len(query_ids)
    results = {
        "recall": {k: recall_sums[k] / num_queries for k in cutoffs},
        "ndcg": {k: ndcg_sums[k] / num_queries for k in cutoffs}
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results directly from JSONL format (QUEST-style)"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to JSONL run file",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to JSONL qrels file",
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
    qrels = load_qrels_jsonl(args.qrels)
    print(f"[evaluation] Loaded {len(qrels)} queries with ground truth")
    
    print(f"[evaluation] Loading run file from: {args.run}")
    runs = load_run_jsonl(args.run)
    print(f"[evaluation] Loaded {len(runs)} queries in run file")
    
    print(f"[evaluation] Evaluating with cutoffs: {args.cutoffs}")
    results = evaluate(qrels, runs, args.cutoffs)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (Direct JSONL Evaluation)")
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
