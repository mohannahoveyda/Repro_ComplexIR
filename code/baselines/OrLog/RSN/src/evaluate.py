
import os
from typing import List, Set
import math

def precision_at_k(preds: List[str], gold: Set[str], k: int) -> float:
    topk = preds[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in gold) / k

def recall_at_k(preds: List[str], gold: Set[str], k: int) -> float:
    topk = preds[:k]
    if not gold:
        return 0.0
    return sum(1 for d in topk if d in gold) / len(gold)

def f1_at_k(preds: List[str], gold: Set[str], k: int) -> float:
    p = precision_at_k(preds, gold, k)
    r = recall_at_k(preds, gold, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def reciprocal_rank(preds: List[str], gold: Set[str]) -> float:
    for idx, d in enumerate(preds, start=1):
        if d in gold:
            return 1.0 / idx
    return 0.0

# NDCG 
def dcg_at_k(preds: List[str], gold: Set[str], k: int) -> float:
    """
    Discounted Cumulative Gain at k, with binary relevance.
    """
    dcg = 0.0
    for i, d in enumerate(preds[:k], start=1):
        rel = 1 if d in gold else 0
        dcg += rel / math.log2(i + 1)
    return dcg

def idcg_at_k(gold: Set[str], k: int) -> float:
    """
    Ideal DCG: assume the top |gold| positions are all relevant,
    up to position k.
    """
    ideal_rels = min(len(gold), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rels + 1))

def ndcg_at_k(preds: List[str], gold: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    """
    idcg = idcg_at_k(gold, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(preds, gold, k) / idcg

