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
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

# Default quest_plus corpus path (relative to repo root), used when --dataset quest_plus
_QUEST_PLUS_CORPUS_REL = os.path.join(
    "data", "QUEST_w_Variants", "data", "quest_text_w_id_withVarients.jsonl"
)

# Default LIMIT / LIMIT+ data paths (relative to repo root)
_LIMIT_DATA_REL = os.path.join(
    "code", "data_generation_utils", "limit_plus", "limit_data",
)


def _repo_root() -> str:
    """Return absolute path to the repo root (two levels up from this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))  # code/evaluation -> repo root


def _default_quest_plus_corpus_path() -> str:
    """Return absolute path to quest_plus corpus from this script's repo root."""
    return os.path.join(_repo_root(), _QUEST_PLUS_CORPUS_REL)


def _default_limit_qrels_path() -> str:
    """Return absolute path to the LIMIT BEIR-style qrels.jsonl."""
    return os.path.join(_repo_root(), _LIMIT_DATA_REL, "qrels.jsonl")


def _default_limit_queries_path() -> str:
    """Return absolute path to the LIMIT test queries file."""
    return os.path.join(_repo_root(), _LIMIT_DATA_REL, "queries.jsonl")


def _default_limit_plus_queries_path() -> str:
    """Return absolute path to the LIMIT+ test queries file."""
    return os.path.join(_repo_root(), _LIMIT_DATA_REL, "limit_quest_queries.jsonl")


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
            
            # Ensure query ID is a string (consistent with JSONL evaluation)
            qid = str(parts[0])
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
            # Ensure docid is a string (consistent with JSONL evaluation)
            docid = str(" ".join(parts[2:-1]))
            
            # Only include documents with positive relevance
            if relevance > 0:
                qrels[qid].add(docid)
    
    return dict(qrels)


def load_qrels_jsonl(qrels_path: str) -> Dict[str, Set[str]]:
    """
    Load ground truth from a BEIR-style JSONL qrels file.

    Format per line: ``{"query-id": "q1", "corpus-id": "d1", "score": 1}``

    Returns:
        Dict mapping query_id (str) to set of relevant document IDs
    """
    qrels: Dict[str, Set[str]] = defaultdict(set)

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(data.get("query-id", ""))
            docid = str(data.get("corpus-id", ""))
            score = data.get("score", 0)
            if qid and docid and int(score) > 0:
                qrels[qid].add(docid)

    return dict(qrels)


def load_qrels_from_test_file(test_file_path: str) -> Dict[str, Set[str]]:
    """
    Build ground truth from a test queries file with inline relevance.

    Supports:
    - LIMIT+ format: ``{"id": 0, "query": "...", "docs": ["doc1", ...]}``
    - Any JSONL with ``_id`` or ``id`` and a ``docs`` list.

    Returns:
        Dict mapping query_id (str) to set of relevant document IDs
    """
    qrels: Dict[str, Set[str]] = defaultdict(set)

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = data.get("_id")
            if qid is None:
                qid = data.get("id")
            if qid is None:
                continue
            qid = str(qid)
            docs = data.get("docs", [])
            for doc in docs:
                qrels[qid].add(str(doc))

    return dict(qrels)


def load_qrels_auto(qrels_path: str) -> Dict[str, Set[str]]:
    """
    Auto-detect qrels format and load.

    - If the file looks like JSONL (first non-empty line is valid JSON),
      load as BEIR JSONL.
    - Otherwise fall back to TREC whitespace format.
    """
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                # First non-empty line is JSON -> JSONL format
                return load_qrels_jsonl(qrels_path)
            except (json.JSONDecodeError, ValueError):
                break
    # Fall back to TREC format
    return load_qrels(qrels_path)


def load_title_to_docid(corpus_path: str) -> Dict[str, str]:
    """
    Load corpus JSONL and build a mapping from document title to document ID (idx).
    Used for quest_plus when the run file contains titles but qrels contain doc IDs.

    Returns:
        Dict mapping title (str) to doc_id (str, e.g. quest_0, quest_94648)
    """
    title_to_id = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = obj.get("title")
            doc_id = obj.get("idx") or obj.get("id")
            if title is not None and doc_id is not None:
                title_to_id[str(title).strip()] = str(doc_id)
    return title_to_id


def _remap_run_docids(
    runs: Dict[str, List[tuple]],
    title_to_docid: Dict[str, str],
) -> None:
    """In-place: replace doc_id (title) with corpus doc_id in runs."""
    for qid in runs:
        new_list = []
        for doc_id, rank, score in runs[qid]:
            mapped_id = title_to_docid.get(doc_id, doc_id)
            new_list.append((mapped_id, rank, score))
        runs[qid] = new_list


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
            
            # Ensure query ID is a string (consistent with JSONL evaluation)
            qid = str(parts[0])
            # Second field should be "Q0"
            if parts[1] != "Q0":
                continue
            
            # Parse from the END of the line (more reliable when docids contain numbers)
            # Format: qid Q0 docid rank score run_name
            # - run_name is the last field
            # - score is the second-to-last field (must be a float)
            # - rank is the third-to-last field (must be an integer)
            # - docid is everything between parts[2] and the rank field
            
            if len(parts) < 6:
                continue
            
            # Try to parse score (second-to-last field)
            try:
                score = float(parts[-2])
            except (ValueError, IndexError):
                continue
            
            # Try to parse rank (third-to-last field)
            try:
                rank = int(parts[-3])
            except (ValueError, IndexError):
                continue
            
            # Docid is everything between parts[2] and parts[-3] (before rank)
            # Ensure docid is a string (consistent with JSONL evaluation)
            doc_id = str(" ".join(parts[2:-3]))
            
            runs[qid].append((doc_id, rank, score))
    
    # Sort by rank for each query (trust the ranks in the TREC file)
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
    cutoffs: List[int],
    query_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate retrieval results.

    Args:
        qrels: Ground truth (query_id -> set of relevant docs)
        runs: Retrieval results (query_id -> list of (doc_id, rank, score))
        cutoffs: List of cutoff values (k)
        query_ids: Optional list of query IDs to evaluate on (must be subset of
            overlapping qrels and runs). If None, use all overlapping queries.

    Returns:
        Dict with keys 'recall' and 'ndcg', each mapping to
        Dict[cutoff -> average metric value]
    """
    # Get all query IDs that appear in both qrels and runs
    overlapping = set(qrels.keys()) & set(runs.keys())
    if query_ids is not None:
        query_ids = [q for q in query_ids if q in overlapping]
    else:
        query_ids = sorted(overlapping)

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
        default=None,
        help="Path to TREC run file (required unless --compare is used with two runs)",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help=(
            "Path to qrels file.  Accepts TREC format (qid 0 docid relevance) "
            "or BEIR JSONL format (auto-detected).  Optional for --dataset "
            "limit / limit_plus (auto-resolved from repo paths)."
        ),
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[5, 20, 100],
        help="List of cutoff values (k) for evaluation",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only on the first N queries (sorted by query ID). E.g. 100 for first 100 queries.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("RUN1", "RUN2"),
        default=None,
        help="Compare two runs: provide two TREC run paths. Ignores --run. Use with --name1 and --name2 for labels.",
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="Run 1",
        help="Label for first run in compare mode (used with --compare).",
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="Run 2",
        help="Label for second run in compare mode (used with --compare).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a LaTeX table row (full line with 12 columns).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Model",
        help="Model name for the LaTeX row (used with --latex).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["quest", "quest_plus", "limit", "limit_plus"],
        default="quest",
        help=(
            "Dataset being evaluated.  Controls LaTeX column placement and "
            "auto-paths.  For 'limit' the BEIR JSONL qrels are loaded "
            "automatically; for 'limit_plus' ground truth is read from the "
            "test queries file."
        ),
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help=(
            "Path to a test queries JSONL file with inline relevance "
            "(e.g. LIMIT+ limit_quest_queries.jsonl).  When provided, "
            "qrels are built from the 'docs' field instead of --qrels."
        ),
    )
    parser.add_argument(
        "--latex-order",
        type=str,
        choices=["recall_first", "ndcg_first"],
        default="recall_first",
        help="LaTeX column order: recall_first (R@5,20,100, N@5,20,100) or ndcg_first (N@5,20,100, R@5,20,100). Cutoffs always 5,20,100.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to quest_plus corpus JSONL (title, text, idx). When --dataset quest_plus, this is auto-set to the repo quest_plus corpus unless overridden.",
    )
    args = parser.parse_args()

    # Auto-select quest_plus corpus when dataset is quest_plus and --corpus not given
    if args.dataset == "quest_plus" and args.corpus is None:
        args.corpus = _default_quest_plus_corpus_path()

    # ---- Auto-resolve paths for LIMIT / LIMIT+ datasets ----
    if args.dataset == "limit":
        if args.qrels is None:
            args.qrels = _default_limit_qrels_path()
    elif args.dataset == "limit_plus":
        if args.test_file is None:
            args.test_file = _default_limit_plus_queries_path()

    # Ensure we have at least one source of ground truth
    if args.qrels is None and args.test_file is None:
        parser.error(
            "Either --qrels or --test-file is required (auto-resolved for "
            "--dataset limit / limit_plus)."
        )

    # Resolve run(s): either single --run or --compare RUN1 RUN2
    if args.compare is not None:
        run_path_1, run_path_2 = args.compare
        run_path = run_path_1  # for qrels/loading logic we load both
        compare_mode = True
    else:
        if args.run is None:
            parser.error("Either --run or --compare RUN1 RUN2 is required")
        run_path = args.run
        run_path_1 = run_path_2 = None
        compare_mode = False

    # ---- Load ground truth ----
    if args.test_file:
        # Build qrels from test file with inline docs (LIMIT+ or similar)
        print(f"[evaluation] Building ground truth from test file: {args.test_file}")
        qrels = load_qrels_from_test_file(args.test_file)
    else:
        print(f"[evaluation] Loading ground truth from: {args.qrels}")
        qrels = load_qrels_auto(args.qrels)
    print(f"[evaluation] Loaded {len(qrels)} queries with ground truth")

    # Title -> doc_id mapping for quest_plus (run file has titles, qrels have doc IDs)
    title_to_docid = None
    if args.corpus:
        if os.path.isfile(args.corpus):
            print(f"[evaluation] Loading quest_plus corpus for title->docid mapping: {args.corpus}")
            title_to_docid = load_title_to_docid(args.corpus)
            print(f"[evaluation] Loaded {len(title_to_docid)} title->docid mappings")
        else:
            print(f"[evaluation] WARNING: quest_plus corpus not found at {args.corpus}; run doc_ids will not be mapped (scores may be 0)")

    # Build query subset: overlapping queries, optionally limited to first N (sorted)
    if compare_mode:
        print(f"[evaluation] Loading run 1: {args.compare[0]}")
        runs1 = load_trec_run(args.compare[0])
        print(f"[evaluation] Loading run 2: {args.compare[1]}")
        runs2 = load_trec_run(args.compare[1])
        if title_to_docid:
            _remap_run_docids(runs1, title_to_docid)
            _remap_run_docids(runs2, title_to_docid)
        overlapping = sorted(set(qrels.keys()) & set(runs1.keys()) & set(runs2.keys()))
    else:
        print(f"[evaluation] Loading run file from: {args.run}")
        runs = load_trec_run(args.run)
        if title_to_docid:
            _remap_run_docids(runs, title_to_docid)
        print(f"[evaluation] Loaded {len(runs)} queries in run file")
        overlapping = sorted(set(qrels.keys()) & set(runs.keys()))
    if args.max_queries is not None:
        query_subset = overlapping[: args.max_queries]
        print(f"[evaluation] Limiting to first {args.max_queries} queries (sorted by qid): {len(query_subset)} queries")
    else:
        query_subset = overlapping
    if not query_subset:
        raise ValueError("No overlapping query IDs between ground truth and run file(s)")

    if compare_mode:
        # Evaluate both runs on query_subset (runs1, runs2 already loaded)
        print(f"[evaluation] Evaluating with cutoffs: {args.cutoffs} (compare mode, {len(query_subset)} queries)")
        results1 = evaluate(qrels, runs1, args.cutoffs, query_ids=query_subset)
        results2 = evaluate(qrels, runs2, args.cutoffs, query_ids=query_subset)

        # Print comparison table
        print("\n" + "=" * 80)
        print(f"Comparison (first {len(query_subset)} queries)")
        print("=" * 80)
        w = 12
        print(f"\n{'Cutoff':<8} {'Recall@k':<{w}} {'NDCG@k':<{w}}  |  {'Recall@k':<{w}} {'NDCG@k':<{w}}")
        print(f"{args.name1:<{w*2+4}}  |  {args.name2}")
        print("-" * 80)
        for k in args.cutoffs:
            r1, n1 = results1["recall"][k], results1["ndcg"][k]
            r2, n2 = results2["recall"][k], results2["ndcg"][k]
            print(f"{k:<8} {r1:<{w}.4f} {n1:<{w}.4f}  |  {r2:<{w}.4f} {n2:<{w}.4f}")
        print("=" * 80)

        if args.output:
            output_data = {
                "num_queries": len(query_subset),
                "max_queries": args.max_queries,
                "cutoffs": args.cutoffs,
                args.name1: {"recall": results1["recall"], "ndcg": results1["ndcg"]},
                args.name2: {"recall": results2["recall"], "ndcg": results2["ndcg"]},
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"\n[evaluation] Comparison results saved to: {args.output}")
        return

    # Single-run evaluation (runs already loaded above)
    print(f"[evaluation] Evaluating with cutoffs: {args.cutoffs} ({len(query_subset)} queries)")
    results = evaluate(qrels, runs, args.cutoffs, query_ids=query_subset)

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

    if args.output:
        output_data = {
            "cutoffs": args.cutoffs,
            "recall": results["recall"],
            "ndcg": results["ndcg"],
            "num_queries": len(query_subset),
            "max_queries": args.max_queries,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[evaluation] Results saved to: {args.output}")

    # LaTeX table row: 12 data columns split into two 6-column blocks.
    #   Block order per dataset pair:
    #     first half  = quest  or limit
    #     second half = quest_plus or limit_plus
    #   Within each block: nDCG@5,20,100  then  Recall@5,20,100
    # Uses \num{} from siunitx for number formatting.
    if args.latex:
        def fmt(v: float) -> str:
            return f"\\num{{{v:.4f}}}"
        place = "\\num{0.0}"
        cutoffs = args.cutoffs
        r_vals = [fmt(results["recall"][k]) for k in cutoffs]
        n_vals = [fmt(results["ndcg"][k]) for k in cutoffs]
        data_block = " & ".join(r_vals + n_vals)
        placeholder_block = " & ".join([place] * 6)
        # quest / limit → first half;  quest_plus / limit_plus → second half
        if args.dataset in ("quest", "limit"):
            line = f"{args.model_name} & &\n{data_block} &\n{placeholder_block} \\\\\n"
        else:
            line = f"{args.model_name} & &\n{placeholder_block} &\n{data_block} \\\\\n"
        print("\n[LaTeX row]\n" + line)


if __name__ == "__main__":
    main()

