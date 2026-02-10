import argparse
import json
import os
from evaluate import _default_quest_plus_corpus_path, load_qrels, load_title_to_docid, load_trec_run, _remap_run_docids, recall_at_k, ndcg_at_k
from typing import Dict, List, Optional, Set
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_TEMPLATES = (
    "_",
    "_ or _",
    "_ or _ or _",
    "_ that are also _",
    "_ that are also both _ and _",
    "_ that are also _ but not _",
    "_ that are not _",
)


def to_heatmap_df(results: dict, metric: str = "ndcg", k: int = 100) -> pd.DataFrame:
    # Collect all templates across all models
    templates = sorted({t for m in results for t in results[m].keys()})
    models = sorted(results.keys())

    df = pd.DataFrame(index=models, columns=templates, dtype=float)

    for model in models:
        for templ in templates:
            try:
                df.loc[model, templ] = float(results[model][templ][metric][k])
            except KeyError:
                df.loc[model, templ] = np.nan  # missing value

    return df


def make_figure(df, dataset, metric='ndcg', k=100):
    plt.figure(figsize=(max(10, 0.6 * df.shape[1]), max(3, 0.6 * df.shape[0])))
    im = plt.imshow(df.values, aspect="auto", cmap="Blues")
    im.set_clim(0.0, 1.0)

    cbar = plt.colorbar(im)
    cbar.set_label(f"{metric}@{k}")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(df.shape[0]), df.index)

    plt.xlabel("templates")
    plt.ylabel("models")

    plt.title(f"{metric}@{k} by template (x) and model (y)")
    plt.tight_layout()

    save_path = f"outputs/results/{dataset}_{metric}@{k}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[evaluation] Saved heat map to: {save_path}")


def get_run_path(model, dataset):
    model_dir = Path('outputs/runs/' + model)
    if not model_dir.exists():
        return None

    # Collect candidates
    candidates = []
    for file in model_dir.iterdir():
        if not file.is_file():  # Check if it is a file
            continue
        if file.suffix:  # Only choose the file if it does not have an extension
            continue
        if model.lower() not in file.name.lower():  # Check if the model name is in the file name
            continue
        if '_' + dataset.lower() + '_' in file.name.lower():  # Check if the dataset is in the file name
            candidates.append(file)

    # Select the last run as candidate (or None if there are no candidates)
    if candidates == []:
        return None
    elif len(candidates) > 1:
        re_iteration = re.compile(r"_r(\d{2})")
        runs = [re_iteration.search(x.name).group() if re_iteration.search(x.name) else -1 for x in candidates]
        idx = runs.index(max(runs))
        return candidates[idx]
    else:  # len(candidates) == 1
        return candidates[0]


def select_queries(query_ids, dataset, corpus):
    template_queries = {}
    if dataset == 'quest':
        with open(corpus, "r") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                qid = str(line_num)
                template = data["metadata"]["template"]
                if qid in query_ids:
                    if template in template_queries.keys():
                        template_queries[template].append(qid)
                    else:
                        template_queries[template] = [qid]

    return template_queries


def evaluate(
    qrels: Dict[str, Set[str]],
    runs: Dict[str, List[tuple]],
    dataset: str,
    corpus: str,
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

    template_results = {}
    query_ids_template = select_queries(query_ids, dataset, corpus)
    for template in _TEMPLATES:
        # Initialize metrics
        recall_sums = {k: 0.0 for k in cutoffs}
        ndcg_sums = {k: 0.0 for k in cutoffs}

        # Evaluate each query
        for qid in query_ids_template[template]:
            relevant = qrels[qid]
            retrieved_items = runs[qid]
            retrieved_docs = [doc_id for doc_id, _, _ in retrieved_items]

            for k in cutoffs:
                recall_sums[k] += recall_at_k(retrieved_docs, relevant, k)
                ndcg_sums[k] += ndcg_at_k(retrieved_docs, relevant, k)

        # Average over queries
        num_queries = len(query_ids_template[template])
        template_results[template] = {
            "recall": {k: recall_sums[k] / num_queries for k in cutoffs},
            "ndcg": {k: ndcg_sums[k] / num_queries for k in cutoffs}
        }

    return template_results


def evaluate_model(model, qrels, args):
    run_path = get_run_path(model, args.dataset)
    if run_path is None:
        raise FileNotFoundError(
            f"No run file found for model='{model}' and dataset='{args.dataset}' "
            f"in outputs/runs/{model}/ (expected something containing '{model}', '{args.dataset}' and '_rXX')."
        )

    # Title -> doc_id mapping for quest_plus (run file has titles, qrels have doc IDs)
    title_to_docid = None
    if args.dataset == "quest_plus":
        if os.path.isfile(args.corpus):
            print(f"[evaluation] Loading quest_plus corpus for title->docid mapping: {args.corpus}")
            title_to_docid = load_title_to_docid(args.corpus)
            print(f"[evaluation] Loaded {len(title_to_docid)} title->docid mappings")
        else:
            print(f"[evaluation] WARNING: quest_plus corpus not found at {args.corpus}; run doc_ids will not be mapped (scores may be 0)")

    print(f"[evaluation] Loading run file from: {run_path}")
    runs = load_trec_run(run_path)
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

    # Run evaluation
    print(f"[evaluation] Evaluating with cutoffs: {args.cutoffs} ({len(query_subset)} queries)")
    run_results = evaluate(qrels, runs, args.dataset, args.queries, args.cutoffs, query_ids=query_subset)

    return run_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results using Recall@k and NDCG@k"
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
        "--dataset",
        type=str,
        choices=["quest", "quest_plus", "limit", "limit_plus"],
        default="quest",
        required=True,
        help="Which dataset is used: quest/limit/quest_plus/limit_plus.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to quest_plus corpus JSONL (title, text, idx). When --dataset quest_plus, this is auto-set to the repo quest_plus corpus unless overridden.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        required=True,
        help="Path to the test queries.",
    )
    args = parser.parse_args()

    # Auto-select quest_plus corpus when dataset is quest_plus and --corpus not given
    if args.dataset == "quest_plus" and args.corpus is None:
        args.corpus = _default_quest_plus_corpus_path()

    # Load qrels
    print(f"[evaluation] Loading ground truth from: {args.qrels}")
    qrels = load_qrels(args.qrels)
    print(f"[evaluation] Loaded {len(qrels)} queries with ground truth")

    # Evaluate each model
    results = {}
    model_dir = Path('outputs/runs/')
    for folder in model_dir.iterdir():
        if folder.is_dir():
            model = folder.name
            print(f"[evaluation] Evaluating {model}")
            results[model] = evaluate_model(model, qrels, args)

    # Make heatmap
    for metric in ['ndcg', 'recall']:
        for k in args.cutoffs:
            df = to_heatmap_df(results, metric=metric, k=k)
            make_figure(df, args.dataset, metric=metric, k=k)


if __name__ == "__main__":
    main()
