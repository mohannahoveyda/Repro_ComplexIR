#!/usr/bin/env python3
"""
Generate constraint-type and depth analysis figures for complex information
retrieval across QUEST, QUEST+, and LIMIT+ datasets.

Produces a 3×2 PDF figure:
    Rows:    QUEST | QUEST+ | LIMIT+
    Columns: Constraint-type analysis | Depth analysis

Each subplot shows nDCG@20 (y-axis) for every model (coloured lines)
grouped by constraint template or compositional depth on the x-axis.

Usage:
    python figure_constraint_depth.py [--output figure.pdf] [--metric ndcg@20]
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
#  Paths  (resolved relative to this script's location)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent            # SIGIR26_Repro_ComplexIR
_PROJECT_ROOT = _REPO_ROOT.parent                  # SIGIR_2026
_LOGICOL_ROOT = _PROJECT_ROOT / "LogiCoL"

# ---------------------------------------------------------------------------
#  Template canonical ordering (the 7 QUEST templates)
# ---------------------------------------------------------------------------
TEMPLATE_ORDER = [
    "_",
    "_ or _",
    "_ or _ or _",
    "_ that are also _",
    "_ that are also both _ and _",
    "_ that are not _",
    "_ that are also _ but not _",
]

TEMPLATE_SHORT = {
    "_":                          "Single",
    "_ or _":                     "A ∨ B",
    "_ or _ or _":                "A ∨ B ∨ C",
    "_ that are also _":          "A ∧ B",
    "_ that are also both _ and _": "A ∧ B ∧ C",
    "_ that are not _":           "A ¬ B",
    "_ that are also _ but not _": "A ∧ B ¬ C",
}

# Map QUEST+ operators → canonical template name
_OPS_TO_TEMPLATE = {
    "SINGLE":  "_",
    "OR":      "_ or _",
    "OR_OR":   "_ or _ or _",
    "AND":     "_ that are also _",
    "AND_AND": "_ that are also both _ and _",
    "NOT":     "_ that are not _",
    "AND_NOT": "_ that are also _ but not _",
}

# ---------------------------------------------------------------------------
#  Dataset configuration
# ---------------------------------------------------------------------------
DATASETS: Dict[str, Dict[str, Any]] = {
    "QUEST": {
        "query_file": _REPO_ROOT / "data" / "QUEST" / "test_id_added.jsonl",
        "qrels_file": _REPO_ROOT / "data" / "QUEST" / "test_qrels",  # title-based
        "id_qrels_file": _REPO_ROOT / "data" / "QUEST" / "test_id_added_qrels",
        "corpus_file": _REPO_ROOT / "data" / "QUEST_w_Variants" / "data" / "quest_text_w_id.jsonl",
        "doc_id_field": "title",  # qrels use titles
    },
    "QUEST+": {
        "query_file": _REPO_ROOT / "data" / "QUEST_w_Variants" / "data" / "quest_test_withVarients_converted.jsonl",
        "qrels_file": _REPO_ROOT / "data" / "QUEST_w_Variants" / "data" / "quest_test_withVarients_converted_qrels",
        "corpus_file": _REPO_ROOT / "data" / "QUEST_w_Variants" / "data" / "quest_text_w_id_withVarients.jsonl",
        "doc_id_field": "idx",  # qrels use quest_xxx ids
    },
    "LIMIT+": {
        "query_file": _REPO_ROOT / "code" / "data_generation_utils" / "limit_plus" / "limit_data" / "limit_quest_queries.jsonl",
        "qrels_file": None,  # will be built from query file
        "corpus_file": None,
        "doc_id_field": "name",  # person names
    },
}

# ---------------------------------------------------------------------------
#  Model run-file registry
#  path = relative to _REPO_ROOT unless starts with / or contains "LogiCoL"
#  fmt  = "trec" | "reasonir_jsonl" | "contriever_jsonl" | "logicol_jsonl"
#  doc  = "title" | "idx" | "name"  (what IDs appear in the run file)
# ---------------------------------------------------------------------------
_R = "outputs/runs"

MODELS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "BM25": {
        "QUEST":  {"path": f"{_R}/bm25/test_top100_sample0_2026-01-22_16-45_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/bm25/test_top100_sample0_withVariants_2026-01-29_16-27_trec", "fmt": "trec", "doc": "idx"},
        "LIMIT+": None,
    },
    "Contriever": {
        "QUEST":  {"path": f"{_R}/contriever/results_20260201_011902.jsonl", "fmt": "reasonir_jsonl", "doc": "title"},
        "QUEST+": {"path": f"{_R}/contriever/results_withVariants_20260201_015242.jsonl", "fmt": "reasonir_jsonl", "doc": "title"},
        "LIMIT+": None,
    },
    "Promptriever": {
        "QUEST":  {"path": f"{_R}/promptriever-7b/results_20260201_010134_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/promptriever-7b/results_withVariants_20260201_095533_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "Qwen3-Emb-8B": {
        "QUEST":  {"path": f"{_R}/qwen3-emb-8b/results_20260209_160140_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/qwen3-emb-8b/results_withVariants_20260209_223122_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "E5-Mistral-Inst.": {
        "QUEST":  {"path": f"{_R}/e5-mistral-7b/results_20260208_212949_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/e5-mistral-7b/results_withVariants_20260209_144352_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "E5-base-V2": {
        "QUEST":  {"path": f"{_R}/e5-base-v2/results_20260208_215255_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/e5-base-v2/results_withVariants_20260208_221812_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "GritLM-7B": {
        "QUEST":  {"path": f"{_R}/gritlm-7b/results_20260130_182659_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/gritlm-7b/results_withVariants_20260131_224222_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "GTE-ColBERT": {
        "QUEST":  None,
        "QUEST+": None,
        "LIMIT+": None,
    },
    "SPLADE": {
        "QUEST":  {"path": f"{_R}/splade/SPLADE-v3_QUEST_r01.trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/splade/SPLADE-v3_QUEST-PLUS_r01.trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "Search-R1": {
        "QUEST":  {"path": f"{_R}/search-r1/results_20260210_162319_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/search-r1/results_withVariants_20260210_204004.jsonl", "fmt": "reasonir_jsonl", "doc": "title"},
        "LIMIT+": None,
    },
    "Set-Comp LSR": {
        "QUEST":  {"path": f"{_R}/set-comp/SET-COMPOSITIONAL_QUEST_r01_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/set-comp/SET-COMPOSITIONAL_QUEST-PLUS_r01_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "ReasonIR-8B": {
        "QUEST":  {"path": f"{_R}/reasonir-8b/results_20260130_183756_trec", "fmt": "trec", "doc": "title"},
        "QUEST+": {"path": f"{_R}/reasonir-8b/results_withVariants_20260201_094721_trec", "fmt": "trec", "doc": "title"},
        "LIMIT+": None,
    },
    "LogiCoL": {
        "QUEST":  {
            "path": str(_LOGICOL_ROOT / "output" / "results" /
                        "run_quest_LogiCol_e5-base-v2_lr1e-05_bs4_len512_fp16_quest_20260211_021943.jsonl"),
            "fmt": "logicol_jsonl",
            "doc": "idx",
        },
        "QUEST+": None,
        "LIMIT+": None,
    },
}

# Display-name order for the legend (determines line colour assignment)
MODEL_ORDER = [
    "BM25",
    "Contriever",
    "Promptriever",
    "Qwen3-Emb-8B",
    "E5-Mistral-Inst.",
    "E5-base-V2",
    "GritLM-7B",
    "GTE-ColBERT",
    "SPLADE",
    "Search-R1",
    "Set-Comp LSR",
    "ReasonIR-8B",
    "LogiCoL",
]

# ---------------------------------------------------------------------------
#  Data-loading helpers
# ---------------------------------------------------------------------------

def load_corpus_mappings(corpus_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (title_to_idx, idx_to_title) dicts from a corpus JSONL."""
    t2i: Dict[str, str] = {}
    i2t: Dict[str, str] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            title = str(obj.get("title", "")).strip()
            idx = str(obj.get("idx", "")).strip()
            if title and idx:
                t2i[title] = idx
                i2t[idx] = title
    return t2i, i2t


def load_qrels_trec(path: Path) -> Dict[str, Set[str]]:
    """Load TREC qrels: qid 0 docid relevance  (docid may contain spaces)."""
    qrels: Dict[str, Set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid = str(parts[0])
            rel = int(parts[-1])
            docid = " ".join(parts[2:-1])
            if rel > 0:
                qrels[qid].add(docid)
    return dict(qrels)


def build_limit_plus_qrels(query_path: Path) -> Dict[str, Set[str]]:
    """Build qrels from LIMIT+ query file (docs are person names)."""
    qrels: Dict[str, Set[str]] = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["id"])
            qrels[qid] = set(obj["docs"])
    return qrels


def load_query_metadata(
    dataset_name: str, query_path: Path
) -> Dict[str, Tuple[str, int]]:
    """
    Return {qid: (template, depth)} for every query.
    template is one of TEMPLATE_ORDER; depth is number of atomic predicates.
    """
    meta: Dict[str, Tuple[str, int]] = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["id"])

            if dataset_name == "LIMIT+":
                template = obj["metadata"]["template"]
                depth = len(obj["metadata"]["attrs"])
            elif dataset_name == "QUEST":
                template = obj.get("metadata", {}).get("template", "_")
                depth = template.count("_") if template else 1
            else:  # QUEST+
                ops = obj.get("metadata", {}).get("operators", [])
                op_key = "_".join(sorted(ops)) if ops else "SINGLE"
                template = _OPS_TO_TEMPLATE.get(op_key, "_")
                queries = obj.get("metadata", {}).get("queries", [])
                depth = len(queries) if queries else template.count("_")
                if depth == 0:
                    depth = template.count("_")

            meta[qid] = (template, depth)
    return meta


# ---------------------------------------------------------------------------
#  Run-file loaders  (all return  Dict[qid, List[str]]  = ranked doc-id list)
# ---------------------------------------------------------------------------

def load_trec_run(path: str) -> Dict[str, List[str]]:
    """Load TREC run file → {qid: [docid_rank1, docid_rank2, ...]}."""
    raw: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6 or parts[1] != "Q0":
                continue
            qid = str(parts[0])
            try:
                rank = int(parts[-3])
                score = float(parts[-2])
            except (ValueError, IndexError):
                continue
            docid = " ".join(parts[2:-3])
            raw[qid].append((rank, score, docid))
    result: Dict[str, List[str]] = {}
    for qid in raw:
        raw[qid].sort(key=lambda x: x[0])
        result[qid] = [docid for _, _, docid in raw[qid]]
    return result


def load_reasonir_jsonl(path: str) -> Dict[str, List[str]]:
    """Load ReasonIR / Contriever-style JSONL (line-number = qid)."""
    result: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            retrieved = obj.get("retrieved", [])
            if not isinstance(retrieved, list):
                continue
            qid = str(line_num)
            docs = []
            for item in sorted(retrieved, key=lambda x: x.get("rank", 0)):
                title = item.get("title", "")
                if title:
                    docs.append(str(title))
            if docs:
                result[qid] = docs
    return result


def load_logicol_jsonl(
    path: str,
    query_text_to_qid: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Load LogiCoL internal JSONL → {integer_qid: [docid, ...]}.
    Maps nl_query text → integer qid via *query_text_to_qid*.
    doc_ids are quest_xxx (internal IDs).
    """
    result: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            nl_query = obj.get("nl_query", "").strip()
            documents = obj.get("documents", [])
            # Try exact match
            qid = query_text_to_qid.get(nl_query)
            if qid is None:
                # Try case-insensitive match
                qid = query_text_to_qid.get(nl_query.lower())
            if qid is None:
                continue
            docs_sorted = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            result[qid] = [d["doc_id"] for d in docs_sorted]
    return result


def load_run(
    cfg: Dict[str, Any],
    dataset_name: str,
    query_text_to_qid: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, List[str]]]:
    """Dispatch to the right loader based on *cfg['fmt']*."""
    path = cfg["path"]
    # Resolve path
    if not os.path.isabs(path):
        path = str(_REPO_ROOT / path)
    if not os.path.isfile(path):
        return None

    fmt = cfg["fmt"]
    if fmt == "trec":
        return load_trec_run(path)
    elif fmt == "reasonir_jsonl":
        return load_reasonir_jsonl(path)
    elif fmt == "logicol_jsonl":
        if query_text_to_qid is None:
            print(f"  [WARN] LogiCoL needs query_text_to_qid mapping; skipping.")
            return None
        return load_logicol_jsonl(path, query_text_to_qid)
    else:
        print(f"  [WARN] Unknown run format '{fmt}'; skipping.")
        return None


# ---------------------------------------------------------------------------
#  Metric computation
# ---------------------------------------------------------------------------

def _dcg(rels: List[float], k: int) -> float:
    rels = rels[:k]
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    dcg = _dcg(rels, k)
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def per_query_scores(
    run: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    metric: str = "ndcg@20",
) -> Dict[str, float]:
    """Compute per-query metric. Returns {qid: score}."""
    name, k_str = metric.split("@")
    k = int(k_str)
    fn = ndcg_at_k if name.lower() == "ndcg" else recall_at_k

    scores: Dict[str, float] = {}
    overlapping = set(run.keys()) & set(qrels.keys())
    for qid in overlapping:
        scores[qid] = fn(run[qid], qrels[qid], k)
    return scores


# ---------------------------------------------------------------------------
#  Remap helpers
# ---------------------------------------------------------------------------

def remap_run_docs(
    run: Dict[str, List[str]],
    mapping: Dict[str, str],
) -> Dict[str, List[str]]:
    """Replace doc IDs in *run* using *mapping* (e.g. title→idx or idx→title)."""
    return {
        qid: [mapping.get(d, d) for d in docs]
        for qid, docs in run.items()
    }


# ---------------------------------------------------------------------------
#  Build per-model, per-dataset score tables
# ---------------------------------------------------------------------------

def build_all_scores(
    metric: str = "ndcg@20",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns nested dict:  model → dataset → {qid: score}
    Handles loading, format conversion, doc-ID remapping.
    """
    all_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Pre-load corpus mappings and qrels per dataset
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    for ds_name, ds_cfg in DATASETS.items():
        cache: Dict[str, Any] = {}

        # Load qrels
        if ds_cfg["qrels_file"] is not None:
            cache["qrels"] = load_qrels_trec(ds_cfg["qrels_file"])
        else:
            cache["qrels"] = build_limit_plus_qrels(ds_cfg["query_file"])

        # ID-based qrels (for models that use quest_xxx IDs)
        if "id_qrels_file" in ds_cfg and ds_cfg.get("id_qrels_file"):
            cache["id_qrels"] = load_qrels_trec(ds_cfg["id_qrels_file"])
        else:
            cache["id_qrels"] = None

        # Corpus mappings
        if ds_cfg["corpus_file"] is not None and os.path.isfile(ds_cfg["corpus_file"]):
            t2i, i2t = load_corpus_mappings(ds_cfg["corpus_file"])
            cache["title_to_idx"] = t2i
            cache["idx_to_title"] = i2t
        else:
            cache["title_to_idx"] = {}
            cache["idx_to_title"] = {}

        # Query metadata
        cache["meta"] = load_query_metadata(ds_name, ds_cfg["query_file"])

        # For LogiCoL: build query-text → qid mapping
        qt2qid: Dict[str, str] = {}
        qt2qid_lower: Dict[str, str] = {}
        with open(ds_cfg["query_file"], "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = str(obj["id"])
                q_text = obj.get("query", obj.get("nl_query", "")).strip()
                if q_text:
                    qt2qid[q_text] = qid
                    qt2qid_lower[q_text.lower()] = qid
        cache["query_text_to_qid"] = {**qt2qid, **qt2qid_lower}

        dataset_cache[ds_name] = cache
        print(f"[{ds_name}] Loaded {len(cache['qrels'])} qrels, "
              f"{len(cache['meta'])} query metadata entries, "
              f"{len(cache.get('title_to_idx', {}))} corpus title→id mappings.")

    # Load each model × dataset
    for model_name in MODEL_ORDER:
        all_scores[model_name] = {}
        for ds_name in DATASETS:
            run_cfg = MODELS.get(model_name, {}).get(ds_name)
            if run_cfg is None:
                all_scores[model_name][ds_name] = {}
                continue

            cache = dataset_cache[ds_name]
            qt_map = cache["query_text_to_qid"]

            # Load run
            run = load_run(run_cfg, ds_name, query_text_to_qid=qt_map)
            if run is None or len(run) == 0:
                print(f"  [{model_name} / {ds_name}] Run file not found or empty.")
                all_scores[model_name][ds_name] = {}
                continue

            # Doc-ID normalisation
            run_doc_type = run_cfg.get("doc", "title")
            ds_doc_type = DATASETS[ds_name]["doc_id_field"]

            if ds_doc_type == "title" and run_doc_type == "idx":
                # Run uses quest_xxx but qrels use titles → convert run to titles
                run = remap_run_docs(run, cache["idx_to_title"])
            elif ds_doc_type == "idx" and run_doc_type == "title":
                # Run uses titles but qrels use quest_xxx → convert run to idx
                run = remap_run_docs(run, cache["title_to_idx"])
            # If both match, or "name" (LIMIT+), no conversion needed

            # Choose correct qrels
            if ds_doc_type == "idx" and run_doc_type == "idx":
                qrels = cache["qrels"]
            elif ds_doc_type == "title" and run_doc_type == "idx":
                # After remapping run is now title-based
                qrels = cache["qrels"]
            elif ds_name == "QUEST" and run_doc_type == "idx":
                # LogiCoL QUEST: use id_qrels
                qrels = cache["id_qrels"] if cache["id_qrels"] else cache["qrels"]
            else:
                qrels = cache["qrels"]

            scores = per_query_scores(run, qrels, metric)
            all_scores[model_name][ds_name] = scores
            n_q = len(scores)
            avg = np.mean(list(scores.values())) if scores else 0.0
            print(f"  [{model_name} / {ds_name}] {n_q} queries evaluated, "
                  f"avg {metric} = {avg:.4f}")

    return all_scores


# ---------------------------------------------------------------------------
#  Aggregation (group by template / depth)
# ---------------------------------------------------------------------------

def aggregate_by_group(
    scores: Dict[str, float],
    meta: Dict[str, Tuple[str, int]],
    group_by: str,  # "template" or "depth"
) -> Dict[Any, Tuple[float, float, int]]:
    """
    Returns {group_key: (mean, stderr, count)}.
    group_key is a template string or an integer depth.
    """
    buckets: Dict[Any, List[float]] = defaultdict(list)
    for qid, score in scores.items():
        if qid not in meta:
            continue
        template, depth = meta[qid]
        key = template if group_by == "template" else depth
        buckets[key].append(score)

    result: Dict[Any, Tuple[float, float, int]] = {}
    for key, vals in buckets.items():
        arr = np.array(vals)
        mean = float(np.mean(arr))
        stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        result[key] = (mean, stderr, len(arr))
    return result


# ---------------------------------------------------------------------------
#  Colour palette & style (publication-quality)
# ---------------------------------------------------------------------------

def _get_palette(n: int) -> List[str]:
    """Generate a colour palette with good contrast for up to ~15 models."""
    # Curated: Wong (2011) colour-blind friendly + carefully chosen extensions
    base = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#8B4513",  # saddle brown
        "#000000",  # black
        "#7570B3",  # medium purple
        "#E7298A",  # deep pink
        "#66A61E",  # lime green
        "#1B9E77",  # teal
        "#A6761D",  # dark goldenrod
    ]
    return base[:n]


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "d", "p", "*", "h", "<", ">"]
_LINESTYLES = [
    "-", "-", "-", "-", "-",
    "--", "--", "--",
    "-.", "-.", "-.",
    ":", ":",
]


def _setup_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_figure(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    all_meta: Dict[str, Dict[str, Tuple[str, int]]],
    metric: str = "ndcg@20",
    output_path: str = "figure_constraint_depth.pdf",
):
    _setup_style()

    dataset_order = ["QUEST", "QUEST+", "LIMIT+"]
    n_rows = len(dataset_order)
    n_cols = 2

    colours = _get_palette(len(MODEL_ORDER))
    model_colour = {m: colours[i] for i, m in enumerate(MODEL_ORDER)}
    model_marker = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(MODEL_ORDER)}
    model_ls = {m: _LINESTYLES[i % len(_LINESTYLES)] for i, m in enumerate(MODEL_ORDER)}

    metric_label = (metric.replace("@", "@").upper()
                    .replace("NDCG", "nDCG").replace("RECALL", "Recall"))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(13.5, 10),
        gridspec_kw={"hspace": 0.38, "wspace": 0.20},
    )

    # ------------------------------------------------------------------
    # Compute y-limits: per-row (shared across both columns in each row)
    # so that the two analyses within a dataset are directly comparable.
    # ------------------------------------------------------------------
    row_ylims: Dict[int, Tuple[float, float]] = {}
    for row_i, ds_name in enumerate(dataset_order):
        meta = all_meta[ds_name]
        vals: List[float] = []
        for model_name in MODEL_ORDER:
            scores = all_scores.get(model_name, {}).get(ds_name, {})
            if not scores:
                continue
            for group_by in ["template", "depth"]:
                agg = aggregate_by_group(scores, meta, group_by)
                for _, (mean, stderr, _) in agg.items():
                    vals.append(mean + stderr)
                    vals.append(mean - stderr)
        if vals:
            lo = max(0, min(vals) - 0.02)
            hi = min(1.0, max(vals) + 0.025)
            # Snap to nice round numbers
            lo = np.floor(lo * 20) / 20  # snap to nearest 0.05 below
            hi = np.ceil(hi * 20) / 20   # snap to nearest 0.05 above
            row_ylims[row_i] = (lo, hi)
        else:
            row_ylims[row_i] = (0, 0.5)

    # ------------------------------------------------------------------
    # Plot each subplot
    # ------------------------------------------------------------------
    for row_i, ds_name in enumerate(dataset_order):
        meta = all_meta[ds_name]

        for col_i, group_by in enumerate(["template", "depth"]):
            ax = axes[row_i, col_i]

            # Determine x-axis categories
            if group_by == "template":
                templates_present = sorted(
                    set(t for t, _ in meta.values()),
                    key=lambda t: (TEMPLATE_ORDER.index(t)
                                   if t in TEMPLATE_ORDER else 99),
                )
                x_labels = [TEMPLATE_SHORT.get(t, t) for t in templates_present]
                x_positions = np.arange(len(templates_present))
            else:
                depths_present = sorted(set(d for _, d in meta.values()))
                x_labels = [str(d) for d in depths_present]
                x_positions = np.arange(len(depths_present))

            has_any_data = False

            for model_name in MODEL_ORDER:
                scores = all_scores.get(model_name, {}).get(ds_name, {})
                if not scores:
                    continue

                agg = aggregate_by_group(scores, meta, group_by)
                if not agg:
                    continue

                keys = templates_present if group_by == "template" else depths_present

                y_means: List[float] = []
                y_errs: List[float] = []
                x_valid: List[float] = []
                for ki, key in enumerate(keys):
                    if key in agg:
                        mean, stderr, _ = agg[key]
                        y_means.append(mean)
                        y_errs.append(stderr)
                        x_valid.append(float(x_positions[ki]))

                if not y_means:
                    continue

                has_any_data = True
                ax.errorbar(
                    x_valid, y_means, yerr=y_errs,
                    label=model_name,
                    color=model_colour[model_name],
                    marker=model_marker[model_name],
                    linestyle=model_ls[model_name],
                    capsize=2.2,
                    capthick=0.7,
                    elinewidth=0.7,
                    alpha=0.88,
                    zorder=3,
                )

            # ---- Axis styling ----
            ax.set_xticks(x_positions)
            if group_by == "template":
                ax.set_xticklabels(
                    x_labels, rotation=35, ha="right",
                    fontsize=7, rotation_mode="anchor",
                )
            else:
                ax.set_xticklabels(x_labels, ha="center")

            ax.set_ylim(row_ylims[row_i])
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            # Subtle alternating background bands for constraint-type columns
            if group_by == "template" and len(x_positions) > 1:
                for xi in range(0, len(x_positions), 2):
                    ax.axvspan(xi - 0.5, xi + 0.5,
                               color="#f5f5f5", zorder=0, lw=0)

            # y-label on left column only
            if col_i == 0:
                ax.set_ylabel(metric_label, fontsize=10)

            # Row label: dataset name to the left of the row (on left column)
            if col_i == 0:
                ax.annotate(
                    ds_name,
                    xy=(-0.23, 0.5), xycoords="axes fraction",
                    fontsize=12, fontweight="bold",
                    ha="center", va="center", rotation=90,
                    color="#333333",
                )

            # Column title on first row
            if row_i == 0:
                title = ("Constraint Type"
                         if group_by == "template"
                         else "Compositional Depth")
                ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

            # x-label on bottom row
            if row_i == n_rows - 1:
                xlabel = ("Constraint Type"
                          if group_by == "template"
                          else "# Atomic Predicates")
                ax.set_xlabel(xlabel, fontsize=9)

            # Empty-data message
            if not has_any_data:
                ax.text(
                    0.5, 0.5, "No runs available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#aaaaaa", fontstyle="italic",
                )

    # ------------------------------------------------------------------
    # Shared legend at the bottom of the figure
    # ------------------------------------------------------------------
    legend_handles = []
    for model_name in MODEL_ORDER:
        has_data = any(
            bool(all_scores.get(model_name, {}).get(ds, {}))
            for ds in dataset_order
        )
        legend_handles.append(
            Line2D(
                [0], [0],
                color=model_colour[model_name],
                marker=model_marker[model_name],
                linestyle=model_ls[model_name],
                markersize=5,
                linewidth=1.4,
                alpha=0.35 if not has_data else 1.0,
                label=model_name,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(7, len(legend_handles)),
        bbox_to_anchor=(0.5, -0.025),
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor="#dddddd",
        facecolor="white",
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.6,
    )

    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    print(f"\n[figure] Saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Constraint-type & depth analysis figure for ComplexIR baselines."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(_REPO_ROOT / "outputs" / "figure_constraint_depth.pdf"),
        help="Output PDF path (default: outputs/figure_constraint_depth.pdf).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg@20",
        choices=["ndcg@20", "ndcg@5", "ndcg@100", "recall@100", "recall@20"],
        help="Metric to plot (default: ndcg@20).",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("=" * 70)
    print("  Constraint-Type & Depth Analysis Figure")
    print(f"  Metric: {args.metric}")
    print("=" * 70)

    # 1. Load all scores
    all_scores = build_all_scores(metric=args.metric)

    # 2. Collect all metadata
    all_meta: Dict[str, Dict[str, Tuple[str, int]]] = {}
    for ds_name, ds_cfg in DATASETS.items():
        all_meta[ds_name] = load_query_metadata(ds_name, ds_cfg["query_file"])

    # 3. Plot
    plot_figure(all_scores, all_meta, metric=args.metric, output_path=args.output)

    # 4. Print summary statistics
    print("\n" + "=" * 70)
    print("  Summary: Mean scores per model × dataset")
    print("=" * 70)
    header = f"{'Model':<20}"
    for ds in ["QUEST", "QUEST+", "LIMIT+"]:
        header += f"  {ds:>10}"
    print(header)
    print("-" * 70)
    for model in MODEL_ORDER:
        row = f"{model:<20}"
        for ds in ["QUEST", "QUEST+", "LIMIT+"]:
            s = all_scores.get(model, {}).get(ds, {})
            if s:
                row += f"  {np.mean(list(s.values())):>10.4f}"
            else:
                row += f"  {'—':>10}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
