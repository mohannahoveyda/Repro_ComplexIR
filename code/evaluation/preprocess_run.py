"""
Preprocess a run file to ensure it's in proper TREC format.

Reads:
    - TREC/TSV/space-separated run lines
    - JSONL with "id", "docs", "scores" (one object per query)
    - ReasonIR JSONL: "query", "relevant", "retrieved" (list of {rank, score, title});
      qid is line number (1-based) to match test_id_added.jsonl order

Writes:
    TREC run file with format: qid Q0 docid rank score run_name
    
    The script will:
    - Ensure proper field order and format
    - Sort results by score for each query:
      * If all scores are negative: sort ascending (more negative = rank 1)
      * If scores are positive: sort descending (highest = rank 1)
    - Assign sequential ranks starting from 1
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_query_ids(test_queries_path: str) -> List[str]:
    """
    Load query IDs from a test queries file (one ID per line, in file order).

    Supports:
    - LIMIT / BEIR format: ``{"_id": "query_0", "text": "..."}``
    - LIMIT+ format: ``{"id": 0, "query": "...", "docs": [...]}``

    Returns:
        List of query IDs (strings) in the order they appear in the file.
    """
    query_ids: List[str] = []
    with open(test_queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Try _id first (LIMIT / BEIR), then id (LIMIT+)
            qid = data.get("_id")
            if qid is None:
                qid = data.get("id")
            if qid is not None:
                query_ids.append(str(qid))
    return query_ids


def generate_trec_qrels(
    output_path: str,
    qrels_jsonl_path: Optional[str] = None,
    test_queries_path: Optional[str] = None,
) -> None:
    """
    Generate a TREC-format qrels file from LIMIT / LIMIT+ data.

    Two modes (at least one source must be provided):

    1. **qrels_jsonl_path** – BEIR-style JSONL
       (``{"query-id": "q1", "corpus-id": "d1", "score": 1}``).
    2. **test_queries_path** – LIMIT+ test file with inline relevance
       (``{"id": 0, "query": "...", "docs": ["doc1", ...]}``) or
       LIMIT test file with ``_id`` and ``docs``.

    Writes TREC qrels: ``qid 0 docid relevance``
    """
    lines: List[str] = []

    if qrels_jsonl_path and os.path.isfile(qrels_jsonl_path):
        # Mode 1: BEIR JSONL qrels
        with open(qrels_jsonl_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                qid = str(data.get("query-id", ""))
                docid = str(data.get("corpus-id", ""))
                score = data.get("score", 0)
                if qid and docid and int(score) > 0:
                    lines.append(f"{qid} 0 {docid} {int(score)}")

    if test_queries_path and os.path.isfile(test_queries_path):
        # Mode 2: test file with inline docs
        with open(test_queries_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
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
                    lines.append(f"{qid} 0 {doc} 1")

    if not lines:
        print("[preprocess] WARNING: No qrels generated – check source paths.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # Count unique queries
    unique_qids = set(l.split()[0] for l in lines)
    print(f"[preprocess] Wrote TREC qrels ({len(unique_qids)} queries, {len(lines)} judgments) to: {output_path}")


def parse_reasonir_jsonl_line(
    line: str, line_num: int, qid_override: Optional[str] = None,
) -> Optional[List[Tuple[str, str, float]]]:
    """
    Parse a ReasonIR JSONL line (one object per query).
    
    Expected format:
    {
        "query": "...",
        "relevant": ["doc1", ...],
        "retrieved": [{"rank": 1, "score": 0.42, "title": "Doc Title"}, ...]
    }
    
    qid is taken as line_num (1-based) so order matches test_id_added.jsonl.
    
    Returns:
        List of (qid, docid, score) tuples or None if parsing fails
    """
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    retrieved = data.get("retrieved", [])
    if not isinstance(retrieved, list):
        return None
    if not retrieved:
        return []  # Valid line with no results (e.g. retrieval timed out)
    qid = qid_override if qid_override is not None else str(line_num)
    out = []
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        score = item.get("score", 0.0)
        if title is None or title == "":
            continue
        try:
            out.append((qid, str(title), float(score)))
        except (TypeError, ValueError):
            continue
    return out if out else None


def parse_jsonl_line(line: str) -> Optional[List[Tuple[str, str, float]]]:
    """
    Parse a JSONL line containing query results.
    
    Expected format:
    {
        "id": "query_id",
        "docs": ["doc1", "doc2", ...],
        "scores": [score1, score2, ...],
        ...
    }
    
    Returns:
        List of (qid, docid, score) tuples or None if parsing fails
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    # ReasonIR format: "retrieved" with rank/score/title (handled separately with line_num)
    if "retrieved" in data and isinstance(data.get("retrieved"), list):
        return None  # Let parse_reasonir_jsonl_line handle it
    
    # Extract query ID
    qid = str(data.get("id", ""))
    if not qid:
        return None
    
    # Extract docs and scores
    docs = data.get("docs", [])
    scores = data.get("scores", [])
    
    # Ensure docs and scores have the same length
    if len(docs) != len(scores):
        return None
    
    # Return list of (qid, docid, score) tuples
    return [(qid, str(doc), float(score)) for doc, score in zip(docs, scores)]
    

def parse_run_line(line: str) -> Tuple[str, str, float, str]:
    """
    Parse a line from a run file.
    
    Supports multiple input formats:
    - TREC format: qid Q0 docid rank score run_name 
    - Simplified: qid docid score
    - Tab-separated: qid\tdocid\tscore
    
    Returns:
        Tuple of (qid, docid, score, run_name) or None if parsing fails
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split()
    if len(parts) >= 6 and parts[1] == "Q0":
        qid = parts[0]
        
        rank_idx = None
        score_idx = None
        for i in range(2, len(parts)):
            if rank_idx is None:
                try:
                    int(parts[i])
                    rank_idx = i
                    continue
                except ValueError:
                    pass
            
            if rank_idx is not None and score_idx is None:
                try:
                    float(parts[i])
                    score_idx = i
                    break
                except ValueError:
                    pass
        
        if rank_idx is None or score_idx is None:
            return None
        
        docid = " ".join(parts[2:rank_idx])
        try:
            score = float(parts[score_idx])
        except (ValueError, IndexError):
            return None
        run_name = " ".join(parts[score_idx + 1:]) if len(parts) > score_idx + 1 else "run"
        return (qid, docid, score, run_name)
    
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) >= 3:
            qid = parts[0].strip()
            docid = parts[1].strip()
            try:
                score = float(parts[2].strip())
            except (ValueError, IndexError):
                return None
            run_name = parts[3].strip() if len(parts) > 3 else "run"
            return (qid, docid, score, run_name)
    
    if len(parts) >= 3:
        qid = parts[0]
        score_idx = None
        for i in range(1, len(parts)):
            try:
                float(parts[i])
                score_idx = i
                break
            except ValueError:
                pass
        
        if score_idx is None:
            return None
        
        docid = " ".join(parts[1:score_idx])
        try:
            score = float(parts[score_idx])
        except (ValueError, IndexError):
            return None
        run_name = " ".join(parts[score_idx + 1:]) if len(parts) > score_idx + 1 else "run"
        return (qid, docid, score, run_name)
    
    return None


def preprocess_run(
    input_path: str,
    output_path: str,
    default_run_name: str = "run",
    test_queries_path: Optional[str] = None,
):
    """
    Preprocess run file to TREC format.
    
    Args:
        input_path: Path to input run file
        output_path: Path to output TREC run file
        default_run_name: Default run name if not found in input
        test_queries_path: Optional path to a test queries file whose IDs
            replace the 1-based line numbers used for ReasonIR runs
            (required for LIMIT / LIMIT+ to get correct query IDs).
    """
    # Load query IDs from test file if provided (for LIMIT / LIMIT+)
    query_ids: Optional[List[str]] = None
    if test_queries_path:
        query_ids = load_query_ids(test_queries_path)
        print(f"[preprocess] Loaded {len(query_ids)} query IDs from: {test_queries_path}")

    # Load all results
    results: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
    
    # Detect if file is JSONL format by checking first line
    is_jsonl = False
    is_reasonir = False
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            first_line = infile.readline().strip()
            if first_line:
                try:
                    data = json.loads(first_line)
                    if isinstance(data, dict):
                        if "retrieved" in data and isinstance(data.get("retrieved"), list):
                            is_reasonir = True
                            is_jsonl = True
                        elif "docs" in data or "scores" in data:
                            is_jsonl = True
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception:
        pass
    
    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            if is_reasonir:
                # Use mapped query ID when a test-queries file was provided
                qid_override = None
                if query_ids is not None and line_num <= len(query_ids):
                    qid_override = query_ids[line_num - 1]
                parsed_list = parse_reasonir_jsonl_line(line, line_num, qid_override=qid_override)
                if parsed_list is not None:
                    for qid, docid, score in parsed_list:
                        results[qid].append((docid, score, default_run_name))
                    continue
            if is_jsonl:
                # Try parsing as JSONL (id/docs/scores) first
                parsed_list = parse_jsonl_line(line)
                if parsed_list is not None:
                    for qid, docid, score in parsed_list:
                        results[qid].append((docid, score, default_run_name))
                    continue
            
            # Try parsing as TREC format
            parsed = parse_run_line(line)
            if parsed is None:
                print(f"[preprocess] Warning: Skipping malformed line {line_num}: {line[:80]}")
                continue
            
            qid, docid, score, run_name = parsed
            if not run_name or run_name == "run":
                run_name = default_run_name
            
            results[qid].append((docid, score, run_name))
    
    # Sort qids: numeric when possible (e.g. ReasonIR 1,2,...,1727), else string
    def qid_sort_key(q):
        try:
            return (0, int(q))
        except ValueError:
            return (1, q)

    # Sort by score (descending) for each query and write output
    with open(output_path, "w", encoding="utf-8") as outfile:
        for qid in sorted(results.keys(), key=qid_sort_key):
            query_results = results[qid]
            
            # Determine if scores are negative (more negative = better)
            # Check if all scores are negative
            all_negative = all(score < 0 for _, score, _ in query_results)
            
            if all_negative:
                # For negative scores: sort ascending (most negative = rank 1)
                query_results.sort(key=lambda x: x[1], reverse=False)
            else:
                # For positive scores: sort descending (highest = rank 1)
                query_results.sort(key=lambda x: x[1], reverse=True)
            
            # Get run_name from first result (should be same for all)
            run_name = query_results[0][2] if query_results else default_run_name
            
            # Write with sequential ranks
            for rank, (docid, score, _) in enumerate(query_results, start=1):
                # TREC format: qid Q0 docid rank score run_name
                outfile.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")
    
    print(f"[preprocess] Processed {len(results)} queries")
    print(f"[preprocess] Wrote TREC run file to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess run file to TREC format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input run file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output TREC run file (default: input path with _trec suffix)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="Run name to use if not found in input file",
    )
    parser.add_argument(
        "--test-queries",
        type=str,
        default=None,
        help=(
            "Path to a test queries file (JSONL) whose IDs replace 1-based "
            "line numbers for ReasonIR runs.  Supports LIMIT format "
            '({"_id": "query_0", ...}) and LIMIT+ format ({"id": 0, ...}).'
        ),
    )
    parser.add_argument(
        "--generate-qrels",
        type=str,
        default=None,
        metavar="QRELS_OUT",
        help=(
            "Also generate a TREC qrels file at this path. Requires at least "
            "one of --qrels-jsonl or --test-queries to supply relevance data."
        ),
    )
    parser.add_argument(
        "--qrels-jsonl",
        type=str,
        default=None,
        help=(
            "Path to a BEIR-style JSONL qrels file "
            '({"query-id": ..., "corpus-id": ..., "score": ...}). '
            "Used with --generate-qrels to convert to TREC format."
        ),
    )
    args = parser.parse_args()
    
    # If no output path provided, create one with _trec suffix
    if args.output is None:
        base_path = os.path.splitext(args.input)[0]
        args.output = f"{base_path}_trec"
    
    preprocess_run(args.input, args.output, args.run_name,
                   test_queries_path=args.test_queries)

    # Optionally generate TREC qrels
    if args.generate_qrels:
        generate_trec_qrels(
            args.generate_qrels,
            qrels_jsonl_path=args.qrels_jsonl,
            test_queries_path=args.test_queries,
        )

    print("[preprocess] Done.")


if __name__ == "__main__":
    main()

