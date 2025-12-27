"""
Preprocess a run file to ensure it's in proper TREC format.

Reads:
    Run file (may be in various formats)

Writes:
    TREC run file with format: qid Q0 docid rank score run_name
    
    The script will:
    - Ensure proper field order and format
    - Sort results by score (descending) for each query
    - Assign sequential ranks starting from 1
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple


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


def preprocess_run(input_path: str, output_path: str, default_run_name: str = "run"):
    """
    Preprocess run file to TREC format.
    
    Args:
        input_path: Path to input run file
        output_path: Path to output TREC run file
        default_run_name: Default run name if not found in input
    """
    # Load all results
    results: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
    
    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            parsed = parse_run_line(line)
            if parsed is None:
                print(f"[preprocess] Warning: Skipping malformed line {line_num}: {line[:80]}")
                continue
            
            qid, docid, score, run_name = parsed
            if not run_name or run_name == "run":
                run_name = default_run_name
            
            results[qid].append((docid, score, run_name))
    
    # Sort by score (descending) for each query and write output
    with open(output_path, "w", encoding="utf-8") as outfile:
        for qid in sorted(results.keys()):
            query_results = results[qid]
            
            # Sort by score descending
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
    args = parser.parse_args()
    
    # If no output path provided, create one with _trec suffix
    if args.output is None:
        base_path = os.path.splitext(args.input)[0]
        args.output = f"{base_path}_trec"
    
    preprocess_run(args.input, args.output, args.run_name)
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()

