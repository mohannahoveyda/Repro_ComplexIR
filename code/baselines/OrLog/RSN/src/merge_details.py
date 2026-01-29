#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser(
        description="Merge all detail.csv from chunked runs into one full_details.csv"
    )
    p.add_argument(
        "exp_dir",
        type=Path,
        help="Path to the experiment folder (e.g. runs/8B_test20)"
    )
    args = p.parse_args()

    exp = args.exp_dir
    results = exp / "results"
    if not results.is_dir():
        print(f"ERROR: {results} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # find all chunk directories
    chunk_dirs = sorted([d for d in results.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    if not chunk_dirs:
        print(f"ERROR: no chunk_* subdirs found under {results}", file=sys.stderr)
        sys.exit(1)

    detail_paths = []
    for d in chunk_dirs:
        f = d / "detail.csv"
        if not f.exists():
            print(f"ERROR: expected {f} but not found.", file=sys.stderr)
            sys.exit(1)
        detail_paths.append(f)

    # read & concat
    dfs = [pd.read_csv(f) for f in detail_paths]
    full = pd.concat(dfs, ignore_index=True)

    # sanity checks
    # 1) no fully identical duplicate rows
    dup_rows = full.duplicated().sum()
    if dup_rows:
        print(f"ERROR: found {dup_rows} duplicate rows in concatenation.", file=sys.stderr)
        sys.exit(1)

    # 2) we saw one file per chunk dir
    if len(dfs) != len(chunk_dirs):
        print(f"ERROR: read {len(dfs)} files but found {len(chunk_dirs)} chunk dirs.", file=sys.stderr)
        sys.exit(1)

    # write out
    out_path = exp / "full_details.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote merged details → {out_path}")

if __name__ == "__main__":
    main()
