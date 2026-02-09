#!/usr/bin/env python3
"""
Quick script to compare costs between two models.

Usage:
    python compare_model_costs.py \
        --model1 qwen/qwen3-vl-8b-instruct \
        --model2 meta-llama/llama-3.3-70b-instruct \
        --top-k 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_estimation(model_name: str, top_k: int, output_file: str):
    """Run cost estimation for a model."""
    script_path = Path(__file__).parent / "estimate_reranking_costs.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--model-name", model_name,
        "--top-k", str(top_k),
        "--output-file", output_file,
    ]
    
    print(f"\n{'='*80}")
    print(f"Estimating costs for: {model_name}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running estimation: {result.stderr}")
        return None
    
    # Extract total cost from output
    for line in result.stdout.split('\n'):
        if 'TOTAL ESTIMATED COST:' in line:
            return line.split('$')[-1].strip()
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare costs between two models"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="First model name",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Second model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of top documents to rerank per query (default: 1000)",
    )
    
    args = parser.parse_args()
    
    # Run estimations
    cost1 = run_estimation(args.model1, args.top_k, f"cost_{args.model1.replace('/', '_')}.txt")
    cost2 = run_estimation(args.model2, args.top_k, f"cost_{args.model2.replace('/', '_')}.txt")
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"COST COMPARISON")
    print(f"{'='*80}")
    print(f"Model 1: {args.model1}")
    print(f"  Total Cost: ${cost1 if cost1 else 'N/A'}")
    print(f"\nModel 2: {args.model2}")
    print(f"  Total Cost: ${cost2 if cost2 else 'N/A'}")
    
    if cost1 and cost2:
        try:
            cost1_val = float(cost1.replace(',', ''))
            cost2_val = float(cost2.replace(',', ''))
            diff = cost2_val - cost1_val
            pct_diff = (diff / cost1_val) * 100
            
            print(f"\nDifference: ${diff:,.2f} ({pct_diff:+.1f}%)")
            if cost1_val < cost2_val:
                print(f"✓ {args.model1} is cheaper by ${abs(diff):,.2f}")
            elif cost2_val < cost1_val:
                print(f"✓ {args.model2} is cheaper by ${abs(diff):,.2f}")
            else:
                print(f"Both models have the same cost")
        except ValueError:
            pass
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
