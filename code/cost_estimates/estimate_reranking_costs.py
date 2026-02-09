#!/usr/bin/env python3
"""
Cost estimation script for reranking experiments using OpenRouter API.

This script estimates the costs of running reranking experiments on top of BM25 results.
It calculates costs per dataset and provides detailed breakdowns.

Usage:
    python code/cost_estimates/estimate_reranking_costs.py \
        --model-name meta-llama/Llama-3.3-70B-Instruct \
        --top-k 1000 \
        --output-file cost_estimates.txt
"""

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from tqdm import tqdm


# Default paths
BASE_DIR = Path(__file__).parent.parent.parent  # Go up from code/cost_estimates/ to project root
CACHE_DIR = BASE_DIR / ".cost_cache"
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_CSV_FILE = BASE_DIR / "cost_estimates.csv"
DEFAULT_CURATED_CSV_FILE = BASE_DIR / "curated_cost_estimates.csv"

# Dataset configurations
DATASETS = {
    "QUEST": {
        "queries": BASE_DIR / "data/QUEST/test_id_added.jsonl",
        "corpus": BASE_DIR / "data/QUEST/documents.jsonl",
    },
    "QUEST+": {
        "queries": BASE_DIR / "data/QUEST_w_Variants/data/quest_test_withVarients_converted.jsonl",
        "corpus": BASE_DIR / "data/QUEST_w_Variants/data/quest_text_w_id_withVarients.jsonl",
    },
    "LIMIT": {
        "queries": BASE_DIR / "code/data_generation_utils/limit_plus/limit_data/queries.jsonl",
        "corpus": BASE_DIR / "code/data_generation_utils/limit_plus/limit_data/corpus.jsonl",
    },
    "LIMIT+": {
        "queries": BASE_DIR / "code/data_generation_utils/limit_plus/limit_data/limit_quest_queries.jsonl",
        "corpus": BASE_DIR / "code/data_generation_utils/limit_plus/limit_data/corpus.jsonl",
    },
}

# Datasets eligible for curated reranking (only these have gold doc annotations)
CURATED_DATASETS = ["QUEST", "LIMIT+"]

# Default token estimates
DEFAULT_INSTRUCTION_TOKENS = 50  # Estimated tokens for instruction prompt
DEFAULT_OUTPUT_TOKENS = 5  # Estimated tokens for relevance score output
DEFAULT_NUM_IRRELEVANT = 5  # Number of irrelevant docs (lowest BM25 scores)
DEFAULT_NUM_NOISE = 5  # Number of semi-relevant noise docs (highest BM25 non-gold)


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token on average.
    This is a conservative estimate (actual tokens may be fewer).
    """
    if not text:
        return 0
    return len(text) // 4


def get_openrouter_pricing(model_name: str, api_key: Optional[str] = None) -> Tuple[float, float]:
    """
    Fetch pricing for a model from OpenRouter API.
    
    Returns:
        (input_price_per_token, output_price_per_token)
    """
    cache_file = CACHE_DIR / f"pricing_{model_name.replace('/', '_')}.pkl"
    
    # Check cache first
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
            return cached["input"], cached["output"]
    
    # Fetch from API
    url = "https://openrouter.ai/api/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        models = response.json().get("data", [])
        
        # Find the model (case-insensitive matching)
        model_name_lower = model_name.lower()
        for model in models:
            model_id = model.get("id", "")
            # Try exact match first, then case-insensitive
            if model_id == model_name or model_id.lower() == model_name_lower:
                pricing = model.get("pricing", {})
                input_price = float(pricing.get("prompt", "0"))
                output_price = float(pricing.get("completion", "0"))
                
                print(f"Found model '{model_id}' with pricing: input=${input_price:.8f}, output=${output_price:.8f}")
                
                # Cache the result
                with open(cache_file, "wb") as f:
                    pickle.dump({"input": input_price, "output": output_price}, f)
                
                return input_price, output_price
        
        # Model not found, try to suggest similar models
        print(f"Warning: Model '{model_name}' not found in OpenRouter catalog.")
        print("Searching for similar model names...")
        similar = [m.get("id") for m in models if model_name_lower in m.get("id", "").lower()][:5]
        if similar:
            print(f"Similar models found: {', '.join(similar)}")
        print("Using default pricing (0.0).")
        return 0.0, 0.0
        
    except Exception as e:
        print(f"Error fetching pricing: {e}")
        print("Using default pricing (0.0). Please check your API key or model name.")
        return 0.0, 0.0


def estimate_document_tokens(corpus_path: Path, quest_plus: bool = False, sample_size: int = 100) -> float:
    """
    Estimate average document tokens by sampling documents from corpus.
    
    Args:
        corpus_path: Path to corpus file
        quest_plus: Whether this is QUEST+ format (different structure)
        sample_size: Number of documents to sample for estimation
    
    Returns:
        Average tokens per document
    """
    cache_file = CACHE_DIR / f"doc_tokens_{corpus_path.name}_{quest_plus}.pkl"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print(f"Estimating document tokens for {corpus_path.name}...")
    
    # Try to load documents, but handle very large files gracefully
    try:
        documents = load_jsonl(corpus_path)
    except Exception as e:
        print(f"Warning: Error loading corpus file: {e}")
        print("Trying to sample directly from file...")
        # Sample directly from file
        documents = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                if line.strip():
                    try:
                        documents.append(json.loads(line))
                    except:
                        continue
    
    if not documents:
        print(f"Warning: No documents found in {corpus_path}")
        return 0.0
    
    # Sample documents
    sample_indices = range(min(sample_size, len(documents)))
    total_tokens = 0
    
    for idx in tqdm(sample_indices, desc="Sampling documents"):
        doc = documents[idx]
        
        if quest_plus:
            # QUEST+ format: has 'text' field
            text = doc.get("text", "")
            title = doc.get("title", "")
            doc_text = f"{title}. {text}".strip() if title else text
        else:
            # Standard format: check for 'text' field
            if "text" in doc:
                text = doc.get("text", "")
                title = doc.get("title", "")
                doc_text = f"{title}. {text}".strip() if title else text
            else:
                # LIMIT format: just text field
                doc_text = doc.get("text", "")
        
        total_tokens += estimate_tokens(doc_text)
    
    avg_tokens = total_tokens / len(sample_indices) if sample_indices else 0.0
    
    # Cache the result
    with open(cache_file, "wb") as f:
        pickle.dump(avg_tokens, f)
    
    return avg_tokens


def estimate_query_tokens(queries_path: Path) -> float:
    """
    Estimate average query tokens.
    
    Returns:
        Average tokens per query
    """
    cache_file = CACHE_DIR / f"query_tokens_{queries_path.name}.pkl"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print(f"Estimating query tokens for {queries_path.name}...")
    queries = load_jsonl(queries_path)
    
    if not queries:
        print(f"Warning: No queries found in {queries_path}")
        return 0.0
    
    total_tokens = 0
    for query_item in tqdm(queries, desc="Processing queries"):
        # Handle different query formats
        query_text = query_item.get("query", "")
        if not query_text:
            # Try alternative field names
            query_text = query_item.get("text", "")
        total_tokens += estimate_tokens(query_text)
    
    avg_tokens = total_tokens / len(queries) if queries else 0.0
    
    # Cache the result
    with open(cache_file, "wb") as f:
        pickle.dump(avg_tokens, f)
    
    return avg_tokens


def count_queries(queries_path: Path) -> int:
    """Count number of queries in file."""
    cache_file = CACHE_DIR / f"query_count_{queries_path.name}.pkl"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    count = len(load_jsonl(queries_path))
    
    # Cache the result
    with open(cache_file, "wb") as f:
        pickle.dump(count, f)
    
    return count


def get_gold_docs_per_query(queries_path: Path) -> List[int]:
    """
    Read gold documents per query from the query JSONL file.
    
    Each query entry is expected to have a 'docs' field listing gold document titles.
    
    Returns:
        List of gold document counts, one per query
    """
    cache_file = CACHE_DIR / f"gold_docs_{queries_path.name}.pkl"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print(f"Counting gold documents per query for {queries_path.name}...")
    queries = load_jsonl(queries_path)
    
    gold_counts = []
    for query_item in queries:
        docs = query_item.get("docs", [])
        gold_counts.append(len(docs))
    
    # Cache the result
    with open(cache_file, "wb") as f:
        pickle.dump(gold_counts, f)
    
    return gold_counts


def calculate_dataset_costs(
    dataset_name: str,
    queries_path: Path,
    corpus_path: Path,
    top_k: int,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
    quest_plus: bool = False,
) -> Dict[str, float]:
    """
    Calculate costs for a single dataset (standard top-k mode).
    
    Returns:
        Dictionary with cost breakdown
    """
    print(f"\nCalculating costs for {dataset_name}...")
    
    # Get statistics
    num_queries = count_queries(queries_path)
    avg_query_tokens = estimate_query_tokens(queries_path)
    avg_doc_tokens = estimate_document_tokens(corpus_path, quest_plus=quest_plus)
    
    # Calculate tokens per reranking call
    tokens_per_input = instruction_tokens + avg_query_tokens + avg_doc_tokens
    
    # Total reranking calls = num_queries * top_k
    total_calls = num_queries * top_k
    
    # Calculate costs
    total_input_tokens = total_calls * tokens_per_input
    total_output_tokens = total_calls * output_tokens
    
    input_cost = total_input_tokens * input_price
    output_cost = total_output_tokens * output_price
    total_cost = input_cost + output_cost
    
    return {
        "dataset": dataset_name,
        "num_queries": num_queries,
        "top_k": top_k,
        "avg_query_tokens": avg_query_tokens,
        "avg_doc_tokens": avg_doc_tokens,
        "tokens_per_input": tokens_per_input,
        "total_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calculate_curated_dataset_costs(
    dataset_name: str,
    queries_path: Path,
    corpus_path: Path,
    num_irrelevant: int,
    num_noise: int,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
    quest_plus: bool = False,
) -> Dict[str, float]:
    """
    Calculate costs for a single dataset using curated candidate sets.
    
    For each query, the candidate set is:
        gold_docs + num_irrelevant (lowest BM25) + num_noise (highest BM25 non-gold)
    
    So per query: candidate_size = num_gold_docs + num_irrelevant + num_noise
    
    Returns:
        Dictionary with cost breakdown including curated-specific stats
    """
    print(f"\nCalculating curated reranking costs for {dataset_name}...")
    
    # Get statistics
    num_queries = count_queries(queries_path)
    avg_query_tokens = estimate_query_tokens(queries_path)
    avg_doc_tokens = estimate_document_tokens(corpus_path, quest_plus=quest_plus)
    
    # Get gold docs per query
    gold_counts = get_gold_docs_per_query(queries_path)
    
    if len(gold_counts) != num_queries:
        print(f"Warning: gold_counts ({len(gold_counts)}) != num_queries ({num_queries})")
    
    # Calculate per-query candidate set sizes
    candidate_sizes = [g + num_irrelevant + num_noise for g in gold_counts]
    total_calls = sum(candidate_sizes)
    avg_gold_docs = sum(gold_counts) / len(gold_counts) if gold_counts else 0.0
    min_gold_docs = min(gold_counts) if gold_counts else 0
    max_gold_docs = max(gold_counts) if gold_counts else 0
    avg_candidate_size = total_calls / num_queries if num_queries > 0 else 0.0
    
    # Calculate tokens per reranking call
    tokens_per_input = instruction_tokens + avg_query_tokens + avg_doc_tokens
    
    # Calculate costs
    total_input_tokens = total_calls * tokens_per_input
    total_output_tokens = total_calls * output_tokens
    
    input_cost = total_input_tokens * input_price
    output_cost = total_output_tokens * output_price
    total_cost = input_cost + output_cost
    
    return {
        "dataset": dataset_name,
        "num_queries": num_queries,
        "num_irrelevant": num_irrelevant,
        "num_noise": num_noise,
        "avg_gold_docs": avg_gold_docs,
        "min_gold_docs": min_gold_docs,
        "max_gold_docs": max_gold_docs,
        "avg_candidate_size": avg_candidate_size,
        "avg_query_tokens": avg_query_tokens,
        "avg_doc_tokens": avg_doc_tokens,
        "tokens_per_input": tokens_per_input,
        "total_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def save_to_csv(
    results: List[Dict[str, float]],
    model_name: str,
    top_k: int,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
    csv_file: Path,
    timestamp: str,
):
    """
    Save cost estimates to CSV file, updating existing entries or appending new ones.
    
    CSV format:
    - Per-dataset rows: model_name, top_k, dataset, num_queries, avg_query_tokens, 
      avg_doc_tokens, tokens_per_input, total_calls, total_input_tokens, 
      total_output_tokens, input_cost, output_cost, total_cost, timestamp
    - Summary row: model_name, top_k, "TOTAL", "", "", "", "", "", "", "", 
      total_input_cost, total_output_cost, total_cost, timestamp
    """
    # Calculate totals
    total_input_cost = sum(r["input_cost"] for r in results)
    total_output_cost = sum(r["output_cost"] for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    
    # Define CSV columns
    fieldnames = [
        "model_name",
        "top_k",
        "dataset",
        "num_queries",
        "avg_query_tokens",
        "avg_doc_tokens",
        "tokens_per_input",
        "total_calls",
        "total_input_tokens",
        "total_output_tokens",
        "input_price_per_token",
        "output_price_per_token",
        "instruction_tokens",
        "output_tokens",
        "input_cost",
        "output_cost",
        "dataset_cost",
        "model_total_cost",
        "timestamp",
    ]
    
    # Read existing CSV if it exists
    existing_rows = []
    if csv_file.exists():
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
    
    # Filter out rows for this model+top_k combination (we'll replace them)
    # Also filter out any legacy TOTAL rows
    filtered_rows = [
        row for row in existing_rows
        if not (row.get("model_name") == model_name and int(row.get("top_k", 0)) == top_k)
        and row.get("dataset") != "TOTAL"
    ]
    
    # Add per-dataset rows, each carrying the model_total_cost
    for result in results:
        row = {
            "model_name": model_name,
            "top_k": str(top_k),
            "dataset": result["dataset"],
            "num_queries": str(result["num_queries"]),
            "avg_query_tokens": f"{result['avg_query_tokens']:.2f}",
            "avg_doc_tokens": f"{result['avg_doc_tokens']:.2f}",
            "tokens_per_input": f"{result['tokens_per_input']:.2f}",
            "total_calls": str(result["total_calls"]),
            "total_input_tokens": str(int(result["total_input_tokens"])),
            "total_output_tokens": str(int(result["total_output_tokens"])),
            "input_price_per_token": f"{input_price:.8f}",
            "output_price_per_token": f"{output_price:.8f}",
            "instruction_tokens": str(instruction_tokens),
            "output_tokens": str(output_tokens),
            "input_cost": f"{result['input_cost']:.2f}",
            "output_cost": f"{result['output_cost']:.2f}",
            "dataset_cost": f"{result['total_cost']:.2f}",
            "model_total_cost": f"{total_cost:.2f}",
            "timestamp": timestamp,
        }
        filtered_rows.append(row)
    
    # Write updated CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"✓ Cost estimates saved to CSV: {csv_file.absolute()}")


def save_curated_to_csv(
    results: List[Dict[str, float]],
    model_name: str,
    num_irrelevant: int,
    num_noise: int,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
    csv_file: Path,
    timestamp: str,
):
    """
    Save curated reranking cost estimates to a separate CSV file.
    
    This CSV includes curated-specific columns like avg_gold_docs and
    avg_candidate_size instead of top_k.
    """
    # Calculate totals
    total_input_cost = sum(r["input_cost"] for r in results)
    total_output_cost = sum(r["output_cost"] for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    
    # Define CSV columns for curated mode
    fieldnames = [
        "model_name",
        "dataset",
        "num_queries",
        "num_irrelevant",
        "num_noise",
        "avg_gold_docs",
        "min_gold_docs",
        "max_gold_docs",
        "avg_candidate_size",
        "avg_query_tokens",
        "avg_doc_tokens",
        "tokens_per_input",
        "total_calls",
        "total_input_tokens",
        "total_output_tokens",
        "input_price_per_token",
        "output_price_per_token",
        "instruction_tokens",
        "output_tokens",
        "input_cost",
        "output_cost",
        "dataset_cost",
        "model_total_cost",
        "timestamp",
    ]
    
    # Read existing CSV if it exists
    existing_rows = []
    if csv_file.exists():
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
    
    # Filter out rows for this model + irrelevant + noise combination (we'll replace them)
    # Also filter out any legacy TOTAL rows
    filtered_rows = [
        row for row in existing_rows
        if not (
            row.get("model_name") == model_name
            and int(row.get("num_irrelevant", 0)) == num_irrelevant
            and int(row.get("num_noise", 0)) == num_noise
        )
        and row.get("dataset") != "TOTAL"
    ]
    
    # Add per-dataset rows, each carrying the model_total_cost
    for result in results:
        row = {
            "model_name": model_name,
            "dataset": result["dataset"],
            "num_queries": str(result["num_queries"]),
            "num_irrelevant": str(result["num_irrelevant"]),
            "num_noise": str(result["num_noise"]),
            "avg_gold_docs": f"{result['avg_gold_docs']:.2f}",
            "min_gold_docs": str(result["min_gold_docs"]),
            "max_gold_docs": str(result["max_gold_docs"]),
            "avg_candidate_size": f"{result['avg_candidate_size']:.2f}",
            "avg_query_tokens": f"{result['avg_query_tokens']:.2f}",
            "avg_doc_tokens": f"{result['avg_doc_tokens']:.2f}",
            "tokens_per_input": f"{result['tokens_per_input']:.2f}",
            "total_calls": str(result["total_calls"]),
            "total_input_tokens": str(int(result["total_input_tokens"])),
            "total_output_tokens": str(int(result["total_output_tokens"])),
            "input_price_per_token": f"{input_price:.8f}",
            "output_price_per_token": f"{output_price:.8f}",
            "instruction_tokens": str(instruction_tokens),
            "output_tokens": str(output_tokens),
            "input_cost": f"{result['input_cost']:.2f}",
            "output_cost": f"{result['output_cost']:.2f}",
            "dataset_cost": f"{result['total_cost']:.2f}",
            "model_total_cost": f"{total_cost:.2f}",
            "timestamp": timestamp,
        }
        filtered_rows.append(row)
    
    # Write updated CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"  Curated cost estimates saved to CSV: {csv_file.absolute()}")


def format_cost_report(
    results: List[Dict[str, float]],
    model_name: str,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
) -> str:
    """Format cost report as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("RERANKING COST ESTIMATION REPORT")
    lines.append("=" * 80)
    lines.append(f"\nModel: {model_name}")
    lines.append(f"Input Price per Token: ${input_price:.8f}")
    lines.append(f"Output Price per Token: ${output_price:.8f}")
    lines.append(f"Instruction Tokens (estimated): {instruction_tokens}")
    lines.append(f"Output Tokens (estimated): {output_tokens}")
    lines.append("\n" + "-" * 80)
    
    # Per-dataset breakdown
    total_input_cost = 0.0
    total_output_cost = 0.0
    
    for result in results:
        lines.append(f"\nDataset: {result['dataset']}")
        lines.append(f"  Number of Queries: {result['num_queries']:,}")
        lines.append(f"  Top-K: {result['top_k']:,}")
        lines.append(f"  Average Query Tokens: {result['avg_query_tokens']:.2f}")
        lines.append(f"  Average Document Tokens: {result['avg_doc_tokens']:.2f}")
        lines.append(f"  Tokens per Input: {result['tokens_per_input']:.2f}")
        lines.append(f"  Total Reranking Calls: {result['total_calls']:,}")
        lines.append(f"  Total Input Tokens: {result['total_input_tokens']:,.0f}")
        lines.append(f"  Total Output Tokens: {result['total_output_tokens']:,.0f}")
        lines.append(f"  Input Cost: ${result['input_cost']:,.2f}")
        lines.append(f"  Output Cost: ${result['output_cost']:,.2f}")
        lines.append(f"  Total Cost: ${result['total_cost']:,.2f}")
        
        total_input_cost += result['input_cost']
        total_output_cost += result['output_cost']
    
    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Input Cost: ${total_input_cost:,.2f}")
    lines.append(f"Total Output Cost: ${total_output_cost:,.2f}")
    lines.append(f"TOTAL COST: ${total_input_cost + total_output_cost:,.2f}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def format_curated_cost_report(
    results: List[Dict[str, float]],
    model_name: str,
    input_price: float,
    output_price: float,
    instruction_tokens: int,
    output_tokens: int,
    num_irrelevant: int,
    num_noise: int,
) -> str:
    """Format curated reranking cost report as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("CURATED RERANKING COST ESTIMATION REPORT")
    lines.append("=" * 80)
    lines.append(f"\nModel: {model_name}")
    lines.append(f"Input Price per Token: ${input_price:.8f}")
    lines.append(f"Output Price per Token: ${output_price:.8f}")
    lines.append(f"Instruction Tokens (estimated): {instruction_tokens}")
    lines.append(f"Output Tokens (estimated): {output_tokens}")
    lines.append(f"\nCandidate Set Composition:")
    lines.append(f"  Gold documents: all relevant docs per query")
    lines.append(f"  Irrelevant docs (lowest BM25 scores): {num_irrelevant}")
    lines.append(f"  Noise docs (highest BM25 non-gold): {num_noise}")
    lines.append(f"  Formula: candidate_size = gold_docs + {num_irrelevant} + {num_noise}")
    lines.append("\n" + "-" * 80)
    
    # Per-dataset breakdown
    total_input_cost = 0.0
    total_output_cost = 0.0
    
    for result in results:
        lines.append(f"\nDataset: {result['dataset']}")
        lines.append(f"  Number of Queries: {result['num_queries']:,}")
        lines.append(f"  Gold Docs per Query: avg={result['avg_gold_docs']:.1f}, "
                      f"min={result['min_gold_docs']}, max={result['max_gold_docs']}")
        lines.append(f"  Avg Candidate Set Size: {result['avg_candidate_size']:.1f}")
        lines.append(f"  Average Query Tokens: {result['avg_query_tokens']:.2f}")
        lines.append(f"  Average Document Tokens: {result['avg_doc_tokens']:.2f}")
        lines.append(f"  Tokens per Input: {result['tokens_per_input']:.2f}")
        lines.append(f"  Total Reranking Calls: {result['total_calls']:,}")
        lines.append(f"  Total Input Tokens: {result['total_input_tokens']:,.0f}")
        lines.append(f"  Total Output Tokens: {result['total_output_tokens']:,.0f}")
        lines.append(f"  Input Cost: ${result['input_cost']:,.2f}")
        lines.append(f"  Output Cost: ${result['output_cost']:,.2f}")
        lines.append(f"  Total Cost: ${result['total_cost']:,.2f}")
        
        total_input_cost += result['input_cost']
        total_output_cost += result['output_cost']
    
    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Input Cost: ${total_input_cost:,.2f}")
    lines.append(f"Total Output Cost: ${total_output_cost:,.2f}")
    lines.append(f"TOTAL COST: ${total_input_cost + total_output_cost:,.2f}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate costs for reranking experiments using OpenRouter API"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="OpenRouter model name (e.g., 'meta-llama/Llama-3.3-70B-Instruct')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top documents to rerank per query (default: 100). Ignored in --curated mode.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="cost_estimates.txt",
        help="Output file for cost report (default: cost_estimates.txt)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help=f"CSV file to save/update estimates (default: {DEFAULT_CSV_FILE.name})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Don't save to CSV file",
    )
    parser.add_argument(
        "--instruction-tokens",
        type=int,
        default=DEFAULT_INSTRUCTION_TOKENS,
        help=f"Estimated tokens for instruction prompt (default: {DEFAULT_INSTRUCTION_TOKENS})",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help=f"Estimated tokens for output (default: {DEFAULT_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (optional, can also use OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached computations before running",
    )
    # Curated reranking mode arguments
    parser.add_argument(
        "--curated",
        action="store_true",
        help="Use curated candidate set mode. Candidate set per query = gold_docs + "
             "irrelevant_docs + noise_docs. Only runs on QUEST and LIMIT+ datasets.",
    )
    parser.add_argument(
        "--num-irrelevant",
        type=int,
        default=DEFAULT_NUM_IRRELEVANT,
        help=f"Number of irrelevant docs (lowest BM25 scores) per query in curated mode "
             f"(default: {DEFAULT_NUM_IRRELEVANT})",
    )
    parser.add_argument(
        "--num-noise",
        type=int,
        default=DEFAULT_NUM_NOISE,
        help=f"Number of noise/semi-relevant docs (highest BM25 non-gold) per query in curated mode "
             f"(default: {DEFAULT_NUM_NOISE})",
    )
    parser.add_argument(
        "--curated-csv-file",
        type=str,
        default=None,
        help=f"CSV file for curated reranking estimates (default: {DEFAULT_CURATED_CSV_FILE.name})",
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    
    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...")
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
    
    # Fetch pricing
    print(f"Fetching pricing for model: {args.model_name}")
    input_price, output_price = get_openrouter_pricing(args.model_name, api_key)
    
    print(f"\n{'='*80}")
    print(f"PRICING INFORMATION")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Input Price per Token: ${input_price:.8f}")
    print(f"Output Price per Token: ${output_price:.8f}")
    
    if input_price == 0.0 and output_price == 0.0:
        print("\n  WARNING: Could not fetch pricing. Please verify:")
        print("  1. Model name is correct")
        print("  2. API key is set (--api-key or OPENROUTER_API_KEY env var)")
        print("  3. You have internet connection")
        print("\nContinuing with $0.00 pricing for estimation...")
    else:
        print(f"\n  Pricing successfully fetched from OpenRouter API")
    print(f"{'='*80}\n")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args.curated:
        # ---- CURATED RERANKING MODE ----
        print(f"{'='*80}")
        print(f"CURATED RERANKING MODE")
        print(f"Candidate set = gold_docs + {args.num_irrelevant} irrelevant + {args.num_noise} noise")
        print(f"Datasets: {', '.join(CURATED_DATASETS)}")
        print(f"{'='*80}\n")
        
        results = []
        
        for dataset_name in CURATED_DATASETS:
            if dataset_name not in DATASETS:
                print(f"Warning: Dataset '{dataset_name}' not in DATASETS config. Skipping.")
                continue
            
            paths = DATASETS[dataset_name]
            queries_path = paths["queries"]
            corpus_path = paths["corpus"]
            
            # Check if files exist
            if not queries_path.exists():
                print(f"Warning: Query file not found: {queries_path}")
                continue
            if not corpus_path.exists():
                print(f"Warning: Corpus file not found: {corpus_path}")
                continue
            
            quest_plus = dataset_name in ["QUEST+"]
            
            result = calculate_curated_dataset_costs(
                dataset_name=dataset_name,
                queries_path=queries_path,
                corpus_path=corpus_path,
                num_irrelevant=args.num_irrelevant,
                num_noise=args.num_noise,
                input_price=input_price,
                output_price=output_price,
                instruction_tokens=args.instruction_tokens,
                output_tokens=args.output_tokens,
                quest_plus=quest_plus,
            )
            results.append(result)
        
        # Generate curated report
        report = format_curated_cost_report(
            results=results,
            model_name=args.model_name,
            input_price=input_price,
            output_price=output_price,
            instruction_tokens=args.instruction_tokens,
            output_tokens=args.output_tokens,
            num_irrelevant=args.num_irrelevant,
            num_noise=args.num_noise,
        )
        
        # Print to console
        print("\n" + report)
        
        # Save to file
        output_path = Path(args.output_file)
        with open(output_path, "w") as f:
            f.write(report)
        
        # Save to CSV
        if not args.no_csv:
            csv_file = Path(args.curated_csv_file) if args.curated_csv_file else DEFAULT_CURATED_CSV_FILE
            save_curated_to_csv(
                results=results,
                model_name=args.model_name,
                num_irrelevant=args.num_irrelevant,
                num_noise=args.num_noise,
                input_price=input_price,
                output_price=output_price,
                instruction_tokens=args.instruction_tokens,
                output_tokens=args.output_tokens,
                csv_file=csv_file,
                timestamp=timestamp,
            )
        
        # Print summary
        total_cost = sum(r["total_cost"] for r in results)
        print(f"\n{'='*80}")
        print(f"QUICK SUMMARY (Curated Reranking)")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Input Price: ${input_price:.8f} per token")
        print(f"Output Price: ${output_price:.8f} per token")
        print(f"Candidate set: gold_docs + {args.num_irrelevant} irrelevant + {args.num_noise} noise")
        for result in results:
            print(f"  {result['dataset']}: {result['num_queries']} queries, "
                  f"avg {result['avg_candidate_size']:.1f} candidates/query, "
                  f"cost=${result['total_cost']:,.2f}")
        print(f"TOTAL ESTIMATED COST: ${total_cost:,.2f}")
        print(f"{'='*80}")
        print(f"\nReport saved to: {output_path.absolute()}")
        if not args.no_csv:
            csv_file = Path(args.curated_csv_file) if args.curated_csv_file else DEFAULT_CURATED_CSV_FILE
            print(f"Curated CSV file: {csv_file.absolute()}")
        print(f"Cache directory: {CACHE_DIR.absolute()}")
    
    else:
        # ---- STANDARD TOP-K MODE ----
        results = []
        
        for dataset_name, paths in DATASETS.items():
            queries_path = paths["queries"]
            corpus_path = paths["corpus"]
            
            # Check if files exist
            if not queries_path.exists():
                print(f"Warning: Query file not found: {queries_path}")
                continue
            if not corpus_path.exists():
                print(f"Warning: Corpus file not found: {corpus_path}")
                continue
            
            # Determine if QUEST+ format
            quest_plus = dataset_name in ["QUEST+"]
            
            result = calculate_dataset_costs(
                dataset_name=dataset_name,
                queries_path=queries_path,
                corpus_path=corpus_path,
                top_k=args.top_k,
                input_price=input_price,
                output_price=output_price,
                instruction_tokens=args.instruction_tokens,
                output_tokens=args.output_tokens,
                quest_plus=quest_plus,
            )
            results.append(result)
        
        # Generate report
        report = format_cost_report(
            results=results,
            model_name=args.model_name,
            input_price=input_price,
            output_price=output_price,
            instruction_tokens=args.instruction_tokens,
            output_tokens=args.output_tokens,
        )
        
        # Print to console
        print("\n" + report)
        
        # Save to file
        output_path = Path(args.output_file)
        with open(output_path, "w") as f:
            f.write(report)
        
        # Save to CSV
        if not args.no_csv:
            csv_file = Path(args.csv_file) if args.csv_file else DEFAULT_CSV_FILE
            save_to_csv(
                results=results,
                model_name=args.model_name,
                top_k=args.top_k,
                input_price=input_price,
                output_price=output_price,
                instruction_tokens=args.instruction_tokens,
                output_tokens=args.output_tokens,
                csv_file=csv_file,
                timestamp=timestamp,
            )
        
        # Print summary
        total_cost = sum(r["total_cost"] for r in results)
        print(f"\n{'='*80}")
        print(f"QUICK SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Input Price: ${input_price:.8f} per token")
        print(f"Output Price: ${output_price:.8f} per token")
        print(f"TOTAL ESTIMATED COST: ${total_cost:,.2f}")
        print(f"{'='*80}")
        print(f"\nReport saved to: {output_path.absolute()}")
        if not args.no_csv:
            csv_file = Path(args.csv_file) if args.csv_file else DEFAULT_CSV_FILE
            print(f"CSV file: {csv_file.absolute()}")
        print(f"Cache directory: {CACHE_DIR.absolute()}")
        print(f"\n  Tip: View CSV file to compare costs across different models and configurations")


if __name__ == "__main__":
    main()
