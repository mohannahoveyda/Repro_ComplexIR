#!/usr/bin/env python
"""
Preprocess QUEST test.jsonl to create a TREC qrels (gold) file.

Reads:
    QUEST test.jsonl file (JSONL format, with or without "id" field)

Writes:
    TREC qrels file with format: qid 0 docid relevance
    
    Where:
    - qid: unique query ID (from "id" field if present, otherwise assigned sequentially)
    - 0: unused field (standard TREC format)
    - docid: relevant document title
    - relevance: 1 (all docs in gold file are relevant)
    
    If input file doesn't have "id" fields, creates an ID-augmented version first.
"""

import argparse
import json
import os
from typing import Dict, List, Optional


def check_has_ids(input_path: str) -> bool:
    """
    Check if the input file has "id" fields.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        True if file has "id" fields, False otherwise
    """
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                return "id" in rec
            except json.JSONDecodeError:
                continue
    return False


def create_id_augmented_file(input_path: str, output_path: str):
    """
    Create an ID-augmented version of the input file.
    
    Args:
        input_path: Path to input JSONL file (without IDs)
        output_path: Path to output JSONL file (with IDs added)
    """
    qid_counter = 1
    
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                rec = json.loads(line)
                # Add ID field
                rec["id"] = qid_counter
                qid_counter += 1
                
                # Write augmented record
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue
    
    print(f"[preprocess] Created ID-augmented file: {output_path}")


def preprocess_qrels(input_path: str, output_path: str):
    """
    Convert JSONL file to TREC qrels format.
    
    Args:
        input_path: Path to input JSONL file (must have "id" field)
        output_path: Path to output TREC qrels file
    """
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                rec = json.loads(line)
                qid = rec.get("id")
                docs = rec.get("docs", [])
                
                if qid is None:
                    continue
                
                # Write one line per relevant document
                for doc in docs:
                    if doc:  # Skip empty doc names
                        # TREC qrels format: qid 0 docid relevance
                        outfile.write(f"{qid} 0 {doc} 1\n")
            except json.JSONDecodeError:
                continue
    
    print(f"[preprocess] Wrote TREC qrels file to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL file to TREC qrels format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file (with or without 'id' field)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output TREC qrels file (default: input path with _qrels suffix)",
    )
    parser.add_argument(
        "--id_augmented_output",
        type=str,
        default=None,
        help="Path to ID-augmented JSONL file if IDs need to be added (default: input path with _id_added suffix)",
    )
    args = parser.parse_args()
    
    # Check if input file has IDs
    has_ids = check_has_ids(args.input)
    
    # If no IDs, create ID-augmented version
    if not has_ids:
        if args.id_augmented_output is None:
            base_path = os.path.splitext(args.input)[0]
            args.id_augmented_output = f"{base_path}_id_added.jsonl"
        
        # Check if ID-augmented file already exists
        if not os.path.exists(args.id_augmented_output):
            print(f"[preprocess] Input file doesn't have 'id' fields. Creating ID-augmented version...")
            create_id_augmented_file(args.input, args.id_augmented_output)
        else:
            print(f"[preprocess] Using existing ID-augmented file: {args.id_augmented_output}")
        
        # Use ID-augmented file for qrels creation
        input_file_for_qrels = args.id_augmented_output
    else:
        print(f"[preprocess] Input file already has 'id' fields. Using it directly.")
        input_file_for_qrels = args.input
    
    # If no output path provided, create one with _qrels suffix
    if args.output is None:
        base_path = os.path.splitext(args.input)[0]
        args.output = f"{base_path}_qrels"
    
    preprocess_qrels(input_file_for_qrels, args.output)
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()

