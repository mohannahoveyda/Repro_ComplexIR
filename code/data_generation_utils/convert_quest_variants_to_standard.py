#!/usr/bin/env python3
"""
Convert QUEST_w_Variants format to standard QUEST format expected by RSN codebase.

Input format (QUEST_w_Variants):
{
    "queries": [...],
    "operators": [...],
    "documents": [...],
    "nl_query": "...",
    "output": "...",
    "domain": "..."
}

Output format (standard QUEST):
{
    "query": "...",
    "docs": [...],
    "original_query": "...",
    "scores": null,
    "metadata": {
        "domain": "...",
        "queries": [...],
        "operators": [...],
        ...
    },
    "id": ...
}
"""

import json
import argparse
from typing import Dict, Any


def convert_example(variant_example: Dict[str, Any], example_id: int) -> Dict[str, Any]:
    """
    Convert a single example from QUEST_w_Variants format to standard format.
    
    Args:
        variant_example: Example in QUEST_w_Variants format
        example_id: Unique ID for the example
        
    Returns:
        Example in standard QUEST format
    """
    # Extract the main query (nl_query)
    query = variant_example.get("nl_query", "")
    
    # Extract documents (list of document IDs/titles)
    docs = variant_example.get("documents", [])
    
    # Create original_query from queries list if available
    queries_list = variant_example.get("queries", [])
    operators_list = variant_example.get("operators", [])
    
    # Build original_query by combining queries with operators
    if queries_list:
        if len(queries_list) == 1:
            original_query = queries_list[0]
        else:
            # Combine queries with operators
            original_parts = []
            for i, q in enumerate(queries_list):
                original_parts.append(f"<mark>{q}</mark>")
                if i < len(operators_list):
                    original_parts.append(f" {operators_list[i]} ")
            original_query = "".join(original_parts)
    else:
        original_query = query  # Fallback to nl_query if no queries list
    
    # Build metadata - include queries and operators as extra fields
    # These will be preserved in JSON but may be ignored by ExampleMetadata dataclass
    metadata = {
        "domain": variant_example.get("domain", None),
        "template": None,
        "fluency": None,
        "meaning": None,
        "naturalness": None,
        "relevance_ratings": None,
        "evidence_ratings": None,
        "attributions": None,
        "queries": queries_list,  # Store queries in metadata
        "operators": operators_list  # Store operators in metadata
    }
    
    # Create standard format example
    standard_example = {
        "query": query,
        "docs": docs,
        "original_query": original_query,
        "scores": None,
        "metadata": metadata,
        "id": example_id
    }
    
    return standard_example


def convert_file(input_path: str, output_path: str):
    """
    Convert entire JSONL file from QUEST_w_Variants format to standard format.
    
    Args:
        input_path: Path to input JSONL file in QUEST_w_Variants format
        output_path: Path to output JSONL file in standard format
    """
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile, start=1):
            if not line.strip():
                continue
                
            try:
                variant_example = json.loads(line)
                standard_example = convert_example(variant_example, idx)
                outfile.write(json.dumps(standard_example, ensure_ascii=False) + '\n')
                converted_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {idx}: {e}")
                continue
    
    print(f"Converted {converted_count} examples from {input_path} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QUEST_w_Variants format to standard QUEST format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file in QUEST_w_Variants format"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSONL file in standard QUEST format"
    )
    
    args = parser.parse_args()
    convert_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
