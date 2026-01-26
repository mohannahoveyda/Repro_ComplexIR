# Data Generation Utilities

This folder contains utilities for converting, preprocessing, and generating datasets.

## convert_quest_variants_to_standard.py

Converts QUEST_w_Variants format to the standard QUEST format expected by the RSN BM25 retriever.

### Usage

```bash
python convert_quest_variants_to_standard.py \
    ../../data/QUEST_w_Variants/data/quest_test_withVarients.jsonl \
    ../../data/QUEST_w_Variants/data/quest_test_withVarients_converted.jsonl
```

### Input Format (QUEST_w_Variants)

```json
{
    "queries": ["query1", "query2"],
    "operators": ["OR"],
    "documents": ["doc_id1", "doc_id2"],
    "nl_query": "Natural language query",
    "output": "...",
    "domain": "films"
}
```

### Output Format (Standard QUEST)

```json
{
    "query": "Natural language query",
    "docs": ["doc_id1", "doc_id2"],
    "original_query": "<mark>query1</mark> OR <mark>query2</mark>",
    "scores": null,
    "metadata": {
        "domain": "films",
        ...
    },
    "id": 1
}
```
