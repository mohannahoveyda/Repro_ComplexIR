import argparse
import os
import sys
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.dirname(script_dir)
if baselines_dir not in sys.path:
    sys.path.insert(0, baselines_dir)

import utils.helper as tools
from gritlm import GritLM


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def main(
    quest_plus=False,
    query_file=None,
    corpus_file=None,
    output_file=None,
    model_name=None,
    index_name=None,
    device=None,
    batch_size=32,
    max_docs=None,
    max_queries=None,
):
    MODEL_NAME = model_name if model_name else "GritLM/GritLM-7B"
    INDEX_NAME = index_name if index_name else "GritLM-7B-QUEST"

    if quest_plus:
        QUERY_FILE = query_file if query_file else "./data/QUEST_w_Varients/quest_test_withVarients.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST_w_Varients/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = output_file if output_file else "results_plus.jsonl"
    else:
        QUERY_FILE = query_file if query_file else "./data/QUEST/test_id_added.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST/documents.jsonl"
        OUTPUT_FILE = output_file if output_file else "results.jsonl"

    print(f"CUDA available: {torch.cuda.is_available()}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    print(f"Using device: {device}")

    print("Collecting documents...")
    print(f"Corpus file: {CORPUS_FILE}")
    print(f"Query file: {QUERY_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, quest_plus)
    print(f"Total documents loaded: {len(doc_ids)}")

    if max_docs is not None and max_docs > 0 and max_docs < len(doc_ids):
        print(f"Limiting to first {max_docs} documents")
        doc_ids = doc_ids[:max_docs]
        doc_texts = doc_texts[:max_docs]
        doc_title_map = {k: doc_title_map.get(k) for k in doc_ids}

    print("Collecting queries...")
    query, ground_truths = tools.queries(QUERY_FILE, quest_plus)
    print(f"Total queries loaded: {len(query)}")

    if max_queries is not None and max_queries > 0 and max_queries < len(query):
        print(f"Limiting to first {max_queries} queries")
        query = query[:max_queries]
        ground_truths = ground_truths[:max_queries]

    print(f"Loading model: {MODEL_NAME}...")
    model = GritLM(MODEL_NAME, torch_dtype="auto", mode="embedding")

    print("Creating embeddings...")
    instruction = "Given a web search query, retrieve the most relevant documents"

    def _to_numpy(emb):
        if hasattr(emb, "cpu"):
            return emb.cpu().numpy()
        return np.asarray(emb)

    with tools.benchmark(MODEL_NAME, "Embedding"):
        # Encode documents in batches to avoid OOM on large corpora
        doc_emb_list = []
        for start in range(0, len(doc_texts), batch_size):
            end = min(start + batch_size, len(doc_texts))
            batch_texts = doc_texts[start:end]
            emb = model.encode(batch_texts, instruction=gritlm_instruction(""))
            doc_emb_list.append(_to_numpy(emb))
        doc_emb = np.vstack(doc_emb_list)

        query_emb = _to_numpy(model.encode(query, instruction=gritlm_instruction(instruction)))

    index = tools.create_index(INDEX_NAME, query_emb, doc_emb)
    scores, indices = tools.search_index(index, query_emb)

    tools.start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, indices, scores)
    print(f"Results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GritLM-7B for information retrieval (embedding + FAISS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gritlm-7b.py
  python gritlm-7b.py --quest-plus
  python gritlm-7b.py --query-file ./data/QUEST/test_id_added.jsonl --corpus-file ./data/QUEST/documents.jsonl --output-file results.jsonl
  python gritlm-7b.py --max-docs 1000 --max-queries 50
        """,
    )
    parser.add_argument("--quest-plus", action="store_true", help="Use quest_plus data paths")
    parser.add_argument("--query-file", type=str, default=None, help="Path to query JSONL file")
    parser.add_argument("--corpus-file", type=str, default=None, help="Path to corpus JSONL file")
    parser.add_argument("--output-file", type=str, default=None, help="Path to output results JSONL")
    parser.add_argument("--model-name", type=str, default=None, help="HuggingFace model name (default: GritLM/GritLM-7B)")
    parser.add_argument("--index-name", type=str, default=None, help="FAISS index path/name (default: GritLM-7B-QUEST)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit to first N documents (debug/small runs)")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit to first N queries (debug/small runs)")
    args = parser.parse_args()

    main(
        quest_plus=args.quest_plus,
        query_file=args.query_file,
        corpus_file=args.corpus_file,
        output_file=args.output_file,
        model_name=args.model_name,
        index_name=args.index_name,
        device=args.device,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )
