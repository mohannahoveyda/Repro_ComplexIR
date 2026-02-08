from src.contriever import Contriever
from transformers import AutoTokenizer
import sys
import os
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.dirname(script_dir)
if baselines_dir not in sys.path:
    sys.path.insert(0, baselines_dir)

from utils import helper as utils
import numpy as np
import torch


def embed_texts(texts, model, tokenizer, device, batch_size=64):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            embs = model(**inputs)

            embs = embs.detach().float().cpu().numpy()
            all_embs.append(embs)

    return np.vstack(all_embs)


def main(plus, limit):
    if not limit and plus:  # QUEST_VAR
        query_path = "./data/QUEST_VAR/quest_test_withVarients.jsonl"
        doc_path = "./data/QUEST_VAR/quest_text_w_id_withVarients.jsonl"
        output_file = "contriever_QUEST_VAR_r01"
        index_name = "contriever_QUEST_VAR_index"
    elif not limit and not plus:  # QUEST
        query_path = "./data/QUEST/test.jsonl"
        doc_path = "./data/QUEST/documents.jsonl"
        output_file = "contriever_QUEST_r01"
        index_name = "contriever_QUEST_index"
    elif limit and not plus:  # LIMIT
        query_path = "./data/LIMIT/queries.jsonl"
        qrels_path = "./data/LIMIT/qrels.jsonl"
        doc_path = "./data/LIMIT/corpus.jsonl"
        output_file = "contriever_LIMIT_r01"
        index_name = "contriever_LIMIT_index"
    elif limit and plus:  # LIMIT+
        query_path = "./data/LIMIT+/limit_quest_queries.jsonl"
        qrels_path = "./data/LIMIT/qrels.jsonl"
        doc_path = "./data/LIMIT+/corpus.jsonl"
        output_file = "contriever_LIMIT+_r01"
        index_name = "contriever_LIMIT+_index"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading documents...")
    if not limit:
        doc_ids, doc_texts, doc_titles = utils.documents(doc_path, quest_plus=plus)
    else:
        doc_ids, doc_texts, doc_titles = utils.documents_limit(doc_path, plus=plus)
    print(f"Loaded {len(doc_ids)} documents!")

    print("Loading queries...")
    if not limit:
        queries, truths = utils.queries(query_path, quest_plus=plus)
    else:
        queries, truths = utils.queries_limit(query_path, qrels_path, plus=plus)
    print(f"Loaded {len(queries)} queries!")

    print("Loading Contriever model and tokenizer...")
    model = Contriever.from_pretrained("facebook/contriever").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    with utils.benchmark('contriever', "Embedding"):
        print("Embedding documents...")
        doc_embeddings = embed_texts(doc_texts, model, tokenizer, device)
        print("Embedding queries...")
        query_embeddings = embed_texts(queries, model, tokenizer, device)

    print("Creating index...")
    index = utils.create_index(index_name, query_embeddings, doc_embeddings)

    print("Calculating scores...")
    scores, indices = utils.search_index(index, query_embeddings)

    utils.start_retrieval(output_file, queries, truths, doc_ids, doc_titles, indices, scores)


if __name__ == "__main__":
    plus = False  # Set to True for QUEST_VAR and LIMIT+
    limit = True  # Set to True for LIMIT and LIMIT+
    main(plus, limit)
