import utils.helper as tools
import torch
from collections import defaultdict

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from __future__ import annotations


def read_trec_run(path: str):
    """
    Reads TREC run: qid Q0 docid rank score tag
    docid may contain spaces. We parse from the right.
    Returns: dict[qid] -> list[(docid, score, tag)]
    """
    by_qid = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split from right into: docid, rank, score, tag
            try:
                left, rank, score, tag = line.rsplit(" ", 3)
                qid, _q0, docid = left.split(" ", 2)
            except ValueError:
                continue
            by_qid[qid].append((docid, float(score)))
    return by_qid

if __name__ == "__main__":
    path = 0
    read_trec_run(path)


def main(quest_plus=False):

    BM25 = "./bm25_top100.txt"

    if quest_plus:
        QUERY_FILE = "./Data_New/quest_test_withVarients.jsonl"
        CORPUS_FILE = "./Data_New/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = "results_plus.jsonl"
    else:
        QUERY_FILE = "./Dataset/quest_test.jsonl"
        CORPUS_FILE = "./Dataset/quest_docs.jsonl"
        OUTPUT_FILE = "results.jsonl"

    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Collecting documents...")
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, quest_plus)

    docid_to_text = {doc_id: doc_texts[i] for i, doc_id in enumerate(doc_ids)}
    title_to_text = {}
    for doc_id, title in doc_title_map.items():
        if title:
            idx = doc_ids.index(doc_id)
            title_to_text[title] = doc_texts[idx]

    print("Collecting queries...")
    qS_list, _ = tools.queries(QUERY_FILE, quest_plus)

    qid_to_query = {}
    for idx, qtext in enumerate(qS_list):
        qid_to_query[str(idx)] = qtext

    print("Loading BM25 run...")
    by_qid = read_trec_run(BM25)

    print("Initializing MonoT5...")
    reranker = MonoT5()

    print("Reranking...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for qid, hits in by_qid.items():
            qtext = qid_to_query.get(qid)
            if not qtext:
                continue

            texts = []
            for docid, _, _ in hits:
                doc_text = docid_to_text.get(docid) or title_to_text.get(docid)
                if not doc_text:
                    continue
                texts.append(Text(doc_text, {"docid": docid}, 0.0))

            if not texts:
                continue

            reranked = reranker.rerank(Query(qtext), texts)

            for rank, item in enumerate(reranked, start=1):
                docid = item.metadata["docid"]
                score = float(item.score)
                out_f.write(f"{qid} Q0 {docid} {rank} {score:.6f} monot5\n")

if __name__ == "__main__":
    main()

