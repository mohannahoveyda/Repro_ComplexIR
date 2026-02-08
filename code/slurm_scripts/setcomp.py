import utils.helper as tools
import torch
from sentence_transformers import SparseEncoder

def main(quest_plus=False):

    MODEL_NAME = "naver/splade-v3"
    INDEX_NAME = "SETCOMP-QUEST"

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
    print(f"Total documents loaded: {len(doc_ids)}")

    print("Collecting queries...")
    qS_list, templates, ground_truths = tools.setcomp_queries(QUERY_FILE, quest_plus)

    print(f"Loading model: {MODEL_NAME}...")
    model = SparseEncoder(MODEL_NAME)

    print("Creating Embeddings...")
    with tools.benchmark(MODEL_NAME, "Embedding"):
        retriever = tools.create_sparse_index(INDEX_NAME, device, doc_texts, MODEL_NAME)
    
        
    with tools.benchmark(MODEL_NAME, "LAO-Search"):
        scores, indices = tools.retrieve_compositional(model, retriever, qS_list, templates, top_k=100)

    tools.start_retrieval(OUTPUT_FILE, qS_list, ground_truths, doc_ids, doc_title_map, indices, scores)


if __name__ == "__main__":
    quest_plus = False
    main(quest_plus)