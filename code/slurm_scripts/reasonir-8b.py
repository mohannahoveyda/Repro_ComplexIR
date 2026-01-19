import utils.helper as tools
from sentence_transformers import SentenceTransformer

def main(quest_plus=False):
    MODEL_NAME = "reasonir/ReasonIR-8B"
    INDEX_NAME = "ReasonIR-8B-QUEST"

    if quest_plus:
        QUERY_FILE = "./Data_New/quest_test_withVarients.jsonl"
        CORPUS_FILE = "./Data_New/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = "results_plus.jsonl"
    else:
        QUERY_FILE = "./Dataset/quest_test.jsonl"
        CORPUS_FILE = "./Dataset/quest_docs.jsonl"
        OUTPUT_FILE = "results.jsonl"
    
    print("Collecting documents...")
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, quest_plus)
    print(f"Total documents loaded: {len(doc_ids)}")

    print("Collecting queries...")
    query, ground_truths = tools.queries(QUERY_FILE, quest_plus)
    print(f"Total queries loaded: {len(query)}")

    print(f"Loading model: {MODEL_NAME}...")
    model_kwargs = {"torch_dtype": "auto"}
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, model_kwargs=model_kwargs)

    print("Creating Embeddings...")
    instruction = "Given a web search query, retrieve the most relevant documents"

    with tools.benchmark(MODEL_NAME, "Embedding"):
        doc_emb = model.encode(doc_texts, prompt="", show_progress_bar=True)
        query_emb = model.encode(query, prompt=instruction, show_progress_bar=True)
    
    index = tools.create_index(INDEX_NAME, query_emb, doc_emb)

    scores, indices = tools.search_index(index, query_emb)

    tools.start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, indices, scores)
 

if __name__ == "__main__":
    quest_plus = False
    main(quest_plus)