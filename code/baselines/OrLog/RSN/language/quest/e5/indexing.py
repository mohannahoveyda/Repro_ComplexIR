import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
print(faiss.__version__)

path = "/home/mhoveyda1/REASON/datasets/QUEST/tmp/documents.jsonl"

documents = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        documents.append(obj["text"]) # Ignored the title as QUEST authors did the same for BM25
        # documents.append(f"{obj.get('title','')} — {obj.get('text','')}")

print(f"Loaded {len(documents)} documents.")


model = SentenceTransformer("intfloat/e5-base-v2")

embs = model.encode(
    documents,
    batch_size=512,           
    convert_to_numpy=True,
    show_progress_bar=True    
)

faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
print(f"FAISS index contains {index.ntotal} vectors.")

faiss.write_index(index, "quest_e5_base_v2.index")

def retrieve(query, k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "id": idx,
            "score": float(score),
            "text": documents[idx][:200].replace("\n", " ") + "…"
        })
    return results

if __name__ == "__main__":
    q = "What is the film directed by Kleber Mendonça Filho?"
    for res in retrieve(q, k=5):
        print(f"→ doc #{res['id']} (score={res['score']:.4f}): {res['text']}")


