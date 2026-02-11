import json
import torch
import pandas as pd
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5


def get_prompt_component(path):
    file = pd.read_json(path, lines=True)
    q = file.get('query').tolist()
    d = file.get('docs').tolist() # These are the doc_ids (titles)
    return q, d

def get_content(path, doc_ids_list):
    df = pd.read_json(path, lines=True)
    doc_map = dict(zip(df.get('title'), df.get('text')))
    content = [[doc_map.get(doc) for doc in doc_list] for doc_list in doc_ids_list]
    return content

def run_monot5_rerank(queries, doc_ids, doc_texts, output_file):
    print(f"Initializing MonoT5 on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    reranker = MonoT5() 
  
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, q_text in enumerate(queries):
            ids = doc_ids[i]
            texts = doc_texts[i]
            
            gaggle_texts = [
                Text(text, {"title": ids[k]}, 0.0) 
                for k, text in enumerate(texts) if text is not None
            ]
            
            if not gaggle_texts:
                continue

            reranked = reranker.rerank(Query(q_text), gaggle_texts)
            
            scored_items = []
            for rank, item in enumerate(reranked, start=1):
                scored_items.append({
                    "rank": rank,
                    "score": float(item.score),
                    "title": item.metadata["title"]
                })
            
            output_node = {
                "query": q_text,
                "retrieved": scored_items
            }
            
            f.write(json.dumps(output_node) + '\n')
            f.flush()
            
            
            print(f"Finished Query {i+1}/{len(queries)}")
                

if __name__ == "__main__":
    OUTPUT_FILE = "results_reranked_quest_monot5.jsonl"
    
    q, d = get_prompt_component('quest_curated.jsonl')
    content = get_content('quest_docs.jsonl', d)

    run_monot5_rerank(q, d, content, OUTPUT_FILE)




