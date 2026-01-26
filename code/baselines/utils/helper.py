import json
import faiss
import time
import torch
import psutil
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

def queries(PATH, quest_plus = False):
    '''
        Functionality:    
            
            collect queries and relevant doc ids

        Input: 

            PATH - path to your queries file
            quest_plus - account for different keys in the original files

        Usage: 

            query, ground_truths = queries(QUERY_FILE, quest_plus)
    ''' 

    qS = []
    truth = []

    if quest_plus:
        q_key = "nl_query"
        t_key = "documents"
    else:
        q_key = "query"
        t_key = "docs"

    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qS.append(data.get(q_key))
            truth.append(data.get(t_key))

    return qS, truth

def documents(PATH, quest_plus = False):
    '''
        Functionality:
            
            collect document title, content and associated id

        Input: 

            PATH - path to your documents file
            quest_plus - account for different keys in the original files

        Usage: 

            doc_ids, doc_texts, doc_title_map = documents(CORPUS_FILE, quest_plus)
    ''' 
    
    ids = []
    texts = []
    title_map = {}
    t_key = "text"
    m_key = "title"

    with open(PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            str_id = item.get("idx") if quest_plus else str(idx)
            ids.append(str_id)
            texts.append(item.get(t_key))
            title_map[str_id] = item.get(m_key)
    
    return ids, texts, title_map

def start_retrieval(PATH, qS, truth, doc_ids, title_map, top_indices, top_scores):
     '''
        Functionality: 

            create a results file for later evaluation

        Input:

            PATH - Output file name
            qS - queries
            truth - relevant documents for queries
            doc_ids - doc identifier
            title_map - titles associated with 
            top_indices - top 100 indices
            top_scores - top 100 scores

        Usage:

            start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, top_indices, top_scores)
    '''

     with open(PATH, "w", encoding="utf-8") as f_out:
        for i, (q_text, relevant_titles) in enumerate(zip(qS, truth)):
            format = []
            indices = top_indices[i].tolist()
            scores = top_scores[i].tolist()

            for rank, (doc_idx, score) in enumerate(zip(indices, scores), start=1):
                rid = doc_ids[doc_idx]
                format.append({"rank": rank, "score": float(score), "title": title_map.get(str(rid))})
                
            output_obj = {"query": q_text, "relevant": relevant_titles, "retrieved": format}
            f_out.write(json.dumps(output_obj) + "\n")

def create_index(INDEX_NAME, query_emb, doc_emb):
    '''
        Functionality:
            
            create a faiss Flat index 
        
        Input:

            INDEX_NAME - name for index
            query_emb - embeddings of queries
            doc_emb - embeddings of documents
        
        Usage: 

            index = create_index(INDEX_NAME, query_emb, doc_emb)
    '''
    d_rep = doc_emb.astype('float32')
    q_rep = query_emb.astype('float32')

    faiss.normalize_L2(d_rep)
    faiss.normalize_L2(q_rep)

    dim = d_rep.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.write_index(index, INDEX_NAME)

    index.add(d_rep)

    return index

def search_index(index, query_emb, top_k= 100):
    '''
        Functionality:
            get scores and distances for queries

        Input: 

            index - faiss index i.e. faiss.read_index()
            query_emb - embedding for queries
            top_k - retrieve top k documents (set to 100)

        Usage:

            scores, indices = search_index(index, query_emb)
    '''
    q_rep = query_emb.astype('float32')
    faiss.normalize_L2(q_rep)

    scores, indices = index.search(q_rep, top_k)

    return scores, indices

@contextmanager
def benchmark(model_name, step_name, log_path="benchmarks.csv"):
    """
    Functionality:

        Context manager to benchmark a block of code.
    
    Input:

        model_name - name of your model
        step_name - name of your current step in the pipeline e.g. embedding, search
        log_path - name for file where we store results
    
    Usage:

        with benchmark(MODEL_NAME, STEP_NAME):
            embeddings = model.encode(texts)
    """
    start_time = time.perf_counter()
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 ** 2) # MB
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    yield # the wrapped code runs here
    
    duration = time.perf_counter() - start_time
    end_mem = process.memory_info().rss / (1024 ** 2)
    ram_used = end_mem - start_mem
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        gpu_mem = 0 

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "step": step_name,
        "duration (seconds)": duration,
        "cpu_usage (MB)": ram_used,
        "gpu_usage (MB)": gpu_mem
    }
    
    df = pd.DataFrame([log_entry])
    df.to_csv(log_path, mode='a', index=False, header=not pd.io.common.file_exists(log_path))


    print("#"*50)
    print("#"*50)
    print()
    print(f"--- {step_name} Done | Time: {duration:.2f}s | RAM: {ram_used:.1f}MB ---\n")
    print()
    print("#"*50)
    print("#"*50)