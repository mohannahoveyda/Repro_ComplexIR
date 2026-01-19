import json
import faiss
import torch

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


