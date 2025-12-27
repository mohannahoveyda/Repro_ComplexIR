import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import chromadb
from sentence_transformers import SentenceTransformer
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# need to install bitsandbytes, accelerate and flash_attn if we want to use quantization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_docs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaDB:
    def __init__(self, db_path):
        start_time = time.time()
        logger.info(f"Initializing ChromaDB at path: {db_path}")

        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.documents_collection = self.client.get_or_create_collection("documents")

        logger.info(f"ChromaDB initialized successfully in {time.time() - start_time:.2f} seconds")

    def store_documents(self, docs_src, model, tokenizer, batch_size=1):
        logger.info(f"Starting document storage process from: {docs_src}")
        logger.info(f"Using batch size: {batch_size}")

        total_docs = 0
        total_batches = 0
        start_time = time.time()
        batch_times = []

        current_ids = []
        current_texts = []
        current_metas = []

        with open(docs_src, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    
                    current_ids.append(row["title"]) # changes needed for LIMIT+ "_id"
                    current_texts.append(row["text"])
                    current_metas.append({"type": "document"})

                    if len(current_ids) >= batch_size:
                        
                        self.process_batch(current_ids, current_texts, current_metas, model, total_batches + 1, tokenizer)
                        
                        
                        batch_time = time.time() - start_time 
                        batch_times.append(batch_time) 
                        
                        total_docs += len(current_ids)
                        total_batches += 1
                        
                        current_ids = []
                        current_texts = []
                        current_metas = []
                        
                except json.JSONDecodeError:
                    logger.warning("Skipped a malformed JSON line.")
                    continue

            if current_ids:
                self.process_batch(current_ids, current_texts, current_metas, model, total_batches + 1, tokenizer)
                total_docs += len(current_ids)
                total_batches += 1

        total_time = time.time() - start_time
        logger.info(f"======================================================")
        logger.info(f"STORAGE COMPLETED! Total docs: {total_docs}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info(f"======================================================")

    def process_batch(self, ids, texts, metas, model, batch_number, tokenizer):

        chunked_texts = []
        chunked_ids = []
        chunked_metas = []
        
        # creates chunks of the documents with some overlap
        MAX_WINDOW = 2048 
        OVERLAP = 200 

        for doc_id, text, meta in zip(ids, texts, metas):
            
            if len(text) < (MAX_WINDOW * 3): 
                chunked_texts.append(text)
                chunked_ids.append(doc_id) 
                chunked_metas.append(meta)
            
            else:
                window_chars = MAX_WINDOW * 4
                overlap_chars = OVERLAP * 4
                
                start = 0
                part_num = 0
                
                while start < len(text):
                    end = min(start + window_chars, len(text))
                    chunk_text = text[start:end]
                    
                    chunked_texts.append(chunk_text)
                    chunked_ids.append(f"{doc_id}_part{part_num}")
                    
                    chunk_meta = meta.copy()
                    chunk_meta["parent_id"] = doc_id
                    chunk_meta["chunk_index"] = part_num
                    chunked_metas.append(chunk_meta)
                    
                    start += (window_chars - overlap_chars)
                    part_num += 1

        logger.info(f"Batch {batch_number}: Split {len(texts)} docs into {len(chunked_texts)} chunks")

        embed_start = time.time()

        if tokenizer:
            # This is not tested properly, as I was running into memory issues last time
            # as stated in the email, this follows the following approach: https://huggingface.co/intfloat/e5-mistral-7b-instruct/discussions/30
            # but it requires flash attention to be installed https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention
            # I ran into subprocess issues related to the versioning of flash_attn and couldn't make it run, similar to here https://stackoverflow.com/questions/79179992/flash-attention-flash-attn-package-fails-to-build-wheel-in-google-colab-due-to
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(chunked_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device) 
                outputs = model(**inputs)

                # get attention mask and token embeddings
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # ignore padding and prepare for mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # normalization
                embeddings = sum_embeddings / sum_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.cpu().numpy()
        else:
            embeddings = model.encode(chunked_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)

        embed_time = time.time() - embed_start

        logger.info(f"Batch {batch_number}: Embedded {len(texts)} docs in {embed_time:.2f}s")

        store_start = time.time()
        self.documents_collection.add(
            ids=chunked_ids,
            embeddings=embeddings,
            documents=chunked_texts,
            metadatas=chunked_metas
        )
        store_time = time.time() - store_start
        logger.info(f"Batch {batch_number}: Stored in {store_time:.2f}s")

        del embeddings
        gc.collect()

def main():
    logger.info("=" * 60)
    logger.info("STARTING EMBEDDING GENERATION PROCESS")
    logger.info("=" * 60)

    # Set this to True for E5 Mistral
    E5MISTRAL = False

    # Set this to True for Reducing Model Size
    REDUCED = False

    tokenizer = None

    device = 'cuda'
    
    model_load_start = time.time()

    if E5MISTRAL:
        logger.info(f"Loading E5 Mistral 7B Instruct Model on {device}")
        if REDUCED:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

            tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
            model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct',torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda", quantization_config=quantization_config)
        else:
            model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    else:
        logger.info(f"Loading QWEN 0.6B Model on {device}")
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        
    model_load_time = time.time() - model_load_start

    logger.info(f"Model loaded successfully in {model_load_time:.2f} seconds")
    logger.info(f"Model device: {model.device}")

    # Change this to your desired destination / location of dataset
    DB_path = "./../../scratch/g8/EXPERIMENTAL_Q"
    DOCS = "./Dataset/documents.jsonl"

    logger.info("Initializing ChromaDB...")
    db_init_start = time.time()
    DB = ChromaDB(DB_path)
    db_init_time = time.time() - db_init_start
    logger.info(f"ChromaDB initialized in {db_init_time:.2f} seconds")

    logger.info("Starting document storage process...")
    DB.store_documents(DOCS, model, tokenizer)
    logger.info("All documents processed successfully!")

    logger.info("=" * 60)
    logger.info("PROCESS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
