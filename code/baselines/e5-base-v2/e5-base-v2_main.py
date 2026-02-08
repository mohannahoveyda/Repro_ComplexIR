#!/usr/bin/env python
"""
E5-base-v2 baseline: end-to-end retrieval on GPU (Snellius).

Full implementation aligned with reasonir-8b and gritlm-7b_main: CLI, cache,
sharded cache, multi-GPU, FAISS/direct similarity, tools.documents/tools.queries,
and tools.start_retrieval().

E5-base-v2 uses SentenceTransformer with "query: " and "passage: " prefixes.
"""

import argparse
import hashlib
import os
import sys
import math
import multiprocessing as mp
from multiprocessing import Process, Queue
import tempfile

import torch
import numpy as np
from tqdm import tqdm

# Set multiprocessing start method to 'spawn' for CUDA isolation
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.dirname(script_dir)
if baselines_dir not in sys.path:
    sys.path.insert(0, baselines_dir)

import utils.helper as tools
from sentence_transformers import SentenceTransformer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

# Try to import optional optimizations
try:
    from torch.nn import DataParallel
    HAS_DATAPARALLEL = True
except ImportError:
    HAS_DATAPARALLEL = False

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def e5_format_query(query_text):
    """Format query text with E5-base-v2 query prefix."""
    return f"query: {query_text}"


def e5_format_passage(passage_text):
    """Format passage text with E5-base-v2 passage prefix."""
    return f"passage: {passage_text}"


def content_hash(texts):
    """Compute a short content-based hash for a list of texts (for cache keys)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:24]


def inspect_batch_samples(texts, batch_size, name="items", max_samples=2):
    """Minimal function to inspect what's in each batch. Enable with INSPECT_BATCHES=1"""
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    if inspect_env != "1":
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting {name} batches (showing first {max_samples} batches)")
    print(f"{'='*60}")
    for batch_idx in range(min(max_samples, (len(texts) + batch_size - 1) // batch_size)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1} (indices {start_idx}-{end_idx-1}, {len(batch_texts)} items):")
        for i, text in enumerate(batch_texts[:3]):
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"  [{i}] {preview}")
        if len(batch_texts) > 3:
            print(f"  ... and {len(batch_texts) - 3} more items in this batch")
    print(f"{'='*60}\n")


def auto_tune_batch_size(model, texts, device, max_length, start_batch=32, max_batch=512):
    """Automatically find optimal batch size based on GPU memory."""
    if device != "cuda":
        return start_batch
    
    model.eval()
    optimal_batch = start_batch
    
    test_sizes = [start_batch]
    if start_batch * 2 <= max_batch:
        test_sizes.append(start_batch * 2)
    if start_batch * 4 <= max_batch:
        test_sizes.append(start_batch * 4)
    if start_batch * 8 <= max_batch:
        test_sizes.append(start_batch * 8)
    
    for test_batch in test_sizes:
        try:
            torch.cuda.empty_cache()
            test_size = min(test_batch, len(texts), 100)
            test_texts = texts[:test_size]
            with torch.inference_mode():
                _ = model.encode(
                    test_texts,
                    batch_size=test_batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            optimal_batch = test_batch
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                print(f"Warning during batch size test: {e}")
                break
    
    return optimal_batch


def encode_worker(gpu_id, model_name, texts_chunk, batch_size, max_length, result_queue, temp_dir):
    """Worker function for multiprocessing: loads SentenceTransformer on specific GPU and encodes texts.
    
    This runs in a separate process with its own CUDA context, preventing memory conflicts.
    Each process only sees its assigned GPU, ensuring complete isolation.
    """
    try:
        # Set CUDA device for this process BEFORE any CUDA operations
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Reinitialize CUDA in this process
        import torch
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        
        # Import here to ensure each process has its own imports
        from sentence_transformers import SentenceTransformer
        
        print(f"[GPU {gpu_id} Process] Loading E5-base-v2 on device cuda:0 (maps to physical GPU {gpu_id})...")
        
        # Load model
        model = SentenceTransformer(model_name, device="cuda:0")
        model.max_seq_length = max_length
        model.eval()
        
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[GPU {gpu_id} Process] Model loaded (memory: {memory_used:.2f} GB)")
        
        # Encode texts in batches
        import numpy as np
        n_batches = (len(texts_chunk) + batch_size - 1) // batch_size
        print(f"[GPU {gpu_id} Process] Encoding {len(texts_chunk)} texts in {n_batches} batch(es)...")
        emb_list = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(texts_chunk), batch_size),
                desc=f"[GPU {gpu_id}] Batches",
                total=n_batches,
                unit="batch",
                file=sys.stderr,
            ):
                batch_texts = texts_chunk[start : start + batch_size]
                emb = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                emb_list.append(emb)
        embeddings = np.concatenate(emb_list, axis=0)
        
        print(f"[GPU {gpu_id} Process] Encoded -> shape {embeddings.shape}")
        
        # Save embeddings to a temporary file
        os.makedirs(temp_dir, exist_ok=True)
        shard_path = os.path.join(temp_dir, f"gpu_{gpu_id}.npy")
        np.save(shard_path, embeddings.astype(np.float32))
        
        # Put only metadata (gpu_id, filename) in the queue
        result_queue.put((gpu_id, shard_path))
        
        # Cleanup
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        import traceback
        error_msg = f"[GPU {gpu_id} Process] ERROR: {e}\n{traceback.format_exc()}"
        print(error_msg)
        result_queue.put((gpu_id, None))
        raise


def encode_with_multigpu(model_name, texts, batch_size, max_length, num_gpus):
    """Encode texts using multiple GPUs by splitting work across GPUs (data parallelism).
    
    This splits the texts across available GPUs and encodes them sequentially on each GPU.
    Each GPU gets its own separate model instance loaded independently.
    """
    if num_gpus <= 1:
        torch.cuda.set_device(0)
        single_model = SentenceTransformer(model_name, device="cuda:0")
        single_model.max_seq_length = max_length
        single_model.eval()
        with torch.inference_mode():
            result = single_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        del single_model
        torch.cuda.empty_cache()
        return np.asarray(result, dtype=np.float32)
    
    print(f"Splitting {len(texts)} texts across {num_gpus} GPUs for PARALLEL encoding...")
    
    # Split texts across GPUs for data parallelism
    chunk_size = math.ceil(len(texts) / num_gpus)
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Ensure we have exactly num_gpus chunks
    while len(chunks) < num_gpus:
        chunks.append([])
    
    # Use multiprocessing for true parallelism
    print(f"Starting {num_gpus} parallel processes (one per GPU)...")
    
    result_queue = Queue()
    temp_dir = tempfile.mkdtemp(prefix="e5_base_v2_mgpu_")
    processes = []
    
    # Start a process for each GPU
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
        
        print(f"  Starting process for GPU {gpu_id} ({len(chunks[gpu_id])} texts)...")
        p = Process(
            target=encode_worker,
            args=(gpu_id, model_name, chunks[gpu_id], batch_size, max_length, result_queue, temp_dir)
        )
        p.start()
        processes.append((gpu_id, p))
    
    # Collect results from all processes
    results_dict = {}
    for _ in range(len(processes)):
        gpu_id_result, shard_path = result_queue.get()
        if shard_path is not None:
            results_dict[gpu_id_result] = shard_path
            print(f"  GPU {gpu_id_result}: Wrote embeddings to {shard_path}")
        else:
            raise RuntimeError(f"GPU {gpu_id_result} process failed!")
    for gpu_id, p in processes:
        p.join()
        print(f"  GPU {gpu_id}: Process completed")
    
    # Concatenate results in order
    if results_dict:
        loaded = []
        for i in range(num_gpus):
            if i in results_dict:
                shard_path = results_dict[i]
                emb = np.load(shard_path, allow_pickle=True)
                loaded.append(emb)
        if loaded:
            final_emb = np.concatenate(loaded, axis=0)
            print(f"Combined embeddings from {len(loaded)} GPUs: {final_emb.shape}")
        else:
            final_emb = np.array([]).astype(np.float32)
        return final_emb
    else:
        return np.array([]).astype(np.float32)


def main(quest_plus=False, query_file=None, corpus_file=None, output_file=None, 
         model_name=None, index_name=None, device=None, batch_size=32, doc_batch_size=None, 
         use_fp16=True, use_cache=True, cache_dir=None, use_faiss=True, max_length=512, 
         auto_batch=False, use_multigpu=False, max_docs=None, max_queries=None,
         task_suffix=None, query_format=None, corpus_format=None):
    # Debug: Check INSPECT_BATCHES environment variable early
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    print(f"[DEBUG] At start of main(), INSPECT_BATCHES = '{inspect_env}'")

    MODEL_NAME = model_name if model_name else "intfloat/e5-base-v2"
    INDEX_NAME = index_name if index_name else "E5-base-v2-QUEST"
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_file) if output_file else ".", "cache") if use_cache else None
    task_name = ("quest_plus" if quest_plus else "quest") + (task_suffix or "")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"PyTorch detected {num_gpus} GPU(s), use_multigpu={use_multigpu}")
    if use_multigpu and num_gpus > 1:
        print(f"Multi-GPU mode: Using {num_gpus} GPUs")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")
    else:
        if use_multigpu and num_gpus <= 1:
            print(f"WARNING: use_multigpu=True but only {num_gpus} GPU(s) detected. Using single GPU mode.")
        num_gpus = 1
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for faster computation")
    print(f"Using cache: {use_cache}")
    if use_cache:
        print(f"Cache directory: {cache_dir}")
    print(f"Using FAISS: {use_faiss}")
    print(f"Auto batch tuning: {auto_batch}")

    if quest_plus:
        QUERY_FILE = query_file if query_file else "./data/QUEST_w_Varients/quest_test_withVarients.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST_w_Varients/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = output_file if output_file else "results_plus.jsonl"
    else:
        QUERY_FILE = query_file if query_file else "./data/QUEST/test_id_added.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST/documents.jsonl"
        OUTPUT_FILE = output_file if output_file else "results.jsonl"
    
    print("Collecting documents...")
    print(f"Corpus file: {CORPUS_FILE}")
    print(f"Query file: {QUERY_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Quest plus: {quest_plus}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Index name: {INDEX_NAME}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Use fp16: {use_fp16}")
    corpus_quest_plus = corpus_format if corpus_format is not None else quest_plus
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, corpus_quest_plus)
    print(f"Total documents loaded: {len(doc_ids)}")
    print(f"doc_ids: {doc_ids[:3]}")
    print(f"doc_title_map (first 3): {dict(list(doc_title_map.items())[:3])}")

    # Optionally limit number of documents for smaller debug runs
    if max_docs is not None and max_docs > 0 and max_docs < len(doc_ids):
        print(f"[DEBUG] Reducing documents from {len(doc_ids)} to first {max_docs}")
        doc_ids = doc_ids[:max_docs]
        doc_texts = doc_texts[:max_docs]
        # Restrict title map to kept doc_ids
        doc_title_map = {doc_id: doc_title_map.get(doc_id) for doc_id in doc_ids}
        print(f"[DEBUG] Documents after reduction: {len(doc_ids)}")

    print("Collecting queries...")
    query_quest_plus = query_format if query_format is not None else quest_plus
    query, ground_truths = tools.queries(QUERY_FILE, query_quest_plus)
    print(f"Total queries loaded: {len(query)}")

    # Optionally limit number of queries for smaller debug runs
    if max_queries is not None and max_queries > 0 and max_queries < len(query):
        print(f"[DEBUG] Reducing queries from {len(query)} to first {max_queries}")
        query = query[:max_queries]
        ground_truths = ground_truths[:max_queries]
        print(f"[DEBUG] Queries after reduction: {len(query)}")
    
    # inspecting how query, ground_truths are structured
    print(f"query: {query[:3]}")
    print(f"ground_truths: {ground_truths[:3]}")
    print(f"Loading model: {MODEL_NAME}...")
    
    # Workaround for tokenizer loading issue
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load SentenceTransformer model
    # For multi-GPU encoding, we use data parallelism (not model sharding)
    # Each GPU will load its own model instance in encode_with_multigpu
    if use_multigpu and num_gpus > 1:
        print(f"Multi-GPU encoding: Will use data parallelism (splitting texts across {num_gpus} GPUs)")
        print("Model instances will be loaded separately on each GPU during encoding")
        model = None
    else:
        # Single GPU: load model normally
        model = SentenceTransformer(MODEL_NAME, device=device_obj)
        model.max_seq_length = max_length
        model.eval()
        print(f"Model loaded successfully. Max sequence length: {model.max_seq_length}")
        
    # Check if flash attention is available
    if HAS_FLASH_ATTN:
        print("Flash Attention: Available (may be used automatically)")

    print("Creating Embeddings...")
    # Format texts with E5 prefixes
    formatted_queries = [e5_format_query(q) for q in query]
    formatted_docs = [e5_format_passage(doc) for doc in doc_texts]

    # Memory optimization: reduce batch size for documents if we have many documents
    if doc_batch_size is None:
        doc_batch_size = batch_size
        if len(formatted_docs) > 100000:
            doc_batch_size = min(16, batch_size)
            print(f"Large document set detected ({len(formatted_docs)} documents). Using reduced batch size: {doc_batch_size}")
    else:
        print(f"Using custom document batch size: {doc_batch_size}")
    
    # Auto-tune batch sizes if requested (skip for multi-GPU as models load separately)
    if auto_batch and device == "cuda" and not (use_multigpu and num_gpus > 1):
        print("Auto-tuning batch sizes based on GPU memory...")
        optimal_doc_batch = auto_tune_batch_size(model, formatted_docs[:1000], device, max_length, 
                                                  start_batch=doc_batch_size, max_batch=512)
        optimal_query_batch = auto_tune_batch_size(model, formatted_queries[:100], device, max_length,
                                                    start_batch=batch_size, max_batch=512)
        if optimal_doc_batch != doc_batch_size:
            print(f"  Document batch size: {doc_batch_size} -> {optimal_doc_batch}")
            doc_batch_size = optimal_doc_batch
        if optimal_query_batch != batch_size:
            print(f"  Query batch size: {batch_size} -> {optimal_query_batch}")
            batch_size = optimal_query_batch
    
    # Content-based cache keys: reuse cache when same doc/query texts are used later
    doc_content_key = content_hash(formatted_docs) if use_cache and cache_dir else ""
    query_content_key = content_hash(formatted_queries) if use_cache and cache_dir else ""

    # Setup cache directories: keyed by model, task, content hash, and batch/maxlen
    if use_cache and cache_dir:
        doc_cache_dir = os.path.join(cache_dir, 'doc_emb', MODEL_NAME.replace('/', '--'), task_name,
                                     doc_content_key, f"batch_{doc_batch_size}_maxlen_{max_length}")
        query_cache_dir = os.path.join(cache_dir, 'query_emb', MODEL_NAME.replace('/', '--'), task_name,
                                        query_content_key, f"batch_{batch_size}_maxlen_{max_length}")
        os.makedirs(doc_cache_dir, exist_ok=True)
        os.makedirs(query_cache_dir, exist_ok=True)
        doc_cache_file = os.path.join(doc_cache_dir, 'embeddings.npy')
        query_cache_file = os.path.join(query_cache_dir, 'embeddings.npy')
        print(f"Document cache key (content): {doc_content_key}")
        print(f"Query cache key (content): {query_content_key}")
    else:
        doc_cache_file = None
        query_cache_file = None
    
    with tools.benchmark(MODEL_NAME, "Embedding"):
        # Clear GPU cache before encoding
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Inspect document batches if enabled
        inspect_batch_samples(formatted_docs, doc_batch_size, name="documents")
        
        # Encode documents with caching (support sharded caching so we can resume)
        if use_cache and doc_cache_file and os.path.isfile(doc_cache_file):
            print(f"Loading cached document embeddings from {doc_cache_file}")
            doc_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            # If we have a cache directory, use sharded caching so that each chunk
            # is saved as we go. This allows resuming long runs without recomputing
            # already-encoded chunks.
            DOC_SHARD_SIZE = 10000  # number of docs per shard
            use_sharded_doc_cache = use_cache and cache_dir is not None and len(formatted_docs) > DOC_SHARD_SIZE
            if use_sharded_doc_cache:
                doc_shard_dir = os.path.join(doc_cache_dir, "shards")
                os.makedirs(doc_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded document cache in {doc_shard_dir} with shard_size={DOC_SHARD_SIZE}")
                doc_emb_chunks = []
                total_docs = len(formatted_docs)
                for start_idx in range(0, total_docs, DOC_SHARD_SIZE):
                    end_idx = min(start_idx + DOC_SHARD_SIZE, total_docs)
                    shard_path = os.path.join(doc_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    expected_rows = end_idx - start_idx
                    shard_loaded = False
                    if os.path.isfile(shard_path):
                        try:
                            emb_chunk = np.load(shard_path, allow_pickle=True)
                            if emb_chunk.shape[0] == expected_rows:
                                shard_loaded = True
                                print(f"Loading cached document shard {start_idx}-{end_idx-1} from {shard_path}")
                            else:
                                print(f"Invalid cached shard shape {emb_chunk.shape} (expected {expected_rows} rows); re-encoding.")
                        except (ValueError, OSError) as e:
                            print(f"Failed to load shard {shard_path}: {e}; re-encoding.")
                        if not shard_loaded and os.path.isfile(shard_path):
                            try:
                                os.remove(shard_path)
                            except OSError:
                                pass
                    if not shard_loaded:
                        print(f"Encoding document shard {start_idx}-{end_idx-1} ({end_idx-start_idx} docs)...")
                        shard_texts = formatted_docs[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME,
                                shard_texts,
                                doc_batch_size,
                                max_length,
                                num_gpus,
                            )
                        else:
                            emb_list = []
                            with torch.inference_mode():
                                for bstart in range(0, len(shard_texts), doc_batch_size):
                                    bend = min(bstart + doc_batch_size, len(shard_texts))
                                    batch_texts = shard_texts[bstart:bend]
                                    emb = model.encode(
                                        batch_texts,
                                        batch_size=len(batch_texts),
                                        show_progress_bar=False,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True,
                                    )
                                    emb_list.append(emb)
                            emb_chunk = np.concatenate(emb_list, axis=0)
                        if use_cache:
                            print(f"Saving document shard {start_idx}-{end_idx-1} to cache: {shard_path}")
                            np.save(shard_path, emb_chunk)
                    doc_emb_chunks.append(emb_chunk)
                doc_emb = np.concatenate(doc_emb_chunks, axis=0)
                print(f"[DEBUG] Finished document encoding; doc_emb shape: {doc_emb.shape}")
                if use_cache and doc_cache_file:
                    print(f"Saving combined document embeddings to cache: {doc_cache_file}")
                    np.save(doc_cache_file, doc_emb)
            else:
                print(f"Encoding {len(formatted_docs)} documents...")
                if use_multigpu and num_gpus > 1:
                    doc_emb = encode_with_multigpu(MODEL_NAME, formatted_docs, doc_batch_size, max_length, num_gpus)
                else:
                    emb_list = []
                    with torch.inference_mode():
                        for start in range(0, len(formatted_docs), doc_batch_size):
                            end = min(start + doc_batch_size, len(formatted_docs))
                            batch_texts = formatted_docs[start:end]
                            emb = model.encode(
                                batch_texts,
                                batch_size=len(batch_texts),
                                show_progress_bar=True,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                            )
                            emb_list.append(emb)
                    doc_emb = np.concatenate(emb_list, axis=0)
                print(f"[DEBUG] Finished document encoding; doc_emb shape: {doc_emb.shape}")
                if use_cache and doc_cache_file:
                    print(f"Saving document embeddings to cache: {doc_cache_file}")
                    np.save(doc_cache_file, doc_emb)
        
        # Clear GPU cache between document and query encoding
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Inspect query batches if enabled
        inspect_batch_samples(formatted_queries, batch_size, name="queries")
        
        # Encode queries with caching (queries are smaller, but mirror sharded logic for robustness)
        if use_cache and query_cache_file and os.path.isfile(query_cache_file):
            print(f"Loading cached query embeddings from {query_cache_file}")
            query_emb = np.load(query_cache_file, allow_pickle=True)
        else:
            QUERY_SHARD_SIZE = 2000  # safe default
            use_sharded_query_cache = use_cache and cache_dir is not None and len(formatted_queries) > QUERY_SHARD_SIZE
            if use_sharded_query_cache:
                query_shard_dir = os.path.join(query_cache_dir, "shards")
                os.makedirs(query_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded query cache in {query_shard_dir} with shard_size={QUERY_SHARD_SIZE}")
                query_emb_chunks = []
                total_queries = len(formatted_queries)
                for start_idx in range(0, total_queries, QUERY_SHARD_SIZE):
                    end_idx = min(start_idx + QUERY_SHARD_SIZE, total_queries)
                    shard_path = os.path.join(query_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    expected_rows = end_idx - start_idx
                    shard_loaded = False
                    if os.path.isfile(shard_path):
                        try:
                            emb_chunk = np.load(shard_path, allow_pickle=True)
                            if emb_chunk.shape[0] == expected_rows:
                                shard_loaded = True
                                print(f"Loading cached query shard {start_idx}-{end_idx-1} from {shard_path}")
                            else:
                                print(f"Invalid cached query shard shape {emb_chunk.shape} (expected {expected_rows} rows); re-encoding.")
                        except (ValueError, OSError) as e:
                            print(f"Failed to load query shard {shard_path}: {e}; re-encoding.")
                        if not shard_loaded and os.path.isfile(shard_path):
                            try:
                                os.remove(shard_path)
                            except OSError:
                                pass
                    if not shard_loaded:
                        print(f"Encoding query shard {start_idx}-{end_idx-1} ({end_idx-start_idx} queries)...")
                        shard_queries = formatted_queries[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME,
                                shard_queries,
                                batch_size,
                                max_length,
                                num_gpus,
                            )
                        else:
                            emb_list = []
                            with torch.inference_mode():
                                for bstart in range(0, len(shard_queries), batch_size):
                                    bend = min(bstart + batch_size, len(shard_queries))
                                    batch_texts = shard_queries[bstart:bend]
                                    emb = model.encode(
                                        batch_texts,
                                        batch_size=len(batch_texts),
                                        show_progress_bar=False,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True,
                                    )
                                    emb_list.append(emb)
                            emb_chunk = np.concatenate(emb_list, axis=0)
                        if use_cache:
                            print(f"Saving query shard {start_idx}-{end_idx-1} to cache: {shard_path}")
                            np.save(shard_path, emb_chunk)
                    query_emb_chunks.append(emb_chunk)
                query_emb = np.concatenate(query_emb_chunks, axis=0)
                print(f"[DEBUG] Finished query encoding; query_emb shape: {query_emb.shape}")
                if use_cache and query_cache_file:
                    print(f"Saving combined query embeddings to cache: {query_cache_file}")
                    np.save(query_cache_file, query_emb)
            else:
                print(f"Encoding {len(formatted_queries)} queries...")
                if use_multigpu and num_gpus > 1:
                    query_emb = encode_with_multigpu(MODEL_NAME, formatted_queries, batch_size, max_length, num_gpus)
                else:
                    with torch.inference_mode():
                        query_emb = model.encode(
                            formatted_queries,
                            batch_size=batch_size,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                    query_emb = np.asarray(query_emb, dtype=np.float32)
                print(f"[DEBUG] Finished query encoding; query_emb shape: {query_emb.shape}")
                if use_cache and query_cache_file:
                    print(f"Saving query embeddings to cache: {query_cache_file}")
                    np.save(query_cache_file, query_emb)
    
    # Compute similarity scores
    print("Computing similarity scores...")
    if use_faiss:
        # Use FAISS for large-scale search (more memory efficient for very large corpora)
        print(f"[DEBUG] Creating FAISS index '{INDEX_NAME}'...")
        index = tools.create_index(INDEX_NAME, query_emb, doc_emb)
        print(f"[DEBUG] FAISS index '{INDEX_NAME}' created; starting search_index...")
        scores, indices = tools.search_index(index, query_emb)
        print("[DEBUG] Finished FAISS search_index()")
    else:
        # Use direct matrix multiplication (faster for smaller corpora, matches retrievers.py approach)
        with tools.benchmark(MODEL_NAME, "Similarity"):
            query_tensor = torch.from_numpy(query_emb).to(device_obj)
            doc_tensor = torch.from_numpy(doc_emb).to(device_obj)
            # Compute cosine similarity (embeddings are already normalized by model)
            scores_tensor = pairwise_cosine_similarity(query_tensor, doc_tensor)
            scores_np = scores_tensor.cpu().numpy()
            
            # Get top-k indices and scores
            top_k = 100
            indices = np.argsort(scores_np, axis=1)[:, ::-1][:, :top_k]
            scores = np.take_along_axis(scores_np, indices, axis=1)
    
    print("[DEBUG] Starting retrieval and writing results...")
    tools.start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, indices, scores)
    print(f"[DEBUG] Finished retrieval. Results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run E5-base-v2 model for information retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default files
  python e5-base-v2_main.py
  
  # Use quest_plus mode
  python e5-base-v2_main.py --quest-plus
  
  # Specify custom file paths
  python e5-base-v2_main.py --query-file ./data/queries.jsonl --corpus-file ./data/documents.jsonl --output-file results.jsonl
  
  # Full example with all options
  python e5-base-v2_main.py --quest-plus --query-file ./custom/queries.jsonl --corpus-file ./custom/docs.jsonl \\
                        --output-file custom_results.jsonl --model-name intfloat/e5-base-v2 --index-name my-index
        """
    )
    
    parser.add_argument(
        "--quest-plus",
        action="store_true",
        help="Use quest_plus mode (default: False)"
    )
    
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Path to query file (JSONL format). Defaults based on quest_plus mode."
    )
    
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus/document file (JSONL format). Defaults based on quest_plus mode."
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output results file (JSONL format). Defaults based on quest_plus mode."
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name to use (default: 'intfloat/e5-base-v2')"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Name for the FAISS index (default: 'E5-base-v2-QUEST')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto-detect)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding queries (default: 32, increase for faster GPU processing)"
    )
    
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=None,
        help="Batch size for encoding documents (default: same as --batch-size, or auto-reduced for large datasets). Use smaller values (8-16) if encountering OOM errors."
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable float16 precision (use full float32). Note: SentenceTransformer handles precision automatically."
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding caching (default: enabled)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching embeddings (default: ./cache relative to output file)"
    )
    
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Use direct matrix multiplication instead of FAISS (faster for smaller corpora, matches retrievers.py approach)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for encoding (default: 512)"
    )
    
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="Automatically tune batch sizes based on GPU memory (default: False)"
    )
    
    parser.add_argument(
        "--multigpu",
        action="store_true",
        help="Use multiple GPUs if available (default: False). Requires multiple GPUs in SLURM."
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit to the first N documents (for debugging / small runs)."
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit to the first N queries (for debugging / small runs)."
    )

    parser.add_argument(
        "--task-suffix",
        type=str,
        default=None,
        help="Suffix for cache/index task folder (e.g. _withVariants). Ensures distinct cache from default runs."
    )

    parser.add_argument(
        "--query-format",
        type=str,
        choices=["quest", "quest_plus"],
        default=None,
        help="Query file format: quest (query/docs) or quest_plus (nl_query/documents). Default: same as --quest-plus."
    )

    parser.add_argument(
        "--corpus-format",
        type=str,
        choices=["quest", "quest_plus"],
        default=None,
        help="Corpus file format: quest (sequential id) or quest_plus (idx key). Default: same as --quest-plus."
    )
    
    args = parser.parse_args()
    
    query_fmt = args.query_format
    corpus_fmt = args.corpus_format
    if query_fmt is not None:
        query_fmt = query_fmt == "quest_plus"
    if corpus_fmt is not None:
        corpus_fmt = corpus_fmt == "quest_plus"
    
    # Handle device selection
    device = args.device
    if device == "auto" or device is None:
        device = None  # Will auto-detect in main()
    
    main(
        quest_plus=args.quest_plus,
        query_file=args.query_file,
        corpus_file=args.corpus_file,
        output_file=args.output_file,
        model_name=args.model_name,
        index_name=args.index_name,
        device=device,
        batch_size=args.batch_size,
        doc_batch_size=args.doc_batch_size,
        use_fp16=not args.no_fp16,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        use_faiss=not args.no_faiss,
        max_length=args.max_length,
        auto_batch=args.auto_batch,
        use_multigpu=args.multigpu,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        task_suffix=args.task_suffix,
        query_format=query_fmt,
        corpus_format=corpus_fmt,
    )
