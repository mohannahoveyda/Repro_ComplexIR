import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import math
import multiprocessing as mp
from multiprocessing import Process, Queue
import tempfile

# Set multiprocessing start method to 'spawn' for CUDA isolation
# This ensures each process has its own CUDA context
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.dirname(script_dir)
if baselines_dir not in sys.path:
    sys.path.insert(0, baselines_dir)

import utils.helper as tools
from transformers import AutoModel, AutoTokenizer
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

def inspect_batch_samples(texts, batch_size, name="items", max_samples=2):
    """Minimal function to inspect what's in each batch. Enable with INSPECT_BATCHES=1"""
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    # Debug: print the environment variable value
    print(f"[DEBUG] INSPECT_BATCHES env var = '{inspect_env}' (type: {type(inspect_env)})")
    if inspect_env != "1":
        print(f"[DEBUG] Inspection disabled (value was '{inspect_env}', expected '1')")
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting {name} batches (showing first {max_samples} batches)")
    print(f"{'='*60}")
    for batch_idx in range(min(max_samples, (len(texts) + batch_size - 1) // batch_size)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1} (indices {start_idx}-{end_idx-1}, {len(batch_texts)} items):")
        for i, text in enumerate(batch_texts[:3]):  # Show first 3 items in batch
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
    
    # Try progressively larger batches
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
            # Test with a small sample (use min to avoid issues)
            test_size = min(test_batch, len(texts), 100)  # Test with up to 100 samples
            test_texts = texts[:test_size]
            with torch.inference_mode():
                _ = model.encode(test_texts, batch_size=test_batch, max_length=max_length)
            optimal_batch = test_batch
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                # For other errors, log but continue
                print(f"Warning during batch size test: {e}")
                break
    
    return optimal_batch

def encode_worker(gpu_id, model_name, model_kwargs, texts_chunk, instruction, batch_size, max_length, result_queue, temp_dir):
    """Worker function for multiprocessing: loads model on specific GPU and encodes texts.
    
    This runs in a separate process with its own CUDA context, preventing memory conflicts.
    Each process only sees its assigned GPU, ensuring complete isolation.
    """
    try:
        # Set CUDA device for this process BEFORE any CUDA operations
        # This ensures the process only uses its assigned GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Reinitialize CUDA in this process (needed after setting CUDA_VISIBLE_DEVICES)
        import torch
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)  # Now GPU 0 in this process is the actual GPU gpu_id
        
        # Import here to ensure each process has its own imports
        from transformers import AutoModel
        
        print(f"[GPU {gpu_id} Process] Loading model on device cuda:0 (maps to physical GPU {gpu_id})...")
        
        # Prepare model kwargs for this GPU
        gpu_kwargs = model_kwargs.copy()
        gpu_kwargs["device_map"] = {"": "cuda:0"}  # Use cuda:0 since CUDA_VISIBLE_DEVICES makes it the only GPU
        gpu_kwargs["low_cpu_mem_usage"] = True
        
        # Load model (will see only one GPU due to CUDA_VISIBLE_DEVICES)
        model = AutoModel.from_pretrained(model_name, **gpu_kwargs)
        model.eval()
        
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[GPU {gpu_id} Process] Model loaded (memory: {memory_used:.2f} GB)")
        
        # Encode texts in batches with our own tqdm so progress is always visible
        # (model.encode()'s internal tqdm can be sparse or in-place for small batch counts, e.g. queries)
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
                    instruction=instruction,
                    batch_size=len(batch_texts),
                    max_length=max_length,
                )
                emb_list.append(emb.cpu().numpy() if hasattr(emb, "cpu") else emb)
        embeddings = np.concatenate(emb_list, axis=0)
        
        print(f"[GPU {gpu_id} Process] Encoded -> shape {embeddings.shape}")
        
        # Save embeddings to a temporary file for this worker to avoid sending
        # large arrays through the multiprocessing.Queue (which would require
        # pickling and large IPC transfers).
        os.makedirs(temp_dir, exist_ok=True)
        shard_path = os.path.join(temp_dir, f"gpu_{gpu_id}.npy")
        np.save(shard_path, embeddings)
        
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

def encode_with_multigpu(model_name, model_kwargs, texts, instruction, batch_size, max_length, num_gpus):
    """Encode texts using multiple GPUs by splitting work across GPUs (data parallelism).
    
    This splits the texts across available GPUs and encodes them sequentially on each GPU.
    Each GPU gets its own separate model instance loaded independently.
    """
    if num_gpus <= 1:
        # This shouldn't be called with num_gpus <= 1, but handle it gracefully
        # Load a single model instance on GPU 0
        torch.cuda.set_device(0)
        single_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        single_model.eval()
        single_model.to("cuda:0")
        with torch.inference_mode():
            result = single_model.encode(texts, instruction=instruction, batch_size=batch_size, max_length=max_length)
        del single_model
        torch.cuda.empty_cache()
        return result
    
    print(f"Splitting {len(texts)} texts across {num_gpus} GPUs for PARALLEL encoding...")
    
    # Split texts across GPUs for data parallelism
    chunk_size = math.ceil(len(texts) / num_gpus)
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Ensure we have exactly num_gpus chunks
    while len(chunks) < num_gpus:
        chunks.append([])
    
    # Use multiprocessing for true parallelism
    # Each process gets its own CUDA context, preventing memory conflicts
    print(f"Starting {num_gpus} parallel processes (one per GPU)...")
    
    # Create queue for results (metadata only) and a temporary directory for
    # per-GPU embedding files. This avoids pushing large arrays through the
    # queue, which would require expensive pickling and IPC.
    result_queue = Queue()
    temp_dir = tempfile.mkdtemp(prefix="reasonir_mgpu_")
    processes = []
    
    # Start a process for each GPU
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
        
        print(f"  Starting process for GPU {gpu_id} ({len(chunks[gpu_id])} texts)...")
        p = Process(
            target=encode_worker,
            args=(gpu_id, model_name, model_kwargs, chunks[gpu_id], instruction, batch_size, max_length, result_queue, temp_dir)
        )
        p.start()
        processes.append((gpu_id, p))
    
    # Collect results from all processes.
    # Drain the queue first, then join. Otherwise we can deadlock: the parent blocks on
    # p.join() while a child blocks on result_queue.put() (queue full with large arrays).
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
        # Load per-GPU shards from disk in gpu_id order
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
            final_emb = np.array([])
        return final_emb
    else:
        return np.array([])

def main(quest_plus=False, query_file=None, corpus_file=None, output_file=None, 
         model_name=None, index_name=None, device=None, batch_size=32, doc_batch_size=None, 
         use_fp16=True, use_cache=True, cache_dir=None, use_faiss=True, max_length=32768, 
         auto_batch=False, use_multigpu=False, max_docs=None, max_queries=None):
    # Debug: Check INSPECT_BATCHES environment variable early
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    print(f"[DEBUG] At start of main(), INSPECT_BATCHES = '{inspect_env}'")

    MODEL_NAME = model_name if model_name else "reasonir/ReasonIR-8B"
    INDEX_NAME = index_name if index_name else "ReasonIR-8B-QUEST"
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_file) if output_file else ".", "cache") if use_cache else None
    task_name = "quest_plus" if quest_plus else "quest"
    
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
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, quest_plus)
    print(f"Total documents loaded: {len(doc_ids)}")
    # Add a tqdm progress bar
    # print("Calculating document lengths...")
    # doc_lengths = [len(tokenizer.encode(doc, add_special_tokens=False)) for doc in tqdm(doc_texts) ]
    # print(f"Min tokens: {min(doc_lengths)}")
    # print(f"Max tokens: {max(doc_lengths)}")
    # print(f"Mean tokens: {sum(doc_lengths)/len(doc_lengths):.0f}")
    # print(f"95th percentile: {sorted(doc_lengths)[int(0.95*len(doc_lengths))]}")
    # inspecting how doc_ids, doc_texts, doc_title_map are structured
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
    query, ground_truths = tools.queries(QUERY_FILE, quest_plus)
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
    
    # Workaround for tokenizer loading issue with ReasonIR-8B
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use AutoModel directly (more efficient than SentenceTransformer)
    # torch_dtype="auto" enables bf16 on supported GPUs (faster than fp16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Load model with optimizations
        model_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}
        # if device == "cuda" and HAS_FLASH_ATTN:
        #     model_kwargs["attn_implementation"] = "flash_attention_2"
        #     print("Flash Attention 2: Enabled")
        
        # For multi-GPU encoding, we use data parallelism (not model sharding)
        # Each GPU will load its own model instance in encode_with_multigpu
        # So we don't need to load a model here for multi-GPU
        if use_multigpu and num_gpus > 1:
            print(f"Multi-GPU encoding: Will use data parallelism (splitting texts across {num_gpus} GPUs)")
            print("Model instances will be loaded separately on each GPU during encoding")
            # Don't load model here - encode_with_multigpu will load separate instances
            model = None
        else:
            # Single GPU: load model normally
            model = AutoModel.from_pretrained(MODEL_NAME, **model_kwargs)
            model.eval()
            model.to(device_obj)
            print(f"Model loaded successfully. Using dtype: {next(model.parameters()).dtype}")
        
        # Check if flash attention is available and being used
        if HAS_FLASH_ATTN:
            print("Flash Attention: Available (may be used automatically)")
    except Exception as e:
        error_msg = str(e)
        if "ModelWrapper" in error_msg or ("tokenizer" in error_msg.lower() and "enum" in error_msg.lower()):
            print(f"\n{'='*60}")
            print("ERROR: Tokenizer loading failed due to version incompatibility.")
            print(f"{'='*60}")
            print("This is a known issue with tokenizers library version mismatch.")
            print("\nTo fix this, try one of the following:")
            print("1. Update tokenizers library:")
            print("   pip install --upgrade tokenizers")
            print("2. Clear the model cache and re-download:")
            print("   rm -rf ~/.cache/huggingface/hub/models--reasonir--ReasonIR-8B")
            print("3. Or update both transformers and tokenizers:")
            print("   pip install --upgrade transformers tokenizers")
            print(f"{'='*60}\n")
            raise RuntimeError(
                f"Tokenizer loading failed: {error_msg[:200]}...\n"
                "Please update the tokenizers library or clear the model cache."
            ) from e
        else:
            raise

    print("Creating Embeddings...")
    query_instruction = ""
    doc_instruction = ""

    # Memory optimization: reduce batch size for documents if we have many documents
    if doc_batch_size is None:
        doc_batch_size = batch_size
        if len(doc_texts) > 100000:
            doc_batch_size = min(16, batch_size)
            print(f"Large document set detected ({len(doc_texts)} documents). Using reduced batch size: {doc_batch_size}")
    else:
        print(f"Using custom document batch size: {doc_batch_size}")
    
    # Auto-tune batch sizes if requested (skip for multi-GPU as models load separately)
    if auto_batch and device == "cuda" and not (use_multigpu and num_gpus > 1):
        print("Auto-tuning batch sizes based on GPU memory...")
        optimal_doc_batch = auto_tune_batch_size(model, doc_texts[:1000], device, max_length, 
                                                  start_batch=doc_batch_size, max_batch=512)
        optimal_query_batch = auto_tune_batch_size(model, query[:100], device, max_length,
                                                    start_batch=batch_size, max_batch=512)
        if optimal_doc_batch != doc_batch_size:
            print(f"  Document batch size: {doc_batch_size} -> {optimal_doc_batch}")
            doc_batch_size = optimal_doc_batch
        if optimal_query_batch != batch_size:
            print(f"  Query batch size: {batch_size} -> {optimal_query_batch}")
            batch_size = optimal_query_batch
    
    # Setup cache directories if using cache
    # Include max_length in cache key to avoid incorrect reuse when max_length changes
    if use_cache and cache_dir:
        doc_cache_dir = os.path.join(cache_dir, 'doc_emb', MODEL_NAME.replace('/', '--'), task_name, f"batch_{doc_batch_size}_maxlen_{max_length}")
        query_cache_dir = os.path.join(cache_dir, 'query_emb', MODEL_NAME.replace('/', '--'), task_name, f"batch_{batch_size}_maxlen_{max_length}")
        os.makedirs(doc_cache_dir, exist_ok=True)
        os.makedirs(query_cache_dir, exist_ok=True)
        doc_cache_file = os.path.join(doc_cache_dir, 'embeddings.npy')
        query_cache_file = os.path.join(query_cache_dir, 'embeddings.npy')
    else:
        doc_cache_file = None
        query_cache_file = None
    
    with tools.benchmark(MODEL_NAME, "Embedding"):
        # Clear GPU cache before encoding
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Inspect document batches if enabled
        inspect_batch_samples(doc_texts, doc_batch_size, name="documents")
        
        # Encode documents with caching (support sharded caching so we can resume)
        if use_cache and doc_cache_file and os.path.isfile(doc_cache_file):
            print(f"Loading cached document embeddings from {doc_cache_file}")
            doc_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            # If we have a cache directory, use sharded caching so that each chunk
            # is saved as we go. This allows resuming long runs without recomputing
            # already-encoded chunks.
            DOC_SHARD_SIZE = 10000  # number of docs per shard; smaller = better resume granularity
            use_sharded_doc_cache = use_cache and cache_dir is not None and len(doc_texts) > DOC_SHARD_SIZE
            if use_sharded_doc_cache:
                doc_shard_dir = os.path.join(doc_cache_dir, "shards")
                os.makedirs(doc_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded document cache in {doc_shard_dir} with shard_size={DOC_SHARD_SIZE}")
                doc_emb_chunks = []
                total_docs = len(doc_texts)
                for start_idx in range(0, total_docs, DOC_SHARD_SIZE):
                    end_idx = min(start_idx + DOC_SHARD_SIZE, total_docs)
                    shard_path = os.path.join(doc_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        print(f"Loading cached document shard {start_idx}-{end_idx-1} from {shard_path}")
                        emb_chunk = np.load(shard_path, allow_pickle=True)
                    else:
                        print(f"Encoding document shard {start_idx}-{end_idx-1} ({end_idx-start_idx} docs)...")
                        shard_texts = doc_texts[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME,
                                model_kwargs,
                                shard_texts,
                                doc_instruction,
                                doc_batch_size,
                                max_length,
                                num_gpus,
                            )
                        else:
                            with torch.inference_mode():
                                emb_chunk = model.encode(
                                    shard_texts,
                                    instruction=doc_instruction,
                                    batch_size=doc_batch_size,
                                    max_length=max_length,
                                )
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
                print(f"Encoding {len(doc_texts)} documents...")
                if use_multigpu and num_gpus > 1:
                    doc_emb = encode_with_multigpu(MODEL_NAME, model_kwargs, doc_texts, doc_instruction, doc_batch_size, max_length, num_gpus)
                else:
                    with torch.inference_mode():
                        doc_emb = model.encode(
                            doc_texts,
                            instruction=doc_instruction,
                            batch_size=doc_batch_size,
                            max_length=max_length
                        )
                print(f"[DEBUG] Finished document encoding; doc_emb shape: {doc_emb.shape}")
                if use_cache and doc_cache_file:
                    print(f"Saving document embeddings to cache: {doc_cache_file}")
                    np.save(doc_cache_file, doc_emb)
        
        # Clear GPU cache between document and query encoding
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Inspect query batches if enabled
        inspect_batch_samples(query, batch_size, name="queries")
        
        # Encode queries with caching (queries are smaller, but mirror sharded logic for robustness)
        if use_cache and query_cache_file and os.path.isfile(query_cache_file):
            print(f"Loading cached query embeddings from {query_cache_file}")
            query_emb = np.load(query_cache_file, allow_pickle=True)
        else:
            QUERY_SHARD_SIZE = 2000  # safe default; larger than typical query counts here
            use_sharded_query_cache = use_cache and cache_dir is not None and len(query) > QUERY_SHARD_SIZE
            if use_sharded_query_cache:
                query_shard_dir = os.path.join(query_cache_dir, "shards")
                os.makedirs(query_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded query cache in {query_shard_dir} with shard_size={QUERY_SHARD_SIZE}")
                query_emb_chunks = []
                total_queries = len(query)
                for start_idx in range(0, total_queries, QUERY_SHARD_SIZE):
                    end_idx = min(start_idx + QUERY_SHARD_SIZE, total_queries)
                    shard_path = os.path.join(query_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        print(f"Loading cached query shard {start_idx}-{end_idx-1} from {shard_path}")
                        emb_chunk = np.load(shard_path, allow_pickle=True)
                    else:
                        print(f"Encoding query shard {start_idx}-{end_idx-1} ({end_idx-start_idx} queries)...")
                        shard_queries = query[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME,
                                model_kwargs,
                                shard_queries,
                                query_instruction,
                                batch_size,
                                max_length,
                                num_gpus,
                            )
                        else:
                            with torch.inference_mode():
                                emb_chunk = model.encode(
                                    shard_queries,
                                    instruction=query_instruction,
                                    batch_size=batch_size,
                                    max_length=max_length,
                                )
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
                print(f"Encoding {len(query)} queries...")
                if use_multigpu and num_gpus > 1:
                    query_emb = encode_with_multigpu(MODEL_NAME, model_kwargs, query, query_instruction, batch_size, max_length, num_gpus)
                else:
                    with torch.inference_mode():
                        query_emb = model.encode(
                            query,
                            instruction=query_instruction,
                            batch_size=batch_size,
                            max_length=max_length
                        )
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
        description="Run ReasonIR-8B model for information retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default files
  python reasonir-8b.py
  
  # Use quest_plus mode
  python reasonir-8b.py --quest-plus
  
  # Specify custom file paths
  python reasonir-8b.py --query-file ./data/queries.jsonl --corpus-file ./data/documents.jsonl --output-file results.jsonl
  
  # Full example with all options
  python reasonir-8b.py --quest-plus --query-file ./custom/queries.jsonl --corpus-file ./custom/docs.jsonl \\
                        --output-file custom_results.jsonl --model-name reasonir/ReasonIR-8B --index-name my-index
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
        help="Model name to use (default: 'reasonir/ReasonIR-8B')"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Name for the FAISS index (default: 'ReasonIR-8B-QUEST')"
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
        help="Disable float16 precision (use full float32). Note: Using AutoModel with torch_dtype='auto' enables bf16 which is faster."
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
        default=32768,
        help="Maximum sequence length for encoding (default: 32768)"
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
    
    args = parser.parse_args()
    
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
        max_queries=args.max_queries
    )