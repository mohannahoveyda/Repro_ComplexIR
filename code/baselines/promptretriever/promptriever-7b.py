"""
Promptriever-7B baseline for QUEST retrieval.
Full implementation aligned with reasonir-8b and gritlm-7b: CLI, cache, sharding, multi-GPU, FAISS/direct similarity.
"""
import argparse
import hashlib
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import multiprocessing as mp
from multiprocessing import Process, Queue
import tempfile

# Set multiprocessing start method to 'spawn' for CUDA isolation
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.dirname(script_dir)
if baselines_dir not in sys.path:
    sys.path.insert(0, baselines_dir)

import utils.helper as tools
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

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


# --- Promptriever model (PEFT LLaMA + merge) ---

class Promptriever:
    """Promptriever: PEFT adapter on LLaMA, merge_and_unload; encode via last-token hidden state."""

    def __init__(self, model_name_or_path, device=None, torch_dtype="auto"):
        self.model, self.tokenizer = self._load_model(model_name_or_path, device, torch_dtype)
        self.model.eval()
        if device:
            self.model = self.model.to(device)
        self._device = next(self.model.parameters()).device
        self._max_length = getattr(self.model.config, "max_length", 512)

    def _load_model(self, peft_model_name, device, torch_dtype):
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path
        base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype if isinstance(torch_dtype, torch.dtype) else "auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.config.max_length = getattr(model.config, "max_length", 512)
        tokenizer.model_max_length = model.config.max_length
        return model, tokenizer

    def _create_batch_dict(self, input_texts, max_length):
        max_len = min(max_length, self.tokenizer.model_max_length or 512)
        batch_dict = self.tokenizer(
            input_texts,
            max_length=max_len - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            ids + [self.tokenizer.eos_token_id] for ids in batch_dict["input_ids"]
        ]
        return self.tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, sentences, max_length=512, batch_size=4):
        if not sentences:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        all_embeddings = []
        dev = self._device
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            batch_dict = self._create_batch_dict(batch_texts, max_length)
            batch_dict = {k: v.to(dev) for k, v in batch_dict.items()}
            with torch.amp.autocast("cuda", enabled=(str(dev.type) == "cuda")):
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden = outputs.last_hidden_state
                    seq_lens = batch_dict["attention_mask"].sum(dim=1) - 1
                    bs = last_hidden.shape[0]
                    reps = last_hidden[
                        torch.arange(bs, device=last_hidden.device),
                        seq_lens,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


# --- Instruction formatting (Promptriever uses "query: ..." / "passage: ...") ---

def promptriever_query_instruction(raw_instruction):
    """Format for query encoding: 'query:  {q} {instruction}' (caller appends to query text)."""
    return (raw_instruction or "").strip()


def promptriever_doc_instruction(raw_instruction):
    """Format for doc encoding: 'passage:  {d}' (doc instruction typically empty)."""
    return (raw_instruction or "").strip()


def _safe_save_npy(path, arr):
    """Save numpy array; on OSError (quota/disk) raise with a helpful message."""
    try:
        np.save(path, arr)
    except OSError as e:
        raise OSError(
            f"Failed to write cache file '{path}' ({e}). "
            "Common causes: disk full or quota exceeded (e.g. on home/GPFS). "
            "Try: 1) Use --no-cache to skip caching, or 2) Set CACHE_DIR to a path on scratch "
            "with more space (e.g. $TMPDIR or /scratch/$USER)."
        ) from e


def content_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:24]


def inspect_batch_samples(texts, batch_size, name="items", max_samples=2):
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    if inspect_env != "1":
        return
    print(f"\n{'='*60}\nInspecting {name} batches (first {max_samples} batches)\n{'='*60}")
    for batch_idx in range(min(max_samples, (len(texts) + batch_size - 1) // batch_size)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        print(f"\nBatch {batch_idx + 1} (indices {start_idx}-{end_idx-1}, {len(batch_texts)} items):")
        for i, text in enumerate(batch_texts[:3]):
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"  [{i}] {preview}")
        if len(batch_texts) > 3:
            print(f"  ... and {len(batch_texts) - 3} more")
    print(f"{'='*60}\n")


def auto_tune_batch_size(model, texts, device, max_length, start_batch=32, max_batch=128):
    if device != "cuda":
        return start_batch
    model.eval()
    optimal_batch = start_batch
    test_sizes = [start_batch]
    if start_batch * 2 <= max_batch:
        test_sizes.append(start_batch * 2)
    if start_batch * 4 <= max_batch:
        test_sizes.append(start_batch * 4)
    for test_batch in test_sizes:
        try:
            torch.cuda.empty_cache()
            test_size = min(test_batch, len(texts), 50)
            test_texts = texts[:test_size]
            with torch.inference_mode():
                _ = model.encode(test_texts, batch_size=test_batch, max_length=max_length)
            optimal_batch = test_batch
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                torch.cuda.empty_cache()
                break
            break
    return optimal_batch


def encode_worker(gpu_id, model_name, texts_chunk, batch_size, max_length, result_queue, temp_dir):
    """Worker: load Promptriever on one GPU, encode chunk, save npy."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        print(f"[GPU {gpu_id} Process] Loading Promptriever on cuda:0 (physical GPU {gpu_id})...")
        model = Promptriever(model_name, device=torch.device("cuda:0"))
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[GPU {gpu_id} Process] Model loaded (memory: {memory_used:.2f} GB)")
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
                emb = model.encode(batch_texts, batch_size=len(batch_texts), max_length=max_length)
                emb_list.append(emb)
        embeddings = np.concatenate(emb_list, axis=0)
        print(f"[GPU {gpu_id} Process] Encoded -> shape {embeddings.shape}")
        os.makedirs(temp_dir, exist_ok=True)
        shard_path = os.path.join(temp_dir, f"gpu_{gpu_id}.npy")
        _safe_save_npy(shard_path, embeddings)
        result_queue.put((gpu_id, shard_path))
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id} Process] ERROR: {e}\n{traceback.format_exc()}")
        result_queue.put((gpu_id, None))
        raise


def encode_with_multigpu(model_name, texts, batch_size, max_length, num_gpus):
    if num_gpus <= 1:
        return None  # caller uses single model
    print(f"Splitting {len(texts)} texts across {num_gpus} GPUs for PARALLEL encoding...")
    chunk_size = math.ceil(len(texts) / num_gpus)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    while len(chunks) < num_gpus:
        chunks.append([])
    result_queue = Queue()
    temp_dir = tempfile.mkdtemp(prefix="promptriever_mgpu_")
    processes = []
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
        print(f"  Starting process for GPU {gpu_id} ({len(chunks[gpu_id])} texts)...")
        p = Process(
            target=encode_worker,
            args=(gpu_id, model_name, chunks[gpu_id], batch_size, max_length, result_queue, temp_dir),
        )
        p.start()
        processes.append((gpu_id, p))
    results_dict = {}
    for _ in range(len(processes)):
        gpu_id_result, shard_path = result_queue.get()
        if shard_path is not None:
            results_dict[gpu_id_result] = shard_path
        else:
            raise RuntimeError(f"GPU {gpu_id_result} process failed!")
    for gpu_id, p in processes:
        p.join()
    loaded = []
    for i in range(num_gpus):
        if i in results_dict:
            loaded.append(np.load(results_dict[i], allow_pickle=True))
    if not loaded:
        return np.array([])
    return np.concatenate(loaded, axis=0)


def main(
    quest_plus=False,
    query_file=None,
    corpus_file=None,
    output_file=None,
    model_name=None,
    index_name=None,
    device=None,
    batch_size=32,
    doc_batch_size=None,
    use_fp16=True,
    use_cache=True,
    cache_dir=None,
    use_faiss=True,
    max_length=512,
    auto_batch=False,
    use_multigpu=False,
    max_docs=None,
    max_queries=None,
    query_instruction_raw=None,
    doc_instruction_raw=None,
    task_suffix=None,
    query_format=None,
    corpus_format=None,
):
    inspect_env = os.getenv("INSPECT_BATCHES", "0")
    print(f"[DEBUG] At start of main(), INSPECT_BATCHES = '{inspect_env}'")

    MODEL_NAME = model_name if model_name else "samaya-ai/promptriever-llama2-7b-v1"
    INDEX_NAME = index_name if index_name else "Promptriever-7B-QUEST"

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_file) if output_file else ".", "cache") if use_cache else None
    task_name = ("quest_plus" if quest_plus else "quest") + (task_suffix or "")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"PyTorch detected {num_gpus} GPU(s), use_multigpu={use_multigpu}")
    if use_multigpu and num_gpus > 1:
        print(f"Multi-GPU mode: Using {num_gpus} GPUs")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")
    else:
        if use_multigpu and num_gpus <= 1:
            print("WARNING: use_multigpu=True but only 1 GPU. Using single GPU mode.")
        num_gpus = 1
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
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
    print(f"Max length: {max_length}")

    corpus_quest_plus = corpus_format if corpus_format is not None else quest_plus
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, corpus_quest_plus)
    print(f"Total documents loaded: {len(doc_ids)}")
    print(f"doc_ids (first 3): {doc_ids[:3]}")
    print(f"doc_title_map (first 3): {dict(list(doc_title_map.items())[:3])}")

    if max_docs is not None and max_docs > 0 and max_docs < len(doc_ids):
        print(f"[DEBUG] Reducing documents from {len(doc_ids)} to first {max_docs}")
        doc_ids = doc_ids[:max_docs]
        doc_texts = doc_texts[:max_docs]
        doc_title_map = {doc_id: doc_title_map.get(doc_id) for doc_id in doc_ids}
        print(f"[DEBUG] Documents after reduction: {len(doc_ids)}")

    print("Collecting queries...")
    query_quest_plus = query_format if query_format is not None else quest_plus
    query, ground_truths = tools.queries(QUERY_FILE, query_quest_plus)
    print(f"Total queries loaded: {len(query)}")
    print(f"query (first 3): {query[:3]}")
    print(f"ground_truths (first 3): {ground_truths[:3]}")

    if max_queries is not None and max_queries > 0 and max_queries < len(query):
        print(f"[DEBUG] Reducing queries from {len(query)} to first {max_queries}")
        query = query[:max_queries]
        ground_truths = ground_truths[:max_queries]
        print(f"[DEBUG] Queries after reduction: {len(query)}")

    print(f"Loading model: {MODEL_NAME}...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_kwargs = {"torch_dtype": "auto"}
    if use_multigpu and num_gpus > 1:
        print(f"Multi-GPU encoding: Will split texts across {num_gpus} GPUs")
        model = None
    else:
        model = Promptriever(MODEL_NAME, device=device_obj)
        print(f"Model loaded successfully.")

    QUERY_INSTRUCTION_RAW_DEFAULT = "Given the query, retrieve the most relevant documents"
    q_inst = promptriever_query_instruction(
        query_instruction_raw if query_instruction_raw is not None else QUERY_INSTRUCTION_RAW_DEFAULT
    )
    d_inst = promptriever_doc_instruction(doc_instruction_raw if doc_instruction_raw is not None else "")

    processed_queries = [f"query:  {q.strip()} {q_inst}".strip() for q in query]
    processed_documents = [f"passage:  {d.strip()} {d_inst}".strip() if d_inst else f"passage:  {d.strip()}".strip() for d in doc_texts]

    if doc_batch_size is None:
        doc_batch_size = batch_size
        if len(doc_texts) > 100000:
            doc_batch_size = min(16, batch_size)
            print(f"Large document set: using doc_batch_size={doc_batch_size}")
    else:
        print(f"Using custom document batch size: {doc_batch_size}")

    if auto_batch and device == "cuda" and not (use_multigpu and num_gpus > 1) and model is not None:
        print("Auto-tuning batch sizes...")
        optimal_doc_batch = auto_tune_batch_size(
            model, processed_documents[:500], device, max_length,
            start_batch=doc_batch_size, max_batch=128,
        )
        optimal_query_batch = auto_tune_batch_size(
            model, processed_queries[:100], device, max_length,
            start_batch=batch_size, max_batch=128,
        )
        if optimal_doc_batch != doc_batch_size:
            print(f"  Document batch size: {doc_batch_size} -> {optimal_doc_batch}")
            doc_batch_size = optimal_doc_batch
        if optimal_query_batch != batch_size:
            print(f"  Query batch size: {batch_size} -> {optimal_query_batch}")
            batch_size = optimal_query_batch

    doc_content_key = content_hash(processed_documents) if use_cache and cache_dir else ""
    query_content_key = content_hash(processed_queries) if use_cache and cache_dir else ""

    if use_cache and cache_dir:
        doc_cache_dir = os.path.join(
            cache_dir, "doc_emb", MODEL_NAME.replace("/", "--"), task_name,
            doc_content_key, f"batch_{doc_batch_size}_maxlen_{max_length}",
        )
        query_cache_dir = os.path.join(
            cache_dir, "query_emb", MODEL_NAME.replace("/", "--"), task_name,
            query_content_key, f"batch_{batch_size}_maxlen_{max_length}",
        )
        os.makedirs(doc_cache_dir, exist_ok=True)
        os.makedirs(query_cache_dir, exist_ok=True)
        doc_cache_file = os.path.join(doc_cache_dir, "embeddings.npy")
        query_cache_file = os.path.join(query_cache_dir, "embeddings.npy")
        print(f"Document cache key (content): {doc_content_key}")
        print(f"Query cache key (content): {query_content_key}")
    else:
        doc_cache_file = None
        query_cache_file = None

    with tools.benchmark(MODEL_NAME, "Embedding"):
        if device == "cuda":
            torch.cuda.empty_cache()

        inspect_batch_samples(processed_documents, doc_batch_size, name="documents")

        if use_cache and doc_cache_file and os.path.isfile(doc_cache_file):
            print(f"Loading cached document embeddings from {doc_cache_file}")
            doc_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            DOC_SHARD_SIZE = 10000
            use_sharded_doc_cache = use_cache and cache_dir is not None and len(processed_documents) > DOC_SHARD_SIZE
            if use_sharded_doc_cache:
                doc_shard_dir = os.path.join(doc_cache_dir, "shards")
                os.makedirs(doc_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded document cache, shard_size={DOC_SHARD_SIZE}")
                doc_emb_chunks = []
                total_docs = len(processed_documents)
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
                                print(f"Loading cached document shard {start_idx}-{end_idx-1}")
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
                        num_in_shard = end_idx - start_idx
                        print(f"Encoding document shard {start_idx}-{end_idx-1} ({num_in_shard} docs)...")
                        shard_texts = processed_documents[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME, shard_texts, doc_batch_size, max_length, num_gpus,
                            )
                        else:
                            with torch.inference_mode():
                                emb_chunk = model.encode(
                                    shard_texts,
                                    batch_size=doc_batch_size,
                                    max_length=max_length,
                                )
                        if use_cache:
                            _safe_save_npy(shard_path, emb_chunk)
                            print(f"Saved document shard {start_idx}-{end_idx-1} to cache")
                    doc_emb_chunks.append(emb_chunk)
                doc_emb = np.concatenate(doc_emb_chunks, axis=0)
                print(f"[DEBUG] Finished document encoding; doc_emb shape: {doc_emb.shape}")
                if use_cache and doc_cache_file:
                    _safe_save_npy(doc_cache_file, doc_emb)
            else:
                print(f"Encoding {len(processed_documents)} documents...")
                if use_multigpu and num_gpus > 1:
                    doc_emb = encode_with_multigpu(
                        MODEL_NAME, processed_documents, doc_batch_size, max_length, num_gpus,
                    )
                else:
                    with torch.inference_mode():
                        doc_emb = model.encode(
                            processed_documents,
                            batch_size=doc_batch_size,
                            max_length=max_length,
                        )
                print(f"[DEBUG] Finished document encoding; doc_emb shape: {doc_emb.shape}")
                if use_cache and doc_cache_file:
                    _safe_save_npy(doc_cache_file, doc_emb)

        if device == "cuda":
            torch.cuda.empty_cache()

        inspect_batch_samples(processed_queries, batch_size, name="queries")

        if use_cache and query_cache_file and os.path.isfile(query_cache_file):
            print(f"Loading cached query embeddings from {query_cache_file}")
            query_emb = np.load(query_cache_file, allow_pickle=True)
        else:
            QUERY_SHARD_SIZE = 2000
            use_sharded_query_cache = use_cache and cache_dir is not None and len(processed_queries) > QUERY_SHARD_SIZE
            if use_sharded_query_cache:
                query_shard_dir = os.path.join(query_cache_dir, "shards")
                os.makedirs(query_shard_dir, exist_ok=True)
                print(f"[DEBUG] Using sharded query cache, shard_size={QUERY_SHARD_SIZE}")
                query_emb_chunks = []
                total_queries = len(processed_queries)
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
                                print(f"Loading cached query shard {start_idx}-{end_idx-1}")
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
                        num_in_shard = end_idx - start_idx
                        print(f"Encoding query shard {start_idx}-{end_idx-1} ({num_in_shard} queries)...")
                        shard_queries = processed_queries[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME, shard_queries, batch_size, max_length, num_gpus,
                            )
                        else:
                            with torch.inference_mode():
                                emb_chunk = model.encode(
                                    shard_queries,
                                    batch_size=batch_size,
                                    max_length=max_length,
                                )
                        if use_cache:
                            _safe_save_npy(shard_path, emb_chunk)
                            print(f"Saved query shard {start_idx}-{end_idx-1} to cache")
                    query_emb_chunks.append(emb_chunk)
                query_emb = np.concatenate(query_emb_chunks, axis=0)
                if use_cache and query_cache_file:
                    _safe_save_npy(query_cache_file, query_emb)
            else:
                print(f"Encoding {len(processed_queries)} queries...")
                if use_multigpu and num_gpus > 1:
                    query_emb = encode_with_multigpu(
                        MODEL_NAME, processed_queries, batch_size, max_length, num_gpus,
                    )
                else:
                    with torch.inference_mode():
                        query_emb = model.encode(
                            processed_queries,
                            batch_size=batch_size,
                            max_length=max_length,
                        )
                print(f"[DEBUG] Finished query encoding; query_emb shape: {query_emb.shape}")
                if use_cache and query_cache_file:
                    _safe_save_npy(query_cache_file, query_emb)

    print("Computing similarity scores...")
    if use_faiss:
        print(f"[DEBUG] Creating FAISS index '{INDEX_NAME}'...")
        index = tools.create_index(INDEX_NAME, query_emb, doc_emb)
        scores, indices = tools.search_index(index, query_emb)
        print("[DEBUG] Finished FAISS search_index()")
    else:
        with tools.benchmark(MODEL_NAME, "Similarity"):
            query_tensor = torch.from_numpy(query_emb).to(device_obj)
            doc_tensor = torch.from_numpy(doc_emb).to(device_obj)
            scores_tensor = pairwise_cosine_similarity(query_tensor, doc_tensor)
            scores_np = scores_tensor.cpu().numpy()
            top_k = 100
            indices = np.argsort(scores_np, axis=1)[:, ::-1][:, :top_k]
            scores = np.take_along_axis(scores_np, indices, axis=1)

    print("[DEBUG] Starting retrieval and writing results...")
    tools.start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, indices, scores)
    print(f"[DEBUG] Finished retrieval. Results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Promptriever-7B for information retrieval (PEFT LLaMA, cache, multi-GPU, FAISS/direct)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python promptriever-7b.py
  python promptriever-7b.py --quest-plus
  python promptriever-7b.py --query-file ./data/QUEST/test_id_added.jsonl --corpus-file ./data/QUEST/documents.jsonl --output-file results.jsonl
  python promptriever-7b.py --quest-plus --max-docs 1000 --max-queries 50 --no-cache
        """,
    )
    parser.add_argument("--quest-plus", action="store_true", help="Use quest_plus mode")
    parser.add_argument("--query-file", type=str, default=None, help="Path to query JSONL")
    parser.add_argument("--corpus-file", type=str, default=None, help="Path to corpus JSONL")
    parser.add_argument("--output-file", type=str, default=None, help="Path to output results JSONL")
    parser.add_argument("--model-name", type=str, default=None, help="PEFT model name (default: samaya-ai/promptriever-llama2-7b-v1)")
    parser.add_argument("--index-name", type=str, default=None, help="FAISS index name/path")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for queries")
    parser.add_argument("--doc-batch-size", type=int, default=None, help="Batch size for documents")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16/bf16")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--no-faiss", action="store_true", help="Use direct similarity instead of FAISS")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length (default 512 for Promptriever)")
    parser.add_argument("--auto-batch", action="store_true", help="Auto-tune batch sizes")
    parser.add_argument("--multigpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit to first N documents")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--query-instruction", type=str, default=None, help="Query instruction text")
    parser.add_argument("--doc-instruction", type=str, default=None, help="Doc instruction text")
    parser.add_argument("--task-suffix", type=str, default=None, help="Suffix for cache task folder")
    parser.add_argument("--query-format", type=str, choices=["quest", "quest_plus"], default=None, help="Query file format")
    parser.add_argument("--corpus-format", type=str, choices=["quest", "quest_plus"], default=None, help="Corpus file format")

    args = parser.parse_args()
    query_fmt = args.query_format
    corpus_fmt = args.corpus_format
    if query_fmt is not None:
        query_fmt = query_fmt == "quest_plus"
    if corpus_fmt is not None:
        corpus_fmt = corpus_fmt == "quest_plus"
    device = args.device
    if device == "auto" or device is None:
        device = None

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
        query_instruction_raw=args.query_instruction,
        doc_instruction_raw=args.doc_instruction,
        task_suffix=args.task_suffix,
        query_format=query_fmt,
        corpus_format=corpus_fmt,
    )
