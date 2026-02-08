#!/usr/bin/env python
"""
E5-Mistral-7B-Instruct baseline: end-to-end retrieval on GPU (Snellius).

Full implementation aligned with reasonir-8b and gritlm-7b_main: CLI, cache,
sharded cache, multi-GPU, FAISS/direct similarity, tools.documents/tools.queries,
and tools.start_retrieval().
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

# Optional libs (for consistency with reasonir/gritlm)
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


# E5 instruct: document prefix is "passage: " (model adds it); query uses prompt_name="web_search_query"
# We format doc text as "title. text" to match build_quest_e5_index.
QUERY_PROMPT_NAME = "web_search_query"
DOC_PROMPT_NAME = None  # default passage behavior


def content_hash(texts):
    """Content-based hash for cache keys."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:24]


def inspect_batch_samples(texts, batch_size, name="items", max_samples=2):
    """Inspect batch contents when INSPECT_BATCHES=1."""
    if os.getenv("INSPECT_BATCHES", "0") != "1":
        return
    print(f"\n{'='*60}\nInspecting {name} batches (first {max_samples} batches)\n{'='*60}")
    for batch_idx in range(min(max_samples, (len(texts) + batch_size - 1) // batch_size)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        print(f"\nBatch {batch_idx + 1} (indices {start_idx}-{end_idx-1}, {len(batch_texts)} items):")
        for i, text in enumerate(batch_texts[:3]):
            preview = (text[:150] + "...") if len(text) > 150 else text
            print(f"  [{i}] {preview}")
        if len(batch_texts) > 3:
            print(f"  ... and {len(batch_texts) - 3} more")
    print(f"{'='*60}\n")


def auto_tune_batch_size(model, texts, device, max_length, prompt_name, start_batch=8, max_batch=64):
    """Auto-tune batch size for E5 on GPU."""
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
            kwargs = dict(
                batch_size=test_batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if prompt_name:
                kwargs["prompt_name"] = prompt_name
            with torch.inference_mode():
                _ = model.encode(test_texts, **kwargs)
            optimal_batch = test_batch
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                torch.cuda.empty_cache()
                break
            break
    return optimal_batch


def encode_worker(
    gpu_id,
    model_name,
    texts_chunk,
    prompt_name,
    batch_size,
    max_length,
    result_queue,
    temp_dir,
):
    """Worker: load SentenceTransformer on one GPU, encode chunk, write npy shard."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch as th
        th.cuda.empty_cache()
        th.cuda.set_device(0)

        from sentence_transformers import SentenceTransformer

        print(f"[GPU {gpu_id}] Loading E5-Mistral on cuda:0 (physical GPU {gpu_id})...")
        model = SentenceTransformer(model_name, device="cuda:0")
        model.max_seq_length = max_length
        model.eval()

        memory_used = th.cuda.memory_allocated(0) / 1024**3
        print(f"[GPU {gpu_id}] Model loaded (memory: {memory_used:.2f} GB)")

        n_batches = (len(texts_chunk) + batch_size - 1) // batch_size
        emb_list = []
        kwargs = dict(
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if prompt_name:
            kwargs["prompt_name"] = prompt_name

        with th.inference_mode():
            for start in tqdm(
                range(0, len(texts_chunk), batch_size),
                desc=f"[GPU {gpu_id}] Batches",
                total=n_batches,
                unit="batch",
                file=sys.stderr,
            ):
                batch_texts = texts_chunk[start : start + batch_size]
                emb = model.encode(batch_texts, **kwargs)
                emb_list.append(np.asarray(emb))
        embeddings = np.concatenate(emb_list, axis=0)

        os.makedirs(temp_dir, exist_ok=True)
        shard_path = os.path.join(temp_dir, f"gpu_{gpu_id}.npy")
        np.save(shard_path, embeddings.astype(np.float32))
        result_queue.put((gpu_id, shard_path))

        del model
        import gc
        gc.collect()
        th.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] ERROR: {e}\n{traceback.format_exc()}")
        result_queue.put((gpu_id, None))


def encode_with_multigpu(
    model_name,
    texts,
    prompt_name,
    batch_size,
    max_length,
    num_gpus,
):
    """Encode texts across multiple GPUs (data parallelism, spawn one process per GPU)."""
    if num_gpus <= 1:
        torch.cuda.set_device(0)
        model = SentenceTransformer(model_name, device="cuda:0")
        model.max_seq_length = max_length
        model.eval()
        kwargs = dict(
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        with torch.inference_mode():
            result = model.encode(texts, **kwargs)
        del model
        torch.cuda.empty_cache()
        return np.asarray(result, dtype=np.float32)

    print(f"Splitting {len(texts)} texts across {num_gpus} GPUs for parallel encoding...")
    chunk_size = math.ceil(len(texts) / num_gpus)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    while len(chunks) < num_gpus:
        chunks.append([])

    result_queue = Queue()
    temp_dir = tempfile.mkdtemp(prefix="e5_mistral_mgpu_")
    processes = []
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
        p = Process(
            target=encode_worker,
            args=(
                gpu_id,
                model_name,
                chunks[gpu_id],
                prompt_name,
                batch_size,
                max_length,
                result_queue,
                temp_dir,
            ),
        )
        p.start()
        processes.append((gpu_id, p))

    results_dict = {}
    for _ in range(len(processes)):
        gpu_id_result, shard_path = result_queue.get()
        if shard_path is not None:
            results_dict[gpu_id_result] = shard_path
        else:
            raise RuntimeError(f"GPU {gpu_id_result} process failed")
    for gpu_id, p in processes:
        p.join()

    loaded = [np.load(results_dict[i], allow_pickle=True) for i in range(num_gpus) if i in results_dict]
    final_emb = np.concatenate(loaded, axis=0) if loaded else np.array([]).astype(np.float32)
    return final_emb


def main(
    quest_plus=False,
    query_file=None,
    corpus_file=None,
    output_file=None,
    model_name=None,
    index_name=None,
    device=None,
    batch_size=8,
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
    task_suffix=None,
    query_format=None,
    corpus_format=None,
):
    MODEL_NAME = model_name if model_name else "intfloat/e5-mistral-7b-instruct"
    INDEX_NAME = index_name if index_name else "E5-Mistral-7B-QUEST"

    if cache_dir is None:
        cache_dir = (
            os.path.join(os.path.dirname(output_file) if output_file else ".", "cache")
            if use_cache
            else None
        )
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
            print("WARNING: use_multigpu=True but only 1 GPU. Using single GPU.")
        num_gpus = 1
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"Using cache: {use_cache}")
    if use_cache:
        print(f"Cache directory: {cache_dir}")
    print(f"Using FAISS: {use_faiss}")

    if quest_plus:
        QUERY_FILE = query_file or "./data/QUEST_w_Varients/quest_test_withVarients.jsonl"
        CORPUS_FILE = corpus_file or "./data/QUEST_w_Varients/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = output_file or "results_plus.jsonl"
    else:
        QUERY_FILE = query_file or "./data/QUEST/test_id_added.jsonl"
        CORPUS_FILE = corpus_file or "./data/QUEST/documents.jsonl"
        OUTPUT_FILE = output_file or "results.jsonl"

    print("Collecting documents...")
    print(f"Corpus file: {CORPUS_FILE}")
    print(f"Query file: {QUERY_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    corpus_quest_plus = corpus_format if corpus_format is not None else quest_plus
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, corpus_quest_plus)
    # E5 format: "title. text"
    doc_texts_for_encode = [
        f"{doc_title_map.get(did, '')}. {t}".strip() for did, t in zip(doc_ids, doc_texts)
    ]
    print(f"Total documents loaded: {len(doc_ids)}")

    if max_docs is not None and max_docs > 0 and max_docs < len(doc_ids):
        doc_ids = doc_ids[:max_docs]
        doc_texts_for_encode = doc_texts_for_encode[:max_docs]
        doc_title_map = {k: doc_title_map.get(k) for k in doc_ids}
        print(f"Reduced to first {max_docs} documents")

    print("Collecting queries...")
    query_quest_plus = query_format if query_format is not None else quest_plus
    query, ground_truths = tools.queries(QUERY_FILE, query_quest_plus)
    print(f"Total queries loaded: {len(query)}")

    if max_queries is not None and max_queries > 0 and max_queries < len(query):
        query = query[:max_queries]
        ground_truths = ground_truths[:max_queries]
        print(f"Reduced to first {max_queries} queries")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if use_multigpu and num_gpus > 1:
        model = None
        print("Multi-GPU: model will be loaded per GPU during encoding")
    else:
        print(f"Loading model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = max_length
        model.eval()
        print("Model loaded.")

    if doc_batch_size is None:
        doc_batch_size = batch_size
        if len(doc_texts_for_encode) > 100000:
            doc_batch_size = min(8, batch_size)
            print(f"Large corpus: using doc_batch_size={doc_batch_size}")
    print(f"doc_batch_size={doc_batch_size}, batch_size={batch_size}")

    if auto_batch and device == "cuda" and not (use_multigpu and num_gpus > 1):
        print("Auto-tuning batch sizes...")
        optimal_doc = auto_tune_batch_size(
            model, doc_texts_for_encode[:500], device, max_length, DOC_PROMPT_NAME,
            start_batch=doc_batch_size, max_batch=64,
        )
        optimal_q = auto_tune_batch_size(
            model, query[:100], device, max_length, QUERY_PROMPT_NAME,
            start_batch=batch_size, max_batch=64,
        )
        doc_batch_size = optimal_doc
        batch_size = optimal_q
        print(f"  doc_batch_size={doc_batch_size}, batch_size={batch_size}")

    doc_content_key = content_hash(doc_texts_for_encode) if use_cache and cache_dir else ""
    query_content_key = content_hash(query) if use_cache and cache_dir else ""

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
    else:
        doc_cache_file = None
        query_cache_file = None

    with tools.benchmark(MODEL_NAME, "Embedding"):
        if device == "cuda":
            torch.cuda.empty_cache()
        inspect_batch_samples(doc_texts_for_encode, doc_batch_size, name="documents")

        if use_cache and doc_cache_file and os.path.isfile(doc_cache_file):
            print(f"Loading cached document embeddings from {doc_cache_file}")
            doc_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            DOC_SHARD_SIZE = 10000
            use_sharded_doc = use_cache and cache_dir and len(doc_texts_for_encode) > DOC_SHARD_SIZE
            if use_sharded_doc:
                doc_shard_dir = os.path.join(doc_cache_dir, "shards")
                os.makedirs(doc_shard_dir, exist_ok=True)
                doc_emb_chunks = []
                for start_idx in range(0, len(doc_texts_for_encode), DOC_SHARD_SIZE):
                    end_idx = min(start_idx + DOC_SHARD_SIZE, len(doc_texts_for_encode))
                    shard_path = os.path.join(doc_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        doc_emb_chunks.append(np.load(shard_path, allow_pickle=True))
                    else:
                        shard_texts = doc_texts_for_encode[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME, shard_texts, DOC_PROMPT_NAME,
                                doc_batch_size, max_length, num_gpus,
                            )
                        else:
                            kwargs = dict(
                                batch_size=doc_batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                            )
                            if DOC_PROMPT_NAME:
                                kwargs["prompt_name"] = DOC_PROMPT_NAME
                            with torch.inference_mode():
                                emb_chunk = model.encode(shard_texts, **kwargs)
                            emb_chunk = np.asarray(emb_chunk, dtype=np.float32)
                        if use_cache:
                            np.save(shard_path, emb_chunk)
                        doc_emb_chunks.append(emb_chunk)
                doc_emb = np.concatenate(doc_emb_chunks, axis=0)
                if use_cache and doc_cache_file:
                    np.save(doc_cache_file, doc_emb)
            else:
                if use_multigpu and num_gpus > 1:
                    doc_emb = encode_with_multigpu(
                        MODEL_NAME, doc_texts_for_encode, DOC_PROMPT_NAME,
                        doc_batch_size, max_length, num_gpus,
                    )
                else:
                    kwargs = dict(
                        batch_size=doc_batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    if DOC_PROMPT_NAME:
                        kwargs["prompt_name"] = DOC_PROMPT_NAME
                    with torch.inference_mode():
                        doc_emb = model.encode(doc_texts_for_encode, **kwargs)
                    doc_emb = np.asarray(doc_emb, dtype=np.float32)
                if use_cache and doc_cache_file:
                    np.save(doc_cache_file, doc_emb)
            print(f"Document embeddings shape: {doc_emb.shape}")

        if device == "cuda":
            torch.cuda.empty_cache()
        inspect_batch_samples(query, batch_size, name="queries")

        if use_cache and query_cache_file and os.path.isfile(query_cache_file):
            print(f"Loading cached query embeddings from {query_cache_file}")
            query_emb = np.load(query_cache_file, allow_pickle=True)
        else:
            QUERY_SHARD_SIZE = 2000
            use_sharded_q = use_cache and cache_dir and len(query) > QUERY_SHARD_SIZE
            if use_sharded_q:
                query_shard_dir = os.path.join(query_cache_dir, "shards")
                os.makedirs(query_shard_dir, exist_ok=True)
                query_emb_chunks = []
                for start_idx in range(0, len(query), QUERY_SHARD_SIZE):
                    end_idx = min(start_idx + QUERY_SHARD_SIZE, len(query))
                    shard_path = os.path.join(query_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        query_emb_chunks.append(np.load(shard_path, allow_pickle=True))
                    else:
                        shard_q = query[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb_chunk = encode_with_multigpu(
                                MODEL_NAME, shard_q, QUERY_PROMPT_NAME,
                                batch_size, max_length, num_gpus,
                            )
                        else:
                            kwargs = dict(
                                batch_size=batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                prompt_name=QUERY_PROMPT_NAME,
                            )
                            with torch.inference_mode():
                                emb_chunk = model.encode(shard_q, **kwargs)
                            emb_chunk = np.asarray(emb_chunk, dtype=np.float32)
                        if use_cache:
                            np.save(shard_path, emb_chunk)
                        query_emb_chunks.append(emb_chunk)
                query_emb = np.concatenate(query_emb_chunks, axis=0)
                if use_cache and query_cache_file:
                    np.save(query_cache_file, query_emb)
            else:
                if use_multigpu and num_gpus > 1:
                    query_emb = encode_with_multigpu(
                        MODEL_NAME, query, QUERY_PROMPT_NAME,
                        batch_size, max_length, num_gpus,
                    )
                else:
                    kwargs = dict(
                        batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        prompt_name=QUERY_PROMPT_NAME,
                    )
                    with torch.inference_mode():
                        query_emb = model.encode(query, **kwargs)
                    query_emb = np.asarray(query_emb, dtype=np.float32)
                if use_cache and query_cache_file:
                    np.save(query_cache_file, query_emb)
            print(f"Query embeddings shape: {query_emb.shape}")

    print("Computing similarity...")
    if use_faiss:
        index = tools.create_index(INDEX_NAME, query_emb, doc_emb)
        scores, indices = tools.search_index(index, query_emb)
    else:
        query_tensor = torch.from_numpy(query_emb).to(device_obj)
        doc_tensor = torch.from_numpy(doc_emb).to(device_obj)
        scores_tensor = pairwise_cosine_similarity(query_tensor, doc_tensor)
        scores_np = scores_tensor.cpu().numpy()
        top_k = 100
        indices = np.argsort(scores_np, axis=1)[:, ::-1][:, :top_k]
        scores = np.take_along_axis(scores_np, indices, axis=1)

    tools.start_retrieval(OUTPUT_FILE, query, ground_truths, doc_ids, doc_title_map, indices, scores)
    print(f"Results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E5-Mistral-7B-Instruct retrieval (GPU, cache, multigpu, FAISS/direct)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quest-plus", action="store_true", help="Use quest_plus paths")
    parser.add_argument("--query-file", type=str, default=None)
    parser.add_argument("--corpus-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--doc-batch-size", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--multigpu", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--task-suffix", type=str, default=None)
    parser.add_argument("--query-format", type=str, choices=["quest", "quest_plus"], default=None)
    parser.add_argument("--corpus-format", type=str, choices=["quest", "quest_plus"], default=None)
    args = parser.parse_args()

    query_fmt = args.query_format
    corpus_fmt = args.corpus_format
    if query_fmt is not None:
        query_fmt = query_fmt == "quest_plus"
    if corpus_fmt is not None:
        corpus_fmt = corpus_fmt == "quest_plus"
    device = None if (args.device == "auto" or args.device is None) else args.device

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
        use_fp16=not getattr(args, "no_fp16", False),
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
