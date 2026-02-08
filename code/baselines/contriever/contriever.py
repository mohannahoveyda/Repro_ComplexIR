"""
Contriever baseline for QUEST retrieval (facebook/contriever or contriever-msmarco).
Full implementation aligned with other baselines: CLI, cache, sharding, multi-GPU, FAISS/direct similarity.
Uses mean pooling over token embeddings + L2 normalize (no instruction prefix).
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
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


# --- Contriever: mean pooling over token embeddings ---

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling over non-padding tokens (Contriever standard)."""
    mask = attention_mask.unsqueeze(-1).float()
    token_embeddings = token_embeddings.masked_fill(mask == 0, 0.0)
    sum_emb = token_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / sum_mask


class Contriever:
    """Contriever: encode with mean pooling + L2 normalize. No instruction prefix."""

    def __init__(self, model_name_or_path, device=None, torch_dtype="auto"):
        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.model.eval()
        self.model = self.model.to(self._device)
        self._hidden_size = getattr(self.model.config, "hidden_size", 768)

    def encode(self, sentences, max_length=512, batch_size=32):
        if not sentences:
            return np.zeros((0, self._hidden_size), dtype=np.float32)
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                pooled = mean_pooling(last_hidden, inputs["attention_mask"])
                embeddings = F.normalize(pooled, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


def content_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:24]


def inspect_batch_samples(texts, batch_size, name="items", max_samples=2):
    if os.getenv("INSPECT_BATCHES", "0") != "1":
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


def auto_tune_batch_size(model, texts, device, max_length, start_batch=64, max_batch=512):
    if device != "cuda":
        return start_batch
    model.eval()
    optimal_batch = start_batch
    for test_batch in [start_batch, min(start_batch * 2, max_batch), min(start_batch * 4, max_batch)]:
        try:
            torch.cuda.empty_cache()
            test_size = min(test_batch, len(texts), 100)
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
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        model = Contriever(model_name, device=torch.device("cuda:0"))
        n_batches = (len(texts_chunk) + batch_size - 1) // batch_size
        emb_list = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts_chunk), batch_size), desc=f"[GPU {gpu_id}]", total=n_batches, file=sys.stderr):
                batch_texts = texts_chunk[start : start + batch_size]
                emb = model.encode(batch_texts, batch_size=len(batch_texts), max_length=max_length)
                emb_list.append(emb)
        embeddings = np.concatenate(emb_list, axis=0)
        os.makedirs(temp_dir, exist_ok=True)
        shard_path = os.path.join(temp_dir, f"gpu_{gpu_id}.npy")
        np.save(shard_path, embeddings)
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
        return None
    chunk_size = math.ceil(len(texts) / num_gpus)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    while len(chunks) < num_gpus:
        chunks.append([])
    result_queue = Queue()
    temp_dir = tempfile.mkdtemp(prefix="contriever_mgpu_")
    processes = []
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
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
    loaded = [np.load(results_dict[i], allow_pickle=True) for i in range(num_gpus) if i in results_dict]
    return np.concatenate(loaded, axis=0) if loaded else np.array([])


def main(
    quest_plus=False,
    query_file=None,
    corpus_file=None,
    output_file=None,
    model_name=None,
    index_name=None,
    device=None,
    batch_size=64,
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
    MODEL_NAME = model_name if model_name else "facebook/contriever"
    INDEX_NAME = index_name if index_name else "Contriever-QUEST"

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_file) if output_file else ".", "cache") if use_cache else None
    task_name = ("quest_plus" if quest_plus else "quest") + (task_suffix or "")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if use_multigpu and num_gpus > 1:
        pass  # multi-GPU
    else:
        num_gpus = 1
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if quest_plus:
        QUERY_FILE = query_file if query_file else "./data/QUEST_w_Varients/quest_test_withVarients.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST_w_Varients/quest_text_w_id_withVarients.jsonl"
        OUTPUT_FILE = output_file if output_file else "results_plus.jsonl"
    else:
        QUERY_FILE = query_file if query_file else "./data/QUEST/test_id_added.jsonl"
        CORPUS_FILE = corpus_file if corpus_file else "./data/QUEST/documents.jsonl"
        OUTPUT_FILE = output_file if output_file else "results.jsonl"

    print("Collecting documents...")
    print(f"Corpus: {CORPUS_FILE}, Query: {QUERY_FILE}, Output: {OUTPUT_FILE}")
    print(f"Model: {MODEL_NAME}, Device: {device}, Batch size: {batch_size}")

    corpus_quest_plus = corpus_format if corpus_format is not None else quest_plus
    doc_ids, doc_texts, doc_title_map = tools.documents(CORPUS_FILE, corpus_quest_plus)
    print(f"Total documents: {len(doc_ids)}")

    if max_docs is not None and max_docs > 0 and max_docs < len(doc_ids):
        doc_ids, doc_texts = doc_ids[:max_docs], doc_texts[:max_docs]
        doc_title_map = {doc_id: doc_title_map.get(doc_id) for doc_id in doc_ids}
        print(f"Reduced to {len(doc_ids)} documents")

    query_quest_plus = query_format if query_format is not None else quest_plus
    query, ground_truths = tools.queries(QUERY_FILE, query_quest_plus)
    print(f"Total queries: {len(query)}")

    if max_queries is not None and max_queries > 0 and max_queries < len(query):
        query, ground_truths = query[:max_queries], ground_truths[:max_queries]
        print(f"Reduced to {len(query)} queries")

    # Contriever: no instruction prefix, use raw text
    processed_documents = [d.strip() for d in doc_texts]
    processed_queries = [q.strip() for q in query]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if use_multigpu and num_gpus > 1:
        model = None
    else:
        print(f"Loading model: {MODEL_NAME}...")
        model = Contriever(MODEL_NAME, device=device_obj)
        print("Model loaded.")

    if doc_batch_size is None:
        doc_batch_size = batch_size
        if len(doc_texts) > 100000:
            doc_batch_size = min(32, batch_size)
    if auto_batch and device == "cuda" and not (use_multigpu and num_gpus > 1) and model is not None:
        doc_batch_size = auto_tune_batch_size(model, processed_documents[:500], device, max_length, start_batch=doc_batch_size)
        batch_size = auto_tune_batch_size(model, processed_queries[:100], device, max_length, start_batch=batch_size)

    doc_content_key = content_hash(processed_documents) if use_cache and cache_dir else ""
    query_content_key = content_hash(processed_queries) if use_cache and cache_dir else ""
    if use_cache and cache_dir:
        doc_cache_dir = os.path.join(cache_dir, "doc_emb", MODEL_NAME.replace("/", "--"), task_name, doc_content_key, f"batch_{doc_batch_size}_maxlen_{max_length}")
        query_cache_dir = os.path.join(cache_dir, "query_emb", MODEL_NAME.replace("/", "--"), task_name, query_content_key, f"batch_{batch_size}_maxlen_{max_length}")
        os.makedirs(doc_cache_dir, exist_ok=True)
        os.makedirs(query_cache_dir, exist_ok=True)
        doc_cache_file = os.path.join(doc_cache_dir, "embeddings.npy")
        query_cache_file = os.path.join(query_cache_dir, "embeddings.npy")
    else:
        doc_cache_file = query_cache_file = None

    with tools.benchmark(MODEL_NAME, "Embedding"):
        if device == "cuda":
            torch.cuda.empty_cache()
        inspect_batch_samples(processed_documents, doc_batch_size, name="documents")

        if use_cache and doc_cache_file and os.path.isfile(doc_cache_file):
            print(f"Loading cached document embeddings from {doc_cache_file}")
            doc_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            DOC_SHARD_SIZE = 10000
            use_sharded = use_cache and cache_dir and len(processed_documents) > DOC_SHARD_SIZE
            if use_sharded:
                doc_shard_dir = os.path.join(doc_cache_dir, "shards")
                os.makedirs(doc_shard_dir, exist_ok=True)
                doc_emb_chunks = []
                for start_idx in range(0, len(processed_documents), DOC_SHARD_SIZE):
                    end_idx = min(start_idx + DOC_SHARD_SIZE, len(processed_documents))
                    shard_path = os.path.join(doc_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        doc_emb_chunks.append(np.load(shard_path, allow_pickle=True))
                    else:
                        shard_texts = processed_documents[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb = encode_with_multigpu(MODEL_NAME, shard_texts, doc_batch_size, max_length, num_gpus)
                        else:
                            with torch.inference_mode():
                                emb = model.encode(shard_texts, batch_size=doc_batch_size, max_length=max_length)
                        if use_cache:
                            np.save(shard_path, emb)
                        doc_emb_chunks.append(emb)
                doc_emb = np.concatenate(doc_emb_chunks, axis=0)
                if use_cache and doc_cache_file:
                    np.save(doc_cache_file, doc_emb)
            else:
                if use_multigpu and num_gpus > 1:
                    doc_emb = encode_with_multigpu(MODEL_NAME, processed_documents, doc_batch_size, max_length, num_gpus)
                else:
                    with torch.inference_mode():
                        doc_emb = model.encode(processed_documents, batch_size=doc_batch_size, max_length=max_length)
                if use_cache and doc_cache_file:
                    np.save(doc_cache_file, doc_emb)
        print(f"Document embeddings shape: {doc_emb.shape}")

        if device == "cuda":
            torch.cuda.empty_cache()
        inspect_batch_samples(processed_queries, batch_size, name="queries")

        if use_cache and query_cache_file and os.path.isfile(query_cache_file):
            print(f"Loading cached query embeddings from {query_cache_file}")
            query_emb = np.load(query_cache_file, allow_pickle=True)
        else:
            QUERY_SHARD_SIZE = 2000
            use_sharded_q = use_cache and cache_dir and len(processed_queries) > QUERY_SHARD_SIZE
            if use_sharded_q:
                query_shard_dir = os.path.join(query_cache_dir, "shards")
                os.makedirs(query_shard_dir, exist_ok=True)
                query_emb_chunks = []
                for start_idx in range(0, len(processed_queries), QUERY_SHARD_SIZE):
                    end_idx = min(start_idx + QUERY_SHARD_SIZE, len(processed_queries))
                    shard_path = os.path.join(query_shard_dir, f"emb_{start_idx:07d}_{end_idx:07d}.npy")
                    if os.path.isfile(shard_path):
                        query_emb_chunks.append(np.load(shard_path, allow_pickle=True))
                    else:
                        shard_texts = processed_queries[start_idx:end_idx]
                        if use_multigpu and num_gpus > 1:
                            emb = encode_with_multigpu(MODEL_NAME, shard_texts, batch_size, max_length, num_gpus)
                        else:
                            with torch.inference_mode():
                                emb = model.encode(shard_texts, batch_size=batch_size, max_length=max_length)
                        if use_cache:
                            np.save(shard_path, emb)
                        query_emb_chunks.append(emb)
                query_emb = np.concatenate(query_emb_chunks, axis=0)
                if use_cache and query_cache_file:
                    np.save(query_cache_file, query_emb)
            else:
                if use_multigpu and num_gpus > 1:
                    query_emb = encode_with_multigpu(MODEL_NAME, processed_queries, batch_size, max_length, num_gpus)
                else:
                    with torch.inference_mode():
                        query_emb = model.encode(processed_queries, batch_size=batch_size, max_length=max_length)
                if use_cache and query_cache_file:
                    np.save(query_cache_file, query_emb)
        print(f"Query embeddings shape: {query_emb.shape}")

    print("Computing similarity scores...")
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
    parser = argparse.ArgumentParser(description="Run Contriever for QUEST retrieval (cache, multi-GPU, FAISS/direct)")
    parser.add_argument("--quest-plus", action="store_true", help="Use quest_plus mode")
    parser.add_argument("--query-file", type=str, default=None)
    parser.add_argument("--corpus-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="Default: facebook/contriever. Use facebook/contriever-msmarco for MS MARCO finetuned.")
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--doc-batch-size", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
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
    query_fmt = args.query_format == "quest_plus" if args.query_format is not None else None
    corpus_fmt = args.corpus_format == "quest_plus" if args.corpus_format is not None else None
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
