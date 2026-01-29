import os
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import random
from pathlib import Path
import torch

from logutils.setup import setup_logging  
from evaluate import precision_at_k, recall_at_k, f1_at_k, reciprocal_rank, ndcg_at_k
from helpers.data_loading import load_and_sample_data

from parser.problog_parser import ProblogParser
from estimator.problog_estimator import ProblogEstimator
from reasoner.problog_reasoner import ProbLogReasoner
from reasoner.rag_reasoner import RAGReasoner
from helpers.cli_setup import parse_cli_arguments


# ── Setup ────────────────────────────────────────────────────────────
MAX_TOKENS  = 500
TEMPERATURE = 0.0

start = datetime.now()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
print("TORCH AVAILABLE:", torch.cuda.is_available())




if __name__ == "__main__":
    
    args = parse_cli_arguments()
    if args.retriever:
        retriever = args.retriever
    else:
        retriever = Path(args.data_path).parent.name
    
    # outdir = Path(args.outdir) / retriever
    outdir = Path(args.outdir) 
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"OUTDIR: {outdir}")

    MODELS   = [args.model_name]
    CONTEXTS = ["atom",
    "wiki",
    # "wd",
    # "full"
    ]
    Ks       = [1, 3, 5, 10, 20]
    detail_rows = []

    reporter, log = setup_logging(log_dir=args.log_dir, mode="md")
    log("Log directory: %s", args.log_dir)

    # load & balance‐sample data
    sampled = load_and_sample_data(args.data_path, args.n_instances)
    log(f"Len Sampled: {len(sampled)}")
    if args.end is None:
        args.end = len(sampled)
    sampled = sampled[args.start:args.end]
    log(f"Processing samples {args.start} to {args.end} (count={len(sampled)})")

    if args.parse_only:
        parser_instance = ProblogParser(
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            prompts_path = args.prompts_path,
            quantize     = args.quantize,
        )
        kept = 0
        for entry in tqdm(sampled, desc="Parsing only gold-hit queries"):
            docs = entry["docs"]
            gold = set(entry.get("gold_docs", []))
            if gold.intersection(docs):
                parser_instance.parse_query(entry["query"])
                kept += 1
        print(f"Parsed and cached {kept}/{len(sampled)} queries (those with ≥1 gold hit); exiting.")
        exit(0)
    if args.baseline_only:
        detail_rows = []
        metrics = { args.retriever.lower(): { **{f"p@{k}":[] for k in Ks},
                                            **{f"r@{k}":[] for k in Ks},
                                            **{f"f1@{k}":[] for k in Ks},
                                            **{f"ndcg@{k}":[] for k in Ks},
                                            "mrr":[] }
                    }

        for entry in sampled:
            docs  = entry["docs"]
            scores= entry["scores"]
            gold  = set(entry["gold_docs"])
            baseline_rank = list(zip(docs, scores))

            # collect detail rows
            for rank,(doc,score) in enumerate(baseline_rank,1):
                detail_rows.append({
                "retriever": retriever,
                "query_id": entry.get("id", ""),
                "method": retriever.lower(),
                "entity": doc,
                "score": score,
                "rank": rank,
                "is_gold": int(doc in gold),
                })

            # compute metrics
            preds = [d for d,_ in baseline_rank]
            for k in Ks:
                metrics[retriever.lower()][f"p@{k}"].append(precision_at_k(preds, gold, k))
                metrics[retriever.lower()][f"r@{k}"].append(recall_at_k   (preds, gold, k))
                metrics[retriever.lower()][f"f1@{k}"].append(f1_at_k       (preds, gold, k))
                metrics[retriever.lower()][f"ndcg@{k}"].append(ndcg_at_k   (preds, gold, k))
            metrics[retriever.lower()]["mrr"].append(reciprocal_rank(preds, gold))

        agg = {"method": retriever.lower()}
        for k in Ks:
            agg[f"{retriever.lower()}_P@{k}"]   = np.mean(metrics[retriever.lower()][f"p@{k}"])
            agg[f"{retriever.lower()}_R@{k}"]   = np.mean(metrics[retriever.lower()][f"r@{k}"])
            agg[f"{retriever.lower()}_F1@{k}"]  = np.mean(metrics[retriever.lower()][f"f1@{k}"])
            agg[f"{retriever.lower()}_nDCG@{k}"] = np.mean(metrics[retriever.lower()][f"ndcg@{k}"])
        agg[f"{retriever.lower()}_MRR"] = np.mean(metrics[retriever.lower()]["mrr"])

        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(outdir/"detail_baseline.csv", index=False)
        pd.DataFrame([agg]).to_csv(outdir/"baseline_summary.csv", index=False)
        print("Baseline evaluation complete.")
        exit(0)

    parser_instance = ProblogParser(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        prompts_path=args.prompts_path,
        quantize=args.quantize,
    )


    with open(args.prompts_path) as f:
        parse_template = json.load(f)["scheme_1"].rstrip()



    all_results = []
    for model_name in MODELS:
        for context in CONTEXTS:
            reporter.report(
            "start_run",
                retriever=retriever,
                llm=model_name,
                method=args.method,
                context=context,
            )

            estimator = ProblogEstimator(
                model_name=model_name,
                prompts_path=args.prompts_path,
                quantize=args.quantize,
                context_mode=context,
                method=args.method,
            )
            reasoner = ProbLogReasoner(
                parser=parser_instance,
                estimator=estimator,
                log_dir=args.log_dir,
                reporter=reporter,
            )

            rag_reasoner = RAGReasoner(
                model_name=model_name,
                prompts_path="configs/RAG/prompts.json",
                quantize=args.quantize,
                context_mode=context,
                experiment_mode="io", # or cot
                reporter=reporter,
                log_dir=args.log_dir,
            )
            metrics = {
                rm: {
                    **{f"p@{k}": [] for k in Ks},
                    **{f"r@{k}": [] for k in Ks},
                    **{f"f1@{k}": [] for k in Ks},
                    **{f"ndcg@{k}": [] for k in Ks},
                    "mrr": []
                }
                for rm in ("prob", retriever.lower(), "rag")
            }
            token_metrics = {
                "prob": {"in": [], "out": []},
                "rag":  {"in": [], "out": []},
            }

            for idx, entry in enumerate(sampled, start=1):
                q        = entry["query"]
                query_id = entry.get("id", f"Query #{idx}")
                gold     = set(entry.get("gold_docs", []))

                # report query index & parse
                reporter.report(
                    "query_index",
                    msg=f"Query ID: {query_id}, {idx}/{len(sampled)}",
                    template=entry["metadata"]["template"]
                )
                if gold.intersection(set(entry['docs'])):
                    reporter.report("parse_query", original_query=q, parsed=parser_instance.parse_query(q))

                baseline_rank = list(zip(entry["docs"], entry["scores"]))


                # The following if-statement will only execute if the base retriever (BM25 or E5) has retrieved at least 1 gold entity. In other words, if there are no gold entities retrieved, we use the base retriever's ranking for evaluation and will not perform any form of reasoning (Neither ProbLog nor RAG)
                if gold.intersection(set(entry['docs'])):
                    try:
                        # prob_rank = reasoner.rank_entities(entry)
                        prob_rank, PROB_in_tokens_count_list, PROB_out_tokens_count_list = reasoner.rank_entities(entry)
                    except Exception as e:
                        log(f"ProbLog failed on {query_id}: {e}. Falling back to {retriever.upper()}.")
                        prob_rank = baseline_rank

                    # 4) RAG ranking with fallback
                    try:
                        rag_rank, RAG_in_tokens_count_list, RAG_out_tokens_count_list = rag_reasoner.rank_entities(entry)
                    except Exception as e:
                        log(f"RAG failed on {query_id}: {e}. Falling back to {retriever.upper()}.")
                        rag_rank = baseline_rank

                else:
                    log(f"\n--->>> Skipping due to 0 intersection with GOLD")
                    prob_rank = baseline_rank
                    rag_rank = baseline_rank
                    PROB_in_tokens_count_list  = []
                    PROB_out_tokens_count_list = []
                    RAG_in_tokens_count_list   = []
                    RAG_out_tokens_count_list  = []

                # prob_entity_in_sums  = [ sum(atom_list) for atom_list in PROB_in_tokens_count_list ]
                # prob_entity_out_sums = [ sum(atom_list) for atom_list in PROB_out_tokens_count_list ]
                # avg_prob_in  = float(np.mean(prob_entity_in_sums )) if prob_entity_in_sums  else 0.0
                # avg_prob_out = float(np.mean(prob_entity_out_sums)) if prob_entity_out_sums else 0.0
                avg_prob_in  = float(np.mean(PROB_in_tokens_count_list )) if PROB_in_tokens_count_list   else 0.0
                avg_prob_out = float(np.mean(PROB_out_tokens_count_list)) if PROB_out_tokens_count_list else 0.0
                avg_rag_in  = float(np.mean(RAG_in_tokens_count_list )) if RAG_in_tokens_count_list  else 0.0
                avg_rag_out = float(np.mean(RAG_out_tokens_count_list)) if RAG_out_tokens_count_list else 0.0

                token_metrics["prob"]["in"].append (avg_prob_in)
                token_metrics["prob"]["out"].append(avg_prob_out)
                token_metrics["rag"]["in"].append  (avg_rag_in)
                token_metrics["rag"]["out"].append (avg_rag_out)

                token_in_map  = {
                    "prob": PROB_in_tokens_count_list,
                    "rag":  RAG_in_tokens_count_list
                    }
                token_out_map = {
                    "prob": PROB_out_tokens_count_list,
                    "rag":  RAG_out_tokens_count_list
                    }

                log(f"TOKEN_IN_MAP: {token_in_map}")
                log(f"TOKEN_OUT_MAP: {token_out_map}")
                # 5) collect detail_rows for all three methods
                for method, ranking in (
                    ("prob", prob_rank),
                    (retriever.lower(), baseline_rank),
                    ("rag",  rag_rank),
                ):

                    for rank, (entity, score) in enumerate(ranking, start=1):
                        in_tokens  = None
                        out_tokens = None
                        if method in token_in_map:
                            lst = token_in_map[method]
                            # if the list has one entry per entity, pick [rank-1], otherwise store the whole thing
                            in_tokens = lst[rank-1] if isinstance(lst, list) and len(lst) >= rank else lst
                        if method in token_out_map:
                            lst = token_out_map[method]
                            out_tokens = lst[rank-1] if isinstance(lst, list) and len(lst) >= rank else lst
                        detail_rows.append({
                            "retriever": retriever,
                            "model":    model_name,
                            "context":  context,
                            "query_id": query_id,
                            "method":   method,
                            "entity":   entity,
                            "score":    score,
                            "rank":     rank,
                            "is_gold":  int(entity in gold),
                            "in_tokens":  in_tokens,
                            "out_tokens": out_tokens,
                        })

                rank_kwargs = {
                    "prob_rank": prob_rank,
                    "rag_rank": rag_rank,
                    "gold": gold,
                }


                if retriever.lower() == "bm25":
                    rank_kwargs["bm25_rank"] = baseline_rank
                elif retriever.lower() == "e5":
                    rank_kwargs["e5_rank"] = baseline_rank
                reporter.report("rankings", **rank_kwargs)

                # accumulate metrics
                for method, preds in (
                    ("prob", [d for d,_ in prob_rank]),
                    (retriever.lower(), [d for d,_ in baseline_rank]),
                    ("rag",  [d for d,_ in rag_rank]),
                ):
                    for k in Ks:
                        metrics[method][f"p@{k}"].append(precision_at_k(preds, gold, k))
                        metrics[method][f"r@{k}"].append(recall_at_k   (preds, gold, k))
                        metrics[method][f"f1@{k}"].append(f1_at_k       (preds, gold, k))
                        metrics[method][f"ndcg@{k}"].append(ndcg_at_k   (preds, gold, k))
                    metrics[method]["mrr"].append(reciprocal_rank(preds, gold))


            log(f"\n--- Aggregate metrics for {model_name} @ {context} ---")
            agg = {"model": model_name, "retriever": retriever, "context": context}
            for method in ("prob", retriever.lower(),"rag"):
                log(f"{method.upper()} SUMMARY:")
                for k in Ks:
                    mean_p = np.mean(metrics[method][f"p@{k}"])
                    mean_r = np.mean(metrics[method][f"r@{k}"])
                    mean_f = np.mean(metrics[method][f"f1@{k}"])
                    mean_n = np.mean(metrics[method][f"ndcg@{k}"])
                    log(f"  P@{k}={mean_p:.3f} | R@{k}={mean_r:.3f} | F1@{k}={mean_f:.3f} | NDCG@{k}={mean_n:.3f}")
                    agg[f"{method}_P@{k}"]  = mean_p
                    agg[f"{method}_R@{k}"]  = mean_r
                    agg[f"{method}_F1@{k}"] = mean_f
                    agg[f"{method}_nDCG@{k}"] = mean_n
                mean_mrr = np.mean(metrics[method]["mrr"])
                log(f"  MRR={mean_mrr:.3f}")
                agg[f"{method}_MRR"] = mean_mrr

            agg["prob_avg_in_tokens_per_entity" ] = float(np.mean(token_metrics["prob"]["in"] ))
            agg["prob_avg_out_tokens_per_entity"] = float(np.mean(token_metrics["prob"]["out"]))
            agg["rag_avg_in_tokens_per_entity"  ] = float(np.mean(token_metrics["rag"]["in"]  ))
            agg["rag_avg_out_tokens_per_entity" ] = float(np.mean(token_metrics["rag"]["out"] ))

            all_results.append(agg)

    df = pd.DataFrame(all_results).set_index(["model","context"])
    print(df.to_markdown())
    log("Final results:")
    log(df.to_markdown())

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(outdir / "detail.csv", index=False)
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    df.to_csv(outdir / "metrics.csv", index=False)
    reporter.save()


    end = datetime.now()
    print(f"TOTAL RUNTIME: {end - start}")




