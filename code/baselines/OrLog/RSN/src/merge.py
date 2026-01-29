# from pathlib import Path
# import pandas as pd

# root = Path("/home/mhoveyda1/REASON/runs")
# dfs = []

# for model_dir in root.iterdir():
#     for csv in (model_dir / "results").rglob("metrics.csv"):
#         df = pd.read_csv(csv)
#         df["model"] = model_dir.name
#         df["chunk"] = csv.parent.name          # chunk_0, chunk_1, …
#         dfs.append(df)

# merged = pd.concat(dfs, ignore_index=True)
# merged.to_csv(root / "results_full.csv", index=False)
# print("Merged CSV written:", root / "results_full.csv")


import glob, pandas as pd

# Merge all the raw detail rows
dd = pd.concat([
    pd.read_csv(f) for f in glob.glob("runs/*/results/chunk_*/detail.csv")
], ignore_index=True)
dd.to_csv("full_detail.csv", index=False)
print("Full detail → full_detail.csv")

sd = pd.concat([
    pd.read_csv(f) for f in glob.glob("runs/*/results/chunk_*/summary.csv")
], ignore_index=True)
sd.to_csv("full_summary.csv", index=False)
print("Chunk summaries → full_summary.csv")


import pandas as pd
from evaluate import precision_at_k, recall_at_k, f1_at_k, reciprocal_rank

dd = pd.read_csv("full_detail.csv")
Ks = [1,3,5]
out = []

# group by model,context,method
for (model,context,method), grp in dd.groupby(["model","context","method"]):
    metrics = {"model":model, "context":context, "method":method}
    # for each query, reconstruct its ranking
    scores = []
    for qid, qgrp in grp.groupby("query_id"):
        qdf = qgrp.sort_values("score", ascending=False)
        preds = qdf["entity"].tolist()
        golds = qdf.loc[qdf.is_gold==1, "entity"].tolist()
        for k in Ks:
            metrics[f"P@{k}"] = metrics.get(f"P@{k}",[]) + [precision_at_k(preds, set(golds), k)]
            metrics[f"R@{k}"] = metrics.get(f"R@{k}",[]) + [recall_at_k   (preds, set(golds), k)]
            metrics[f"F1@{k}"] = metrics.get(f"F1@{k}",[]) + [f1_at_k       (preds, set(golds), k)]
        metrics["MRR"] = metrics.get("MRR",[]) + [reciprocal_rank(preds, set(golds))]

    # average across queries
    for metric in list(metrics):
        if isinstance(metrics[metric], list):
            metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
    out.append(metrics)

pd.DataFrame(out).to_csv("final_metrics.csv", index=False)
print("Recomputed overall metrics → final_metrics.csv")
