import os
import json
import random

random.seed(42)

def log(msg):
    # Simple logger—just prints to stdout
    print(msg)


def load_and_sample_data(DATA_PATH, N_instances):
    # 4) load & balance‐sample data
    log(f"Loading data from {DATA_PATH}")
    full_data = [json.loads(l) for l in open(DATA_PATH)]
    log(f"Loaded {len(full_data)} examples")
    templates = sorted({e["metadata"]["template"] for e in full_data})
    N_per_tpl = N_instances // len(templates)

    orig_dir  = os.path.dirname(DATA_PATH)
    orig_name = os.path.basename(DATA_PATH)           # e.g. "foo.jsonl"
    base, ext = os.path.splitext(orig_name)           # base="foo", ext=".jsonl"
    new_name  = f"{base}_sampled_equi_{N_instances}.jsonl"
    new_path  = os.path.join(orig_dir, new_name)
    # Check if a file called sampled_data_{N_instances}.jsonl already exists and if so, load it
    if os.path.exists(new_path):
        log(f"File sampled_data_{N_instances}.jsonl already exists. Loading it...")
        with open(new_path) as f:
            sampled = [json.loads(line) for line in f]
        log(f"Loaded {len(sampled)} examples from {f.name}")
        # Check if the number of examples is equal to N_instances
        if len(sampled) != N_instances:
            log(f"Warning: Number of examples in {f.name} is not equal to N_instances ({N_instances}).")
    else:
        log("Sampling data...")

        sampled   = []
        for t in templates:
            sampled += random.sample(
                [e for e in full_data if e["metadata"]["template"] == t],
                min(N_per_tpl, sum(1 for e in full_data if e["metadata"]["template"] == t))
            )
        sampled = sampled[:N_instances]
        log(f"Sampled {len(sampled)} examples across {len(templates)} templates")

        # Save the sampled data to a file called sampled_data_{N_instances}.jsonl
        
        with open(new_path, "w") as f:
            for entry in sampled:
                f.write(json.dumps(entry) + "\n")
        log(f"Wrote sampled data to {new_path}")
        return sampled

if __name__ == "__main__":
    path = "/home/mhoveyda1/RSN_Z/test_top20_sample0_2025-05-22_12-23_filtered_with_wikidata_and_wikipedia_metadata_filtered_based_on_pred_maps.jsonl"

    sampled_data = load_and_sample_data(path, 110)

