import json
import os
import logging

log = logging.getLogger(__name__).info



def load_and_sample_data(DATA_PATH, N_instances=None):
    log(f"Loading data from {DATA_PATH}")
    full_data = [json.loads(l) for l in open(DATA_PATH)]
    log(f"Loaded {len(full_data)} examples")

    if N_instances is not None:
        full_data = full_data[:N_instances]
        log(f"Truncated to first {N_instances} instances")

    return full_data