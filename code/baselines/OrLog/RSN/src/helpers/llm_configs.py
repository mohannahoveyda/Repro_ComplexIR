import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def llm_config_setup(quantize, n_gpus):
    if quantize == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_enable_fp32_cpu_offload=True,
        )
    elif quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            # llm_int8_enable_fp32_cpu_offload=True,
        )
    if n_gpus == 3:
            designated_max_memory = {
                0: "85GiB",
                1: "85GiB",
                2: "80GiB",
            }
    elif n_gpus == 2:
        designated_max_memory = {
            0: "85GiB",
            1: "85GiB",
        }
    elif n_gpus == 1:
        designated_max_memory = {
            0: "40GiB",
        }
    else:
        raise RuntimeError(f"Expected 1, 2 or 3 GPUs, but found {n_gpus}. Check SLURM allocation.")

    print(f"bnb_config\n", bnb_config)
    return bnb_config, designated_max_memory