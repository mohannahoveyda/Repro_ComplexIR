# check_models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_llm(model_name, logger):

    n_gpus = torch.cuda.device_count()
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
    else:
        raise RuntimeError(f"Expected 2 or 3 GPUs, but found {n_gpus}. Check SLURM allocation.")
    logger(f"Designated {n_gpus} GPUs with max_memory = {designated_max_memory}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    try:
        logger(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}")

    try:
        logger(f"Loading model {model_name} with 8-bit quant, device_map, offload...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=designated_max_memory,
            offload_folder="offload",
            offload_state_dict=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            logger(f"→ Layer on {param.device}: {name}")
            break
    else:
        logger("All parameters on GPU (or correctly offloaded).")

    model = torch.compile(model)
    model.eval()

    return tokenizer, model


def check_model(model_name):
    def log(msg):
        print(f"[{model_name}] {msg}")

    print(f"\n=== Checking model: {model_name} ===")
    try:
        tokenizer, model = load_llm(model_name, logger=log)
    except Exception as e:
        print(f"{e}")
        return

    try:
        true_ids  = tokenizer.encode(" True",  add_special_tokens=False)
        false_ids = tokenizer.encode(" False", add_special_tokens=False)
        log(f"Token IDs for ' True'  → {true_ids}")
        log(f"Token IDs for ' False' → {false_ids}")
        for idx in true_ids:
            print(idx, "→", tokenizer.decode([idx]))
        for idx in false_ids:
            print(idx, "→", tokenizer.decode([idx]))

        if len(true_ids) != 1 or len(false_ids) != 1:
            log("Warning: ' True' or ' False' is tokenized into multiple tokens.")
        else:
            log("' True' and ' False' each map to a single token.")
    except Exception as e:
        log(f" Tokenization check failed: {e}")
        return

    prompt = "Paris is the capital of France. Is this sentence true or false?"
    log(f"Running a short generation for prompt:\n  \"{prompt}\"")
    try:
        enc = tokenizer(prompt, return_tensors="pt")
        gen_output = model.generate(
            **enc,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
        continuation = full_text[len(prompt):]
        log("Generated continuation:")
        print(continuation.strip())
    except Exception as e:
        log(f" Generation failed: {e}")
        return

    log("Computing P('True') vs P('False') from final logits:")
    try:
        enc_tf = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**enc_tf)
            final_logits = outputs.logits[0, -1]  # shape: [vocab_size]

        id_t = true_ids[0]
        id_f = false_ids[0]
        pair_logits = torch.stack([final_logits[id_t], final_logits[id_f]]).to(final_logits.device)
        probs = torch.softmax(pair_logits, dim=0).cpu().tolist()
        log(f"P('True'  | prompt) = {probs[0]:.4f}")
        log(f"P('False' | prompt) = {probs[1]:.4f}")
    except Exception as e:
        log(f"Logits/TF probability check failed: {e}")
        return

    log(" All checks passed.")


if __name__ == "__main__":
    model_list = [
        # "allenai/OLMo-2-0325-32B",
        # "allenai/OLMo-2-1124-7B-Instruct",
        # The following two Mistral models are gated and may require access:
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.1",
        "allenai/OLMo-2-0325-32B-Instruct",
    ]

    for name in model_list:
        try:
            check_model(name)
        except Exception as e:
            print(f"[{name}] Unexpected failure: {e}")
