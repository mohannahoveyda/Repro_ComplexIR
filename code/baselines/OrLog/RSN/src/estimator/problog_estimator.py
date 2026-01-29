# /home/mhoveyda1/REASON/src/estimator/problog_estimator.py
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cache_utils import DiskKV, make_key
from .base import ProbabilityEstimatorABC

from helpers.llm_configs import llm_config_setup
from helpers.text_utils import _sanitize
import logging
from typing import Dict, List


log = logging.getLogger(__name__).info

class ProblogEstimator(ProbabilityEstimatorABC):
    def __init__(self, model_name: str, prompts_path: str, quantize: str = "8bit", context_mode: str = "atom", method: str = "tf"):
        with open(prompts_path) as f:
            self.parse_template = json.load(f)["scheme_1"].rstrip()
        self.model_name_in_use = model_name
        bnb_config = None
        n_gpus = torch.cuda.device_count()
        self.bnb_config, self.designated_max_memory = llm_config_setup(quantize, n_gpus)

        self.tokenizer = None
        self.model     = None
        


        self.context_mode = context_mode
        self.method       = method
        self._prob_cache  = DiskKV(f"prob-{_sanitize(self.model_name_in_use)}-{method}-{context_mode}.pkl")
        self._logit_cache = DiskKV(f"logits-{_sanitize(self.model_name_in_use)}-{method}-{context_mode}.pkl")

        # for cost accounting
        self.in_tokens_list: List[int] = []
        self.out_tokens_list: List[int] = []
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_in_use, use_fast=False)
        self.model     = AutoModelForCausalLM.from_pretrained(
            self.model_name_in_use,
            quantization_config=self.bnb_config,
            device_map="auto",
            max_memory=self.designated_max_memory,
            offload_folder="offload",
            offload_state_dict=True,
        )
        for name, param in self.model.named_parameters():
            if param.device.type != "cuda":
                log(f"Layer on {param.device}: {name}")
                break
        else:
            log("All parameters on GPU.")
        self.model = torch.compile(self.model)
        self.model.eval()
    def reset_costs(self):
        """Call once per entity to zero‐out the cost accumulators."""
        self.in_tokens_list.clear()
        self.out_tokens_list.clear()

    def _build_prompt(self, atom: str, entity_meta: dict) -> str:
        stmt  = atom.replace("{x}", entity_meta["doc"])
        # Add to the end of the atom 
        stmt += ". Is this phrase true or false?"
        wiki  = entity_meta.get("wikipedia_text", "")
        props = entity_meta.get("wikidata_properties", {})
        if not props:
            props_txt = " "
        else:
            props_txt = " ".join(f"{k}: {','.join(v)}." for k, v in props.items())
        if self.context_mode == "atom":
            return stmt
        if self.context_mode == "wiki":
            return f"{wiki}\n{stmt}"
        if self.context_mode == "wd":
            return f"{props_txt}\n{stmt}"
        if self.context_mode == "full":
            return f"{wiki}\n{props_txt}\n{stmt}"
        raise ValueError(f"Unknown context_mode {self.context_mode}")

    def _prob_true_false(self, prompt: str) -> float:
        key = make_key(prompt)
        logits_list = self._logit_cache.get(key)

        if logits_list is None:
            
            log("[CACHE-MISS] get_logits")

            if self.model is None:
                self.load_model()

            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            self.in_tokens_list.append(enc.input_ids.shape[-1])

            with torch.no_grad():
                logits_tensor = self.model(**enc).logits[0, -1]
            logits_list = logits_tensor.cpu().tolist()
            self._logit_cache.put(key, logits_list)
        else:
            log(f"[CACHE-HIT] get_logits")
        
            enc = self.tokenizer(prompt, return_tensors="pt")
            self.in_tokens_list.append(enc.input_ids.shape[-1])

        id_t = self.tokenizer.encode(" True",  add_special_tokens=False)[0]
        id_f = self.tokenizer.encode(" False", add_special_tokens=False)[0]
        t_logit = logits_list[id_t]
        f_logit = logits_list[id_f]

        probs = torch.softmax(torch.tensor([t_logit, f_logit]), dim=0)
        self.out_tokens_list.append(1)
        return probs[0].item()

    def _prob_likelihood(self, prompt: str) -> float:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.in_tokens_list.append(tokens.input_ids.shape[-1])
        with torch.no_grad():
            loss = self.model(**tokens, labels=tokens.input_ids).loss
        self.out_tokens_list.append(1)
        return torch.exp(-loss).item()


    def get_probability(self, atom: str, entity_meta: dict) -> float:
        prompt = self._build_prompt(atom, entity_meta)
        log(f"[Prob-Req] Requesting Prob for atom %r", atom)
        key = make_key(prompt)

        try:
            cached = self._prob_cache.get(key)
        except Exception as e:
            # Instead of re-raising, treat as a cache miss:
            log(f"[DEBUG] could not get cached prob (error: {e}). Will compute from LLM.")
            cached = None

        if cached is not None:
            log("[CACHE-HIT] get_probability")
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_in_use, use_fast=False)
            enc = self.tokenizer(prompt, return_tensors="pt")
            self.in_tokens_list.append(enc.input_ids.shape[-1])
            self.out_tokens_list.append(1)
            return cached

        # At this point, either get() returned None or an exception occurred.
        log("[CACHE-MISS] get_probability for atom %r → computing LLM call", atom)

        if self.method == "tf":
            prob = self._prob_true_false(prompt)
            log(f"[CACHE-MISS-LLM] Got the prob directly from LLM: {prob}")
        elif self.method == "likelihood":
            prob = self._prob_likelihood(prompt)
        else:
            raise ValueError(f"Unknown method {self.method}")

        # Save the computed probability back into cache (so future calls can reuse it)
        try:
            self._prob_cache.put(key, prob)
        except Exception as e:
            # Optionally, log if putting into cache fails, but do NOT let this crash the function
            log(f"[DEBUG] could not write prob to cache (error: {e}). Ignoring.")
        return prob
    def get_costs(self) -> Dict[str, List[int]]:

        return {
            "in_tokens":  sum(self.in_tokens_list),
            "out_tokens": sum(self.out_tokens_list),
        }
        
    def get_probabilities(self, atoms: list, entity_meta: dict):
        return {a: self.get_probability(a, entity_meta) for a in atoms}

