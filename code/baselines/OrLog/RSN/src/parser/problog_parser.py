# src/parser/problog.py
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cache_utils import DiskKV, make_key
from .base import ParserABC

from helpers.llm_configs import llm_config_setup
from helpers.text_utils import _sanitize
import logging

log = logging.getLogger(__name__).info

class ProblogParser(ParserABC):
    def __init__(self, model_name: str, prompts_path: str, quantize: str = "8bit"):
        self.model_name = model_name
        bnb_config = None

        with open(prompts_path) as f:
            self.parse_template = json.load(f)["scheme_1"].rstrip()

        n_gpus = torch.cuda.device_count()
        self.bnb_config, self.designated_max_memory = llm_config_setup(quantize, n_gpus)
        
        self.tokenizer = None
        self.model     = None

        self._parse_cache = DiskKV(f"parse-{_sanitize(self.model_name)}.pkl")
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        self.model     = AutoModelForCausalLM.from_pretrained(
            self.model_name,
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
    def parse_query(self, query: str, max_new_tokens: int = 500, temperature: float = 0.0):
            # build cache-key: model + template + kwargs + exact text
            key = make_key(self.parse_template, max_new_tokens, temperature, query)
            cached = self._parse_cache.get(key)
            if cached is not None:
                log("[CACHE-HIT] parse_query")
                return cached
            if self.model is None:
                self.load_model()

            prompt = f"{self.parse_template}\n\n\n\"{query}\"\n\nOutput:\n```json\n"
            # log(f"Prompt:\n{prompt}")
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            m = re.search(r"```json\s*(\{.*?\})\s*```", out, re.DOTALL)
            if not m:
                raise ValueError(f"Failed to extract JSON parse:\n{out}")
            result = json.loads(m.group(1))


            self._parse_cache.put(key, result)

            return result
