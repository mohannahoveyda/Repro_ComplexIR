# src/reasoner/rag_reasoner.py


import os
import json
import logging
from typing import List, Tuple, Dict, Any
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from cache_utils import DiskKV, make_key
from .base import ReasonerABC
from helpers.text_utils import _sanitize, _sanitize_entity, _sanitize_predicate, extract_answer_with_regex

from helpers.llm_configs import llm_config_setup


log = logging.getLogger(__name__).info

class RAGReasoner(ReasonerABC):
    """
        Retrieval-Augmented Generation (RAG) reasoner.
        Given an entry dict with 'query' and 'pred_docs_metadata',
        builds per-entity prompts, calls the LLM to get True/False,
        caches outputs, and returns a ranked list.
    """

    def __init__(
        self, 
        model_name: str,
        prompts_path: str,
        quantize: str = "8bit",
        context_mode: str = "atom",
        experiment_mode: str = "IO",
        reporter = None,
        log_dir = None,
    ):

        self.model_name = model_name
        self.prompts_path = prompts_path
        self.quantize = quantize
        self.context_mode = context_mode
        self.experiment_mode = experiment_mode
        self.reporter       = reporter
        self.log_dir        = log_dir or os.getcwd()

        self.in_tokens_list  = []
        self.out_tokens_list = []
        # Based on the context mode, choose prompts["scheme_1"] or prompts["scheme_2"] or prompts["scheme_3"] or prompts["scheme_4"]
        with open(prompts_path) as f:
            prompts = json.load(f)
            if context_mode == "atom":
                self.parse_template = prompts["scheme_1"].rstrip()
            elif context_mode == "wiki":
                self.parse_template = prompts["scheme_2"].rstrip()
            elif context_mode == "wd":
                self.parse_template = prompts["scheme_3"].rstrip()
            elif context_mode == "full":
                # self.parse_template = prompts["scheme_4"].rstrip()
                # Say not implemented
                raise NotImplementedError("Full context mode is not implemented yet.")
            else:
                raise ValueError(f"Unknown context_mode {context_mode}")
        # Model --------------------------------------------------------------
        bnb_config = None
        n_gpus = torch.cuda.device_count()
        self.bnb_config, self.designated_max_memory = llm_config_setup(quantize, n_gpus)

        self.model = None
        self.tokenizer = None

        cache_id = _sanitize(f"{model_name}-{quantize}-{context_mode}-{experiment_mode}")
        self._rag_cache = DiskKV(f"rag_answer_{cache_id}.pkl")
    def reset_costs(self):
        self.in_tokens_list.clear()
        self.out_tokens_list.clear()
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

    def _build_prompt(
        self,
        query: str,
        entity_meta: dict,
    ):
        # Build: The prompt scheme + the context + the query + the entity name
        # pick the right context for this entity
        if self.context_mode == "atom":
            ctx = ""
        elif self.context_mode == "wiki":
            ctx = entity_meta.get("wikipedia_text","")
        elif self.context_mode == "wd":
            props = entity_meta.get("wikidata_properties",{})
            ctx = " ".join(f"{k}: {','.join(v)}." for k,v in props.items()) 
        elif self.context_mode == "full":
            wiki  = entity_meta.get("wikipedia_text","")
            props = entity_meta.get("wikidata_properties",{})
            props_txt = " ".join(f"{k}: {','.join(v)}." for k,v in props.items())
            ctx = "\n\n".join(filter(None,[wiki, props_txt]))
        else:
            ctx = ""
        prompt = ""
        if self.experiment_mode == "COT":
            prompt += "\n\nLet's think step by step."
            
        # First add the prompt scheme then the context then the entity name then the query end with <EVALUATE> and indicate query with <QUERY> and entity with <ENTITY> close all of them with </
        # EVALUATE> and </QUERY> and </ENTITY>
        prompt += f"{self.parse_template}\n\n{ctx}\n\n<ENTITY>{entity_meta['doc']}</ENTITY>\n\n<QUERY>{query}</QUERY>\n\n<EVALUATE>"        

        return prompt

    def score_entity(
        self,
        input_prompt: str,
        max_new_tokens: int = 500, 
        temperature: float = 0.0,
        entity_meta: dict = None,
    ):
        """
        feed the input prompt to the model and get the the generated text. using REGEX Extract the answer which will be in form of
            ```
            {
            'Answer': 'False'
            }
            ```
        Capture the answer which will be in form of 'True' or 'False' for the entity.
        """

        key = make_key(self.model_name,
                       self.quantize,
                       self.context_mode,
                       self.experiment_mode,
                       input_prompt)

        # 2) check cache
        cached = self._rag_cache.get(key)
        if cached is not None:

            log("[CACHE-HIT] RAG score_entity")
            output = cached

            self.reporter.report("rag_output", output=output, entity=entity_meta["doc"])

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

            in_enc = self.tokenizer(input_prompt, return_tensors="pt")
            self.in_tokens_list.append(in_enc.input_ids.shape[-1])

            out_enc = self.tokenizer(cached, return_tensors="pt")
            self.out_tokens_list.append(out_enc.input_ids.shape[-1])

            result = extract_answer_with_regex(output)

            score  = 1.0 if result.lower() == "true" else 0.0
            return score

        if self.model is None:
            self.load_model()

        enc = self.tokenizer(input_prompt, return_tensors="pt").to(self.model.device)
        input_ids = enc.input_ids

        self.in_tokens_list.append(enc.input_ids.shape[-1])

        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # out = self.tokenizer.decode(gen[0], skip_special_tokens=True)

        new_tokens = gen[0][ input_ids.shape[-1] : ]
        out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.out_tokens_list.append(new_tokens.shape[-1])


        # log(f"Output of RAG:\n{out}")

        self.reporter.report("rag_output", output=out, entity=entity_meta["doc"])
        result = extract_answer_with_regex(out)
        # log(f"Extracted answer: {result}")
        score  = 1.0 if result.lower() == "true" else 0.0

        self._rag_cache.put(key, out)

        return score


    def rank_entities(
        self,
        entry
    ):
        """
        For each entity in the entry, build the prompt and score the entity. Then sort the entities based on the score.
        """
        self.reset_costs()

        scored = []
        for d in entry["pred_docs_metadata"]:
            prompt = self._build_prompt(entry["query"], d)
            score = self.score_entity(input_prompt=prompt, entity_meta=d)
            scored.append((d["doc"], score))
        ranking = sorted(scored, key=lambda x: x[1], reverse=True)


        return ranking, self.in_tokens_list, self.out_tokens_list

