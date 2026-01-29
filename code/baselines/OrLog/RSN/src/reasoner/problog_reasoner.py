# /home/mhoveyda1/REASON/src/reasoner/problog_reasoner.py
import os
from problog.program import PrologString
from problog import get_evaluatable
from .base import ReasonerABC
from estimator.problog_estimator import ProblogEstimator
from parser.problog_parser import ProblogParser
from .prolog_parser import BoolParser

from helpers.text_utils import _sanitize_entity, _sanitize_predicate, _sanitize
import logging

log = logging.getLogger(__name__).info

class ProbLogReasoner(ReasonerABC):
    """
    ProbLog-based reasoner: uses a Parser to extract atoms and logical form,
    then an Estimator to get P(True) for each atom, and builds+evaluates a ProbLog program.
    """
    def __init__(
        self,
        parser: ProblogParser,
        estimator: ProblogEstimator,
        log_dir: str,
        reporter
    ):
        self.reporter = reporter
        self.parser    = parser
        self.estimator = estimator
        self.eval_mod  = get_evaluatable()
        self.log_dir   = log_dir
        self.all_entities_in_costs = []
        self.all_entities_out_costs = []

    # --------------------------- Program build ---------------------------
    def _build_program_for_entity(self, atoms, expr, entity_meta):
            # # 1) sanitize
            # log(f"[BUILD-Program] for entity [{entity_meta['doc']}], atoms [{atoms}], expr [{expr}]")
            raw_ent      = _sanitize_entity(entity_meta["doc"])
            entity_const = f"'{raw_ent}'"

            # entity_const = _sanitize_entity(entity_meta["doc"])
            mapping      = {a: _sanitize_predicate(a) for a in atoms}
            # log(f"[DEBUG] predicate mapping: {mapping}")

            probs        = self.estimator.get_probabilities(atoms, entity_meta)

            # log(f"Probabilities for {entity_meta['doc']}is: {probs}")

            # 2) build probabilistic facts
            lines = [f"{probs[a]:.6f}::{mapping[a]}({entity_const})." for a in atoms]

            # log(f"[BUILD-Program] for entity [{entity_meta['doc']}], Lines: {lines}")

            # 3) parse & convert to Prolog body
            for atom in atoms:
                if " not " in atom:
                    # e.g. atom = "{x} is not in English"
                    positive = atom.replace(" not", "")  
                    # positive == "{x} is in English"
                    expr = expr.replace(f"NOT({positive})", atom)
            # log(f"[DEBUG] cleaned expr: {expr}")
            parser = BoolParser(atoms)
            # log(f"[BUILD-Program] for entity [{entity_meta["doc"]}]")
            ast    = parser.parse(expr)
            body   = ast.to_prolog(mapping, entity_const)
            # log(f"[DEBUG] generated Prolog body = {body}")

            # 4) finalize
            lines += [f"result :- {body}.", "query(result)."]
            
            program = PrologString("\n".join(lines))
            program_src = "\n".join(lines)
            # log(f"[DEBUG] final program_src =\n{program_src}")


    
            cm = self.estimator.context_mode
            if cm != "atom":
                # build exactly the same context you pass to the LLM
                if cm == "wiki":
                    ctx = entity_meta.get("wikipedia_text", "")
                elif cm == "wd":
                    props = entity_meta.get("wikidata_properties", {})
                    ctx = " ".join(f"{k}: {','.join(v)}." for k, v in props.items()) if props else ""
                elif cm == "full":
                    wiki     = entity_meta.get("wikipedia_text", "")
                    props    = entity_meta.get("wikidata_properties", {})
                    props_txt = " ".join(f"{k}: {','.join(v)}." for k, v in props.items()) if props else ""
                    # join with a blank line so it’s readable
                    ctx = "\n\n".join(filter(None, [wiki, props_txt]))
                else:
                    ctx = ""
                

                # add the atoms to the end of the context
                ctx += "\n\nAtoms:\n" + "\n".join(atoms)


                self.reporter.report(
                    "entity_context",
                    entity=entity_meta["doc"],
                    context=ctx
                )

      
            self.reporter.report(
                "prolog_program",
                entity=entity_meta["doc"],
                program=program_src,
            )
            return program, program_src
    # ------------------------------ Scoring ------------------------------
    def score_entity(self, atoms, expr, entity_meta):
        program, program_src = self._build_program_for_entity(atoms, expr, entity_meta)
        # log(f"Program for entity [{entity_meta['doc']}] is: \n{program_src}")
        try:
            result  = self.eval_mod.create_from(program).evaluate()
            # log(f"Result for entity [{entity_meta['doc']}] is: {result}")
        except Exception as e:
            log("[ERROR] Problog evaluation failed for entity %r", entity_meta["doc"])
            # 2) Dump the exact Prolog program that caused the failure
            fail_path = os.path.join(self.log_dir, _sanitize(entity_meta["doc"]) + "_failed.prolog")
            with open(fail_path, "w") as f:
                f.write(program_src)
            log("→ Wrote failing Prolog program to %s", fail_path)
            raise 

        return next(iter(result.values()))

    def rank_entities(self, entry):
    # def rank_entities(self, query: str, candidates: List[dict]) -> List[Tuple[str,float]]:
        query      = entry["query"]
        candidates = entry["pred_docs_metadata"]
        parsed     = self.parser.parse_query(query)
        atoms, expr = parsed["atoms"], parsed["logical query"]

        all_scores = []
        self.all_entities_in_costs.clear()
        self.all_entities_out_costs.clear()
        for m in candidates: 
            self.estimator.reset_costs()
            p = self.score_entity(atoms, expr, m)
            costs = self.estimator.get_costs()
            self.all_entities_in_costs.append(costs['in_tokens'])
            self.all_entities_out_costs.append(costs['out_tokens'])
            print(f"[DEBUG] entity={m['doc']} → in={costs['in_tokens']} out={costs['out_tokens']}")
            print(f"[DEBUG] entity={m['doc']} → in={self.all_entities_in_costs} out={self.all_entities_out_costs}")

            all_scores.append((m["doc"], p))



        ranking = sorted(all_scores, key=lambda x: (x[1], x[0]), reverse=True)
        return ranking, self.all_entities_in_costs, self.all_entities_out_costs
    
            