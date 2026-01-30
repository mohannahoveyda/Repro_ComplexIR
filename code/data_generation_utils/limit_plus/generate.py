import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

import argparse
import time

DEFAULT_DATA_DIR = "./limit_data/"


AttrSet = Set[str]
PredicateFn = Callable[[AttrSet], bool]


@dataclass
class TemplateInstance:
    name: str              # template in QUEST
    attrs: Tuple[str, ...] # the attributes used (A, B, C, ...)
    query: str             # natural language query ("Who likes A or B?")
    original_query: str    # query with <mark>...</mark>
    predicate: PredicateFn 



def load_limit_small_corpus(corpus_path: str):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found at: {corpus_path}")

    entity2attrs: Dict[str, AttrSet] = {}
    attr2entities: Dict[str, Set[str]] = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            eid = obj.get("_id") or obj.get("id")
            text = obj["text"]
            attrs = parse_likes_attributes(text)
            if not attrs:
                continue
            entity2attrs[eid] = attrs
            for a in attrs:
                attr2entities.setdefault(a, set()).add(eid)

    return entity2attrs, attr2entities

def load_limit_single_template_queries(
    data_dir: str,
    max_per_template: int,
) -> List[dict]:
    queries_path = os.path.join(data_dir, "queries.jsonl")
    qrels_path = os.path.join(data_dir, "qrels.jsonl")

    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        print("No queries.jsonl/qrels.jsonl found; skipping dataset-provided '_' queries.")
        return []

    # load qrels: query-id -> list of corpus-id
    qid2docs: Dict[str, List[str]] = {}
    with open(qrels_path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["query-id"]
            docid = obj["corpus-id"]
            score = obj.get("score", 1)
            if score > 0:
                qid2docs.setdefault(qid, []).append(docid)

    samples: List[dict] = []
    next_id = 0

    with open(queries_path, "r", encoding="utf-8") as fq:
        for line in fq:
            if len(samples) >= max_per_template:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("_id") or obj.get("id")
            text = obj["text"]

            docs = qid2docs.get(qid)
            if not docs:
                continue

            attr = text
            prefix = "Who likes "
            if text.startswith(prefix) and text.endswith("?"):
                attr = text[len(prefix):-1].strip()


            sample = {
                "id": next_id,
                "query": text,
                "num_docs": len(docs),
                "docs": docs,
                "original_query": f"<mark>{attr}</mark>",
                "metadata": {
                    "template": "_",
                    "attrs": [attr],
                    "source": "limit-original",
                },
            }
            samples.append(sample)
            next_id += 1

    print(f"Loaded {len(samples)} '_' queries from LIMIT queries.jsonl")
    return samples


def parse_likes_attributes(text: str) -> AttrSet:
    if "likes" not in text:
        return set()

    _, after = text.split("likes", 1)
    after = after.strip()

    raw_items = after.split(",")
    attrs: Set[str] = set()

    n = len(raw_items)
    for idx, item in enumerate(raw_items):
        token = item.strip()
        token = token.rstrip(".")

        if n > 1 and idx == n - 1 and " and " in token:
            parts = [p.strip() for p in token.split(" and ") if p.strip()]
            for p in parts:
                if p.lower().startswith("and "):
                    p = p[4:].strip()
                if p:
                    attrs.add(p)
            continue

        if token.lower().startswith("and "):
            token = token[4:].strip()
        if token:
            attrs.add(token)

    return attrs



def make_template_single(all_attrs: List[str], **_) -> TemplateInstance:
    # _
    A = random.choice(all_attrs)
    name = "_"
    query = f"Who likes {A}?"
    original_query = f"<mark>{A}</mark>"

    def predicate(s: AttrSet) -> bool:
        return A in s

    return TemplateInstance(name, (A,), query, original_query, predicate)


def make_template_or_2(
    all_attrs: List[str],
    attr2entities: Dict[str, Set[str]],
    **_,
) -> TemplateInstance:
    # _ or _
    A = random.choice(all_attrs)
    ents_A = attr2entities[A]

    candidates = [
        b for b in all_attrs
        if b != A and attr2entities[b] != ents_A
    ] or [b for b in all_attrs if b != A]

    B = random.choice(candidates)

    name = "_ or _"
    query = f"Who likes {A} or {B}?"
    original_query = f"<mark>{A}</mark> or <mark>{B}</mark>"

    def predicate(s: AttrSet) -> bool:
        return (A in s) or (B in s)

    return TemplateInstance(name, (A, B), query, original_query, predicate)


def make_template_or_3(
    all_attrs: List[str],
    attr2entities: Dict[str, Set[str]],
    **_,
) -> TemplateInstance:
    # _ or _ or _
    A = random.choice(all_attrs)

    candidates_B = [
        b for b in all_attrs
        if b != A and attr2entities[b] != attr2entities[A]
    ] or [b for b in all_attrs if b != A]

    B = random.choice(candidates_B)

    candidates_C = [
        c for c in all_attrs
        if c not in {A, B}
        and not (
            attr2entities[c] == attr2entities[A]
            or attr2entities[c] == attr2entities[B]
        )
    ] or [c for c in all_attrs if c not in {A, B}]

    C = random.choice(candidates_C)

    name = "_ or _ or _"
    query = f"Who likes {A} or {B} or {C}?"
    original_query = f"<mark>{A}</mark> or <mark>{B}</mark> or <mark>{C}</mark>"

    def predicate(s: AttrSet) -> bool:
        return (A in s) or (B in s) or (C in s)

    return TemplateInstance(name, (A, B, C), query, original_query, predicate)

def make_template_that_also(
    entity2attrs: Dict[str, AttrSet],
    **_,
) -> TemplateInstance:
    # _ that are also _
    eid = random.choice(list(entity2attrs.keys()))
    attrs = list(entity2attrs[eid])
    if len(attrs) < 2:
        return make_template_that_also(entity2attrs=entity2attrs)

    A, B = random.sample(attrs, 2)

    name = "_ that are also _"
    query = f"Who likes {A} and also likes {B}?"
    original_query = f"<mark>{A}</mark> that are also <mark>{B}</mark>"

    def predicate(s: AttrSet) -> bool:
        return (A in s) and (B in s)

    return TemplateInstance(name, (A, B), query, original_query, predicate)


def make_template_that_also_both(
    entity2attrs: Dict[str, AttrSet],
    **_,
) -> TemplateInstance:
    # _ that are also both _ and _
    eid = random.choice(list(entity2attrs.keys()))
    attrs = list(entity2attrs[eid])
    if len(attrs) < 3:
        return make_template_that_also_both(entity2attrs=entity2attrs)

    A, B, C = random.sample(attrs, 3)

    name = "_ that are also both _ and _"
    query = f"Who likes {A} and also likes both {B} and {C}?"
    original_query = (
        f"<mark>{A}</mark> that are also both <mark>{B}</mark> and <mark>{C}</mark>"
    )

    def predicate(s: AttrSet) -> bool:
        return (A in s) and (B in s) and (C in s)

    return TemplateInstance(name, (A, B, C), query, original_query, predicate)


def make_template_that_also_but_not(
    entity2attrs: Dict[str, AttrSet],
    attr2entities: Dict[str, Set[str]],
    **_,
) -> TemplateInstance:
    # _ that are also _ but not _
    attrs = list(attr2entities.keys())
    max_tries = 1000

    for _ in range(max_tries):
        A, B = random.sample(attrs, 2)
        S_A = attr2entities[A]
        S_B = attr2entities[B]
        S_AB = S_A & S_B
        if len(S_AB) < 2:
            continue 

        candidate_Cs = []
        for C, S_C in attr2entities.items():
            if C in (A, B):
                continue
            have_C = S_AB & S_C
            if not have_C:
                continue
            no_C = [e for e in S_AB if C not in entity2attrs[e]]
            if not no_C:
                continue
            candidate_Cs.append(C)

        if not candidate_Cs:
            continue

        C = random.choice(candidate_Cs)

        name = "_ that are also _ but not _"
        query = f"Who likes {A} and also likes {B} but not {C}?"
        original_query = (
            f"<mark>{A}</mark> that are also <mark>{B}</mark> but not <mark>{C}</mark>"
        )

        def predicate(s: AttrSet) -> bool:
            return (A in s) and (B in s) and (C not in s)

        return TemplateInstance(name, (A, B, C), query, original_query, predicate)


    A, B, C = random.sample(attrs, 3)
    name = "_ that are also _ but not _"
    query = f"Who likes {A} and also likes {B} but not {C}?"
    original_query = (
        f"<mark>{A}</mark> that are also <mark>{B}</mark> but not <mark>{C}</mark>"
    )

    def predicate(s: AttrSet) -> bool:
        return (A in s) and (B in s) and (C not in s)

    return TemplateInstance(name, (A, B, C), query, original_query, predicate)

def make_template_that_not(
    entity2attrs: Dict[str, AttrSet],
    attr2entities: Dict[str, Set[str]],
    **_,
) -> TemplateInstance:
    # _ that are not _
    candidate_As = [a for a, ents in attr2entities.items() if len(ents) >= 2]
    if not candidate_As:
        A = random.choice(list(attr2entities.keys()))
        B = random.choice([b for b in attr2entities.keys() if b != A])

        name = "_ that are not _"
        query = f"Who likes {A} but not {B}?"
        original_query = f"<mark>{A}</mark> that are not <mark>{B}</mark>"

        def predicate(s: AttrSet) -> bool:
            return (A in s) and (B not in s)

        return TemplateInstance(name, (A, B), query, original_query, predicate)

    max_tries = 1000
    for _ in range(max_tries):
        A = random.choice(candidate_As)
        base_ents = attr2entities[A]


        candidate_Bs = []
        for B, entsB in attr2entities.items():
            if B == A:
                continue
            have_B = base_ents & entsB
            if not have_B:
                continue
            no_B = [e for e in base_ents if B not in entity2attrs[e]]
            if not no_B:
                continue
            candidate_Bs.append(B)

        if not candidate_Bs:
            continue

        B = random.choice(candidate_Bs)

        name = "_ that are not _"
        query = f"Who likes {A} but not {B}?"
        original_query = f"<mark>{A}</mark> that are not <mark>{B}</mark>"

        def predicate(s: AttrSet) -> bool:
            return (A in s) and (B not in s)

        return TemplateInstance(name, (A, B), query, original_query, predicate)

    A = random.choice(list(attr2entities.keys()))
    B = random.choice([b for b in attr2entities.keys() if b != A])
    name = "_ that are not _"
    query = f"Who likes {A} but not {B}?"
    original_query = f"<mark>{A}</mark> that are not <mark>{B}</mark>"

    def predicate(s: AttrSet) -> bool:
        return (A in s) and (B not in s)

    return TemplateInstance(name, (A, B), query, original_query, predicate)


# these are based on QUEST templates, you can check the paper for details of these
TEMPLATES = [
    ("_", make_template_single),
    ("_ or _", make_template_or_2),
    ("_ or _ or _", make_template_or_3),
    ("_ that are also _", make_template_that_also),
    ("_ that are also both _ and _", make_template_that_also_both),
    ("_ that are also _ but not _", make_template_that_also_but_not),
    ("_ that are not _", make_template_that_not),

    # new complex templates
    # ("_ or _ but not both", make_template_xor_2),
    # ("at least two of _ _ _", make_template_at_least_two_of_three),
    # ("( _ and _ ) or ( _ and _ ) but not _", make_template_dnf_or_and_not),
    # ("_ and _ but none of _ _ _", make_template_and_not_any_of_three),
]



def build_quest_like_samples(
    entity2attrs: Dict[str, AttrSet],
    attr2entities: Dict[str, Set[str]],
    max_per_template: int = 100,
    seed: int = 42,
    skip_single: bool = False,
    start_id: int = 0,
) -> List[dict]:
    random.seed(seed)

    all_attrs = sorted(attr2entities.keys())
    samples: List[dict] = []
    sample_id = start_id

    seen_keys_per_template: Dict[str, Set[Tuple[str, ...]]] = {
        tmpl_name: set() for tmpl_name, _ in TEMPLATES
    }

    max_failures_per_template = 5_000

    for tmpl_name, builder in TEMPLATES:
        if skip_single and tmpl_name == "_":
            continue

        failures = 0

        while len(seen_keys_per_template[tmpl_name]) < max_per_template and failures < max_failures_per_template:
            inst = builder(
                all_attrs=all_attrs,
                entity2attrs=entity2attrs,
                attr2entities=attr2entities,
            )

            key = tuple(sorted(inst.attrs))
            if key in seen_keys_per_template[tmpl_name]:
                failures += 1
                continue

            docs = [
                eid for eid, attrs in entity2attrs.items()
                if inst.predicate(attrs)
            ]
            if not docs:
                failures += 1
                continue

            seen_keys_per_template[tmpl_name].add(key)

            sample = {
                "id": sample_id,
                "query": inst.query,
                "num_docs": len(docs),
                "docs": docs,
                "original_query": inst.original_query,
                "metadata": {
                    "template": inst.name,
                    "attrs": list(inst.attrs),
                },
            }
            samples.append(sample)
            sample_id += 1
            failures = 0  

        print(
            f"Template '{tmpl_name}': "
            f"{len(seen_keys_per_template[tmpl_name])} unique queries generated "
            f"(stopped after {failures} failed attempts)."
        )

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Path to LIMIT data directory (e.g. .../limit-small or .../limit)",
    )
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=100,
        help="Max distinct queries per template",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="After saving limit_quest_queries.jsonl, run statistics and save table/figure",
    )

    args = parser.parse_args()
    start_time = time.time()

    data_dir = args.data_dir
    corpus_path = os.path.join(data_dir, "corpus.jsonl")

    print(f"Loading corpus from: {corpus_path}")
    entity2attrs, attr2entities = load_limit_small_corpus(corpus_path)
    print(f"Parsed {len(entity2attrs)} entities and {len(attr2entities)} attributes.")

    attrs_out_path = os.path.join(data_dir, "parsed_attributes.tsv")
    with open(attrs_out_path, "w", encoding="utf-8") as fa:
        for attr, ents in sorted(attr2entities.items(), key=lambda x: x[0]):
            fa.write(f"{attr}\t{len(ents)}\n")
    print(f"Saved parsed attributes to: {attrs_out_path}")

    max_per_template = args.max_per_template

    single_samples = load_limit_single_template_queries(
        data_dir=data_dir,
        max_per_template=max_per_template,
    )

    print(f"\nGenerating up to {max_per_template} unique queries per template for non-vanilla templates...")
    gen_samples = build_quest_like_samples(
        entity2attrs=entity2attrs,
        attr2entities=attr2entities,
        max_per_template=max_per_template,
        seed=0,
        skip_single=True,
        start_id=len(single_samples),
    )

    samples = single_samples + gen_samples
    print(f"\nTotal queries generated: {len(samples)}")

    out_path = os.path.join(data_dir, "limit_quest_queries.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    elapsed = time.time() - start_time
    print(f"Saved queries to: {out_path}")
    print(f"Total time: {elapsed:.2f} seconds")
    if args.stats:
        from quest_stats import compute_and_display_limit_quest_stats
        compute_and_display_limit_quest_stats(
            queries_path=out_path,
            output_dir=data_dir,
            save_table=True,
            save_figure=True,
        )
    else:
        print("To get a statistics table and distribution figure, run: python quest_stats.py")

if __name__ == "__main__":
    main()
