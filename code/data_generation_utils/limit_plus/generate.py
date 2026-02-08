"""
Generate LIMIT-QUEST queries with controlled distribution of num_relevant_docs.

Method: Stratified sampling by relevance-set size (scientifically defendable).
- All queries have num_relevant_docs in [1, 200] (configurable).
- We use fixed buckets over this range (e.g. [1–20], [21–50], [51–100], [101–150], [151–200]).
- Per template we aim for roughly equal count per bucket so templates have similar
  distributions (no template dominating with very high or very low counts).
- Phase 1: accept queries whose num_docs falls in an underfilled bucket for that template.
- Phase 2 (top-up): fill remaining slots with any valid query so we don't get stuck when
  a bucket is impossible to fill (e.g. "_ that are not _" rarely has 150+ docs).
- Diversity is preserved: attributes are still drawn at random; we only change acceptance.
- Fast: bounded failures per phase (8k stratified, 3k top-up) so the script does not hang.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import argparse
import time

DEFAULT_DATA_DIR = "./limit_data/"

# Stratified sampling: buckets over [1, 200] for similar num_relevant_docs distribution per template.
# Each bucket is (low_inclusive, high_inclusive). Queries with num_docs in [1, 200] are accepted;
# we aim for roughly equal count per bucket per template so templates have similar distributions.
RELEVANT_DOCS_MIN = 1
RELEVANT_DOCS_MAX = 200
STRATIFIED_BUCKETS: List[Tuple[int, int]] = [
    (1, 20),
    (21, 50),
    (51, 100),
    (101, 150),
    (151, 200),
]


def bucket_index_for(num_docs: int, buckets: List[Tuple[int, int]]) -> int:
    """Return bucket index (0-based) for num_docs. Assumes num_docs is within some bucket."""
    for i, (lo, hi) in enumerate(buckets):
        if lo <= num_docs <= hi:
            return i
    return -1


def in_relevant_range(num_docs: int, min_d: int, max_d: int) -> bool:
    return min_d <= num_docs <= max_d


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
    *,
    min_relevant_docs: Optional[int] = None,
    max_relevant_docs: Optional[int] = None,
    stratify_buckets: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    queries_path = os.path.join(data_dir, "queries.jsonl")
    qrels_path = os.path.join(data_dir, "qrels.jsonl")

    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        print("No queries.jsonl/qrels.jsonl found; skipping dataset-provided '_' queries.")
        return []

    min_d = min_relevant_docs if min_relevant_docs is not None else 1
    max_d = max_relevant_docs if max_relevant_docs is not None else 999_999

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

    # Load all candidate queries in [min_d, max_d]
    candidates: List[dict] = []
    with open(queries_path, "r", encoding="utf-8") as fq:
        for line in fq:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("_id") or obj.get("id")
            text = obj["text"]
            docs = qid2docs.get(qid)
            if not docs or not in_relevant_range(len(docs), min_d, max_d):
                continue
            attr = text
            prefix = "Who likes "
            if text.startswith(prefix) and text.endswith("?"):
                attr = text[len(prefix):-1].strip()
            candidates.append({
                "qid": qid,
                "text": text,
                "attr": attr,
                "docs": docs,
                "num_docs": len(docs),
            })

    if not candidates:
        print("No '_' queries in relevant-docs range; skipping.")
        return []

    # Stratified selection: aim for similar num_docs distribution across buckets
    if stratify_buckets and len(stratify_buckets) > 0:
        buckets = stratify_buckets
        target_per_bucket = max(1, max_per_template // len(buckets))
        by_bucket: Dict[int, List[dict]] = {i: [] for i in range(len(buckets))}
        for c in candidates:
            bi = bucket_index_for(c["num_docs"], buckets)
            if bi >= 0:
                by_bucket[bi].append(c)
        samples_raw: List[dict] = []
        for bi in range(len(buckets)):
            pool = by_bucket[bi]
            random.shuffle(pool)
            take = min(target_per_bucket, len(pool))
            samples_raw.extend(pool[:take])
        # Top-up to max_per_template if we have room and leftover candidates
        remaining = max_per_template - len(samples_raw)
        if remaining > 0:
            used_qids = {c["qid"] for c in samples_raw}
            extra = [c for c in candidates if c["qid"] not in used_qids]
            random.shuffle(extra)
            for c in extra:
                if len(samples_raw) >= max_per_template:
                    break
                samples_raw.append(c)
    else:
        random.shuffle(candidates)
        samples_raw = candidates[:max_per_template]

    samples = []
    for i, c in enumerate(samples_raw):
        samples.append({
            "id": i,
            "query": c["text"],
            "num_docs": c["num_docs"],
            "docs": c["docs"],
            "original_query": f"<mark>{c['attr']}</mark>",
            "metadata": {
                "template": "_",
                "attrs": [c["attr"]],
                "source": "limit-original",
            },
        })
    n_candidates = len(candidates)
    print(
        f"Loaded {n_candidates} '_' candidates from LIMIT queries.jsonl; "
        f"selected {len(samples)} (stratified by num_relevant_docs, max_per_template={max_per_template})."
    )
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


def query_and_original_for_template(tmpl_name: str, attrs: Tuple[str, ...]) -> Tuple[str, str]:
    """Return (query, original_query) for the given template and attribute tuple."""
    if tmpl_name == "_ or _":
        A, B = attrs[0], attrs[1]
        return f"Who likes {A} or {B}?", f"<mark>{A}</mark> or <mark>{B}</mark>"
    if tmpl_name == "_ or _ or _":
        A, B, C = attrs[0], attrs[1], attrs[2]
        return (
            f"Who likes {A} or {B} or {C}?",
            f"<mark>{A}</mark> or <mark>{B}</mark> or <mark>{C}</mark>",
        )
    if tmpl_name == "_ that are also _":
        A, B = attrs[0], attrs[1]
        return (
            f"Who likes {A} and also likes {B}?",
            f"<mark>{A}</mark> that are also <mark>{B}</mark>",
        )
    if tmpl_name == "_ that are also both _ and _":
        A, B, C = attrs[0], attrs[1], attrs[2]
        return (
            f"Who likes {A} and also likes both {B} and {C}?",
            f"<mark>{A}</mark> that are also both <mark>{B}</mark> and <mark>{C}</mark>",
        )
    if tmpl_name == "_ that are also _ but not _":
        A, B, C = attrs[0], attrs[1], attrs[2]
        return (
            f"Who likes {A} and also likes {B} but not {C}?",
            f"<mark>{A}</mark> that are also <mark>{B}</mark> but not <mark>{C}</mark>",
        )
    if tmpl_name == "_ that are not _":
        A, B = attrs[0], attrs[1]
        return (
            f"Who likes {A} but not {B}?",
            f"<mark>{A}</mark> that are not <mark>{B}</mark>",
        )
    raise ValueError(f"Unknown template for query text: {tmpl_name}")


def enumerate_valid_combinations(
    tmpl_name: str,
    attr2entities: Dict[str, Set[str]],
    min_d: int,
    max_d: int,
    progress_interval: Optional[int] = 200_000,
) -> Optional[List[Tuple[Tuple[str, ...], int]]]:
    """
    Enumerate all (attrs, num_docs) for this template with num_docs in [min_d, max_d].
    Returns None if this template is not supported for enumeration (e.g. '_').
    Used to know exactly how many queries are possible and to sample without rejection.
    """
    all_attrs = sorted(attr2entities.keys())
    if not all_attrs:
        return []

    result: List[Tuple[Tuple[str, ...], int]] = []
    count = 0

    def maybe_progress() -> None:
        nonlocal count
        count += 1
        if progress_interval and count % progress_interval == 0:
            print(f"    ... {count} combinations checked", flush=True)

    if tmpl_name == "_ or _":
        for i, A in enumerate(all_attrs):
            for B in all_attrs[i + 1 :]:
                maybe_progress()
                n = len(attr2entities[A] | attr2entities[B])
                if min_d <= n <= max_d:
                    result.append(((A, B), n))
        return result

    if tmpl_name == "_ or _ or _":
        for i, A in enumerate(all_attrs):
            for j, B in enumerate(all_attrs[i + 1 :], i + 1):
                for C in all_attrs[j + 1 :]:
                    maybe_progress()
                    n = len(attr2entities[A] | attr2entities[B] | attr2entities[C])
                    if min_d <= n <= max_d:
                        result.append(((A, B, C), n))
        return result

    if tmpl_name == "_ that are also _":
        for i, A in enumerate(all_attrs):
            for B in all_attrs[i + 1 :]:
                maybe_progress()
                n = len(attr2entities[A] & attr2entities[B])
                if min_d <= n <= max_d:
                    result.append(((A, B), n))
        return result

    if tmpl_name == "_ that are also both _ and _":
        for i, A in enumerate(all_attrs):
            for j, B in enumerate(all_attrs[i + 1 :], i + 1):
                for C in all_attrs[j + 1 :]:
                    maybe_progress()
                    n = len(
                        attr2entities[A]
                        & attr2entities[B]
                        & attr2entities[C]
                    )
                    if min_d <= n <= max_d:
                        result.append(((A, B, C), n))
        return result

    if tmpl_name == "_ that are also _ but not _":
        for A in all_attrs:
            for B in all_attrs:
                if B == A:
                    continue
                for C in all_attrs:
                    if C == A or C == B:
                        continue
                    maybe_progress()
                    n = len(
                        (attr2entities[A] & attr2entities[B]) - attr2entities[C]
                    )
                    if min_d <= n <= max_d:
                        result.append(((A, B, C), n))
        return result

    if tmpl_name == "_ that are not _":
        for A in all_attrs:
            for B in all_attrs:
                if B == A:
                    continue
                maybe_progress()
                n = len(attr2entities[A] - attr2entities[B])
                if min_d <= n <= max_d:
                    result.append(((A, B), n))
        return result

    return None


def docs_for_template_from_sets(
    tmpl_name: str,
    attrs: Tuple[str, ...],
    attr2entities: Dict[str, Set[str]],
) -> Optional[Tuple[List[str], int]]:
    """
    Compute (docs, num_docs) using set operations on attr2entities only (no corpus scan).
    Returns None if no fast path for this template (fall back to predicate scan).
    """
    if not attrs or any(a not in attr2entities for a in attrs):
        return None
    if tmpl_name == "_ or _":
        if len(attrs) != 2:
            return None
        A, B = attrs[0], attrs[1]
        docs_set = attr2entities[A] | attr2entities[B]
    elif tmpl_name == "_ or _ or _":
        if len(attrs) != 3:
            return None
        A, B, C = attrs[0], attrs[1], attrs[2]
        docs_set = attr2entities[A] | attr2entities[B] | attr2entities[C]
    elif tmpl_name == "_ that are also _":
        if len(attrs) != 2:
            return None
        A, B = attrs[0], attrs[1]
        docs_set = attr2entities[A] & attr2entities[B]
    elif tmpl_name == "_ that are also both _ and _":
        if len(attrs) != 3:
            return None
        A, B, C = attrs[0], attrs[1], attrs[2]
        docs_set = attr2entities[A] & attr2entities[B] & attr2entities[C]
    elif tmpl_name == "_ that are also _ but not _":
        if len(attrs) != 3:
            return None
        A, B, C = attrs[0], attrs[1], attrs[2]
        docs_set = (attr2entities[A] & attr2entities[B]) - attr2entities[C]
    elif tmpl_name == "_ that are not _":
        if len(attrs) != 2:
            return None
        A, B = attrs[0], attrs[1]
        docs_set = attr2entities[A] - attr2entities[B]
    else:
        return None
    docs_list = list(docs_set)
    return docs_list, len(docs_list)


# When a template would require checking more than this many combinations,
# we skip full enumeration and use fast rejection sampling instead (still O(1) per try).
ENUMERATION_MAX_COMBINATIONS = 50_000


def _max_combinations_for_template(tmpl_name: str, n_attrs: int) -> int:
    """Upper bound on number of (attrs) combinations for this template."""
    if n_attrs <= 0:
        return 0
    if tmpl_name in ("_ or _", "_ that are also _"):
        return n_attrs * (n_attrs - 1) // 2  # unordered pairs
    if tmpl_name == "_ that are not _":
        return n_attrs * (n_attrs - 1)  # ordered pairs
    if tmpl_name in ("_ or _ or _", "_ that are also both _ and _"):
        return (n_attrs * (n_attrs - 1) * (n_attrs - 2)) // 6  # unordered triples
    if tmpl_name == "_ that are also _ but not _":
        return n_attrs * (n_attrs - 1) * (n_attrs - 2)  # ordered triples
    return 0


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



def _write_checkpoint(
    checkpoint_path: str,
    single_samples: List[dict],
    gen_samples: List[dict],
) -> None:
    """Write current progress to checkpoint file (vanilla + non-vanilla so far)."""
    combined = single_samples + gen_samples
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for s in combined:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Checkpointed to {checkpoint_path} ({len(combined)} queries).", flush=True)


def build_quest_like_samples(
    entity2attrs: Dict[str, AttrSet],
    attr2entities: Dict[str, Set[str]],
    max_per_template: int = 100,
    seed: int = 42,
    skip_single: bool = False,
    start_id: int = 0,
    *,
    min_relevant_docs: Optional[int] = None,
    max_relevant_docs: Optional[int] = None,
    use_stratified: bool = True,
    buckets: Optional[List[Tuple[int, int]]] = None,
    checkpoint_path: Optional[str] = None,
    single_samples_for_checkpoint: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Generate query samples per template with controlled relevance-set size [min, max].

    If use_stratified is True (default), we use stratified sampling by num_relevant_docs:
    - Queries must have num_docs in [min_relevant_docs, max_relevant_docs] (e.g. 1–200).
    - We aim for roughly equal count per bucket per template so all templates have similar
      distribution of num_relevant_docs (scientifically defendable, avoids one template
      dominating with very high or very low counts).
    - Phase 1: accept queries whose bucket for this template is below target.
    - Phase 2 (top-up): fill remaining slots with any valid query so we don't get stuck
      when a bucket is impossible to fill (e.g. "_ that are not _" rarely has 150+ docs).
    Diversity is preserved: we still draw attributes at random; we only change acceptance.
    """
    random.seed(seed)

    min_d = min_relevant_docs if min_relevant_docs is not None else RELEVANT_DOCS_MIN
    max_d = max_relevant_docs if max_relevant_docs is not None else RELEVANT_DOCS_MAX
    buck = buckets if buckets is not None else STRATIFIED_BUCKETS
    target_per_bucket = max(1, max_per_template // len(buck)) if use_stratified else max_per_template

    all_attrs = sorted(attr2entities.keys())
    samples: List[dict] = []
    sample_id = start_id

    seen_keys_per_template: Dict[str, Set[Tuple[str, ...]]] = {
        tmpl_name: set() for tmpl_name, _ in TEMPLATES
    }
    bucket_counts_per_template: Dict[str, List[int]] = {
        tmpl_name: [0] * len(buck) for tmpl_name, _ in TEMPLATES
    }

    max_failures_phase1 = 8_000
    max_failures_topup = 3_000

    for tmpl_name, builder in TEMPLATES:
        if skip_single and tmpl_name == "_":
            continue

        # Fast path: enumerate all valid (attrs, num_docs) when combination count is small
        n_attrs = len(attr2entities)
        max_comb = _max_combinations_for_template(tmpl_name, n_attrs)
        if max_comb > ENUMERATION_MAX_COMBINATIONS:
            print(
                f"  Template '{tmpl_name}': skipping full enumeration (~{max_comb} combinations); "
                f"using fast rejection sampling to get {max_per_template}.",
                flush=True,
            )
            candidates = None
        else:
            print(f"  Enumerating template '{tmpl_name}'...", flush=True)
            candidates = enumerate_valid_combinations(
                tmpl_name, attr2entities, min_d, max_d
            )

        if candidates is not None and len(candidates) > 0:
            n_possible = len(candidates)
            print(
                f"  Template '{tmpl_name}': {n_possible} valid combinations in range.",
                flush=True,
            )
            # Stratified sample from candidates (no rejection loop)
            by_bucket: Dict[int, List[Tuple[Tuple[str, ...], int]]] = {
                i: [] for i in range(len(buck))
            }
            for (attrs, num_docs) in candidates:
                bi = bucket_index_for(num_docs, buck)
                if bi >= 0:
                    by_bucket[bi].append((attrs, num_docs))

            samples_raw: List[Tuple[Tuple[str, ...], int]] = []
            if use_stratified:
                for bi in range(len(buck)):
                    pool = by_bucket[bi]
                    random.shuffle(pool)
                    take = min(target_per_bucket, len(pool))
                    samples_raw.extend(pool[:take])
                used_attrs = {c[0] for c in samples_raw}
                remaining = [c for c in candidates if c[0] not in used_attrs]
                random.shuffle(remaining)
                for (attrs, num_docs) in remaining:
                    if len(samples_raw) >= max_per_template:
                        break
                    samples_raw.append((attrs, num_docs))
            else:
                random.shuffle(candidates)
                samples_raw = candidates[:max_per_template]

            # Cap at max_per_template (we may have more from stratified+top-up)
            samples_raw = samples_raw[:max_per_template]

            for (attrs, num_docs) in samples_raw:
                bi = bucket_index_for(num_docs, buck)
                if bi >= 0:
                    bucket_counts_per_template[tmpl_name][bi] += 1
                fast_result = docs_for_template_from_sets(
                    tmpl_name, attrs, attr2entities
                )
                assert fast_result is not None
                docs, _ = fast_result
                query, original_query = query_and_original_for_template(
                    tmpl_name, attrs
                )
                sample = {
                    "id": sample_id,
                    "query": query,
                    "num_docs": num_docs,
                    "docs": docs,
                    "original_query": original_query,
                    "metadata": {"template": tmpl_name, "attrs": list(attrs)},
                }
                samples.append(sample)
                sample_id += 1

            n = len(samples_raw)
            cap_note = (
                f" (capped at {n_possible} possible)" if n_possible < max_per_template else ""
            )
            print(
                f"Template '{tmpl_name}': {n} unique queries "
                f"(stratified buckets: {bucket_counts_per_template[tmpl_name]}){cap_note}.",
                flush=True,
            )
            print(f"  Total generated so far: {len(samples)}", flush=True)
            if checkpoint_path and single_samples_for_checkpoint is not None:
                _write_checkpoint(checkpoint_path, single_samples_for_checkpoint, samples)
            continue

        # Fallback: enumeration not supported or empty — use rejection sampling
        print(f"  Building template '{tmpl_name}' (rejection sampling)...", flush=True)
        failures = 0
        phase = "stratified"
        attempts = 0

        while len(seen_keys_per_template[tmpl_name]) < max_per_template:
            if phase == "stratified" and failures >= max_failures_phase1:
                phase = "topup"
                failures = 0
            if phase == "topup" and failures >= max_failures_topup:
                break

            attempts += 1
            if attempts % 2000 == 0:
                n_so_far = len(seen_keys_per_template[tmpl_name])
                print(
                    f"    '{tmpl_name}': {n_so_far}/{max_per_template} queries "
                    f"(phase={phase}, failures={failures})",
                    flush=True,
                )

            inst = builder(
                all_attrs=all_attrs,
                entity2attrs=entity2attrs,
                attr2entities=attr2entities,
            )

            key = tuple(sorted(inst.attrs))
            if key in seen_keys_per_template[tmpl_name]:
                failures += 1
                continue

            fast_result = docs_for_template_from_sets(
                tmpl_name, inst.attrs, attr2entities
            )
            if fast_result is not None:
                docs, num_docs = fast_result
            else:
                docs = [
                    eid
                    for eid, attrs in entity2attrs.items()
                    if inst.predicate(attrs)
                ]
                num_docs = len(docs)
            if not docs:
                failures += 1
                continue
            if not in_relevant_range(num_docs, min_d, max_d):
                failures += 1
                continue

            if phase == "stratified" and use_stratified:
                bi = bucket_index_for(num_docs, buck)
                if (
                    bi < 0
                    or bucket_counts_per_template[tmpl_name][bi]
                    >= target_per_bucket
                ):
                    failures += 1
                    continue

            seen_keys_per_template[tmpl_name].add(key)
            bi = bucket_index_for(num_docs, buck)
            if bi >= 0:
                bucket_counts_per_template[tmpl_name][bi] += 1

            sample = {
                "id": sample_id,
                "query": inst.query,
                "num_docs": num_docs,
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

        n = len(seen_keys_per_template[tmpl_name])
        if n < max_per_template:
            print(
                f"Template '{tmpl_name}': {n} unique queries "
                f"(stratified buckets: {bucket_counts_per_template[tmpl_name]}) "
                f"[stopped before {max_per_template} — not enough valid combinations].",
                flush=True,
            )
        else:
            print(
                f"Template '{tmpl_name}': {n} unique queries "
                f"(stratified buckets: {bucket_counts_per_template[tmpl_name]}).",
                flush=True,
            )
        print(f"  Total generated so far: {len(samples)}", flush=True)
        if checkpoint_path and single_samples_for_checkpoint is not None:
            _write_checkpoint(checkpoint_path, single_samples_for_checkpoint, samples)

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
        "--min-relevant",
        type=int,
        default=RELEVANT_DOCS_MIN,
        metavar="N",
        help=f"Only keep queries with at least N relevant docs (default: {RELEVANT_DOCS_MIN})",
    )
    parser.add_argument(
        "--max-relevant",
        type=int,
        default=RELEVANT_DOCS_MAX,
        metavar="N",
        help=f"Only keep queries with at most N relevant docs (default: {RELEVANT_DOCS_MAX})",
    )
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="Disable stratified sampling; only apply min/max filter (may yield skewed per-template distributions)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Output path for queries JSONL (default: {data-dir}/limit_quest_queries.jsonl). Use to avoid overwriting an existing file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="After each template, write current progress to PATH. If the process is killed, you keep partial output.",
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
    min_relevant = args.min_relevant
    max_relevant = args.max_relevant
    use_stratified = not args.no_stratified

    print(f"Relevant-docs: min={min_relevant}, max={max_relevant}; stratified={use_stratified}")

    stratify_buckets = STRATIFIED_BUCKETS if use_stratified else None
    single_samples = load_limit_single_template_queries(
        data_dir=data_dir,
        max_per_template=max_per_template,
        min_relevant_docs=min_relevant,
        max_relevant_docs=max_relevant,
        stratify_buckets=stratify_buckets,
    )

    print(f"\nGenerating up to {max_per_template} unique queries per template for non-vanilla templates...")
    gen_samples = build_quest_like_samples(
        entity2attrs=entity2attrs,
        attr2entities=attr2entities,
        max_per_template=max_per_template,
        seed=0,
        skip_single=True,
        start_id=len(single_samples),
        min_relevant_docs=min_relevant,
        max_relevant_docs=max_relevant,
        use_stratified=use_stratified,
        buckets=STRATIFIED_BUCKETS,
        checkpoint_path=args.checkpoint,
        single_samples_for_checkpoint=single_samples if args.checkpoint else None,
    )

    samples = single_samples + gen_samples
    print(f"\nTotal queries generated: {len(samples)}")

    out_path = args.output if args.output else os.path.join(data_dir, "limit_quest_queries.jsonl")
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
