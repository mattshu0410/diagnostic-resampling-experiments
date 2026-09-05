"""Choosing which cases to resample.

The pool is drawn once for every model together, spread as evenly over the cohort's sources as
their sizes allow. Drawing per model instead would leave the sweeps sharing almost no cases, and
every cross-model comparison would then be over different clinical material.

Selection reads the cohort file and nothing else, so it reproduces from the cohort and the seed
alone, before any model has run. Per-model eligibility is reported but never filters, since a
case one model cannot be addressed on is still worth screening on the others.

Screening resamples boundary 0 only, which asks how much a model's answer moves when it
re-reasons the case from scratch. The sweep goes to the cases where it moves most, plus a few
where it does not move at all: a case whose answer is fixed cannot have a causally important
sentence, so it reads as a floor for the importance metric.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq

from dctax.config import data_root, load_model
from dctax.data import cohort as cohorts
from dctax.generate.responses import load_existing, responses_path
from dctax.rollouts import boundaries, store
from dctax.utils.traces import extract_reasoning, split_into_sentences


def taxonomy_cases(solve: str, model: str) -> set[str]:
    """Cases the named barycentre solve assigns for one model."""
    path = data_root() / "taxonomy" / solve / "assignments.csv"
    with path.open() as handle:
        return {r["case_id"] for r in csv.DictReader(handle) if r["model"] == model}


def eligible(slug: str) -> dict[str, int]:
    """Sentence count per case this model can be addressed on.

    A case is eligible when its reasoning extracts as a verbatim slice of the response, since
    only then can sentences be located as exact token spans.

    The question is not passed to `extract_reasoning`, matching `activations.collect`, which is
    what the stored sentence indices are keyed to. Passing it is a different call.
    """
    cfg = load_model(slug)
    out: dict[str, int] = {}
    for record in load_existing(responses_path(slug)):
        full = record.get("full_response") or ""
        reasoning = extract_reasoning(full, trace_format=cfg.expected_trace_format)
        if not reasoning or full.find(reasoning) < 0:
            continue
        count = len(split_into_sentences(reasoning, cfg.sentence_rules))
        if count:
            out[record["pmcid"]] = count
    return out


def source_quotas(available: dict[str, int], total: int) -> dict[str, int]:
    """Cases per source, as even as their sizes allow.

    A source holding fewer than the even share contributes all of it, and its shortfall is
    spread over the sources that still have surplus. Repeated until every remaining source can
    meet the share, so an exhausted source never blocks the target.
    """
    quota = {s: 0 for s in available}
    pool, left = sorted(available), total
    while pool:
        share = left // len(pool)
        short = [s for s in pool if available[s] <= share]
        if not short:
            for i, s in enumerate(pool):
                quota[s] = share + (1 if i < left - share * len(pool) else 0)
            break
        for s in short:
            quota[s] = available[s]
            left -= available[s]
            pool.remove(s)
    return quota


def build_pool(
    slugs: Sequence[str],
    *,
    cohort: str = "main2073",
    total: int = 1000,
    seed: int = 2026,
    verbose: bool = True,
) -> dict:
    """One pool of cases shared by every model, spread evenly over the cohort's sources.

    Sources are visited in name order and drawn from one seeded stream, so the whole selection
    is fixed by (cohort, total, seed). Case ids are sorted within each source before sampling,
    so the draw does not depend on the order records sit in the cohort file.
    """
    cases = cohorts.load(cohort)
    by_source: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_source[case.source].append(case.case_id)
    for ids in by_source.values():
        ids.sort()

    quota = source_quotas({s: len(v) for s, v in by_source.items()}, total)
    rng = random.Random(seed)
    picked: list[str] = []
    for source in sorted(by_source):
        picked.extend(rng.sample(by_source[source], quota[source]))
    picked.sort()

    source_of = {c.case_id: c.source for c in cases}
    counts = {s: eligible(s) for s in slugs} if verbose else {}
    if verbose:
        print(f"cohort {cohort}: {len(cases)} cases, drawing {total} at seed {seed}")
        for source in sorted(by_source):
            print(f"  {source:<12} available {len(by_source[source]):>5}   taken {quota[source]:>5}")
        for slug in slugs:
            have = sum(1 for c in picked if c in counts[slug])
            print(f"  eligible in {slug:<32} {have:>5}/{len(picked)}")

    rows = [
        {
            "case_id": c,
            "source": source_of[c],
            "sentences": {s: counts.get(s, {}).get(c) for s in slugs},
        }
        for c in picked
    ]
    return {"cohort": cohort, "models": list(slugs), "seed": seed, "total": total,
            "quotas": quota, "case_ids": picked, "rows": rows}


def capped_cases(run: str, cap: int) -> set[str]:
    """Cases where any rollout ran to the token cap, which the sweep would run slowly on."""
    out: set[str] = set()
    for path in store.part_files(run):
        table = pq.read_table(path, columns=["case_id", "finish_reason", "output_token_ids"])
        for case, reason, ids in zip(
            table.column("case_id").to_pylist(),
            table.column("finish_reason").to_pylist(),
            table.column("output_token_ids").to_pylist(),
        ):
            if reason == "length" or len(ids) >= cap:
                out.add(case)
    return out


def variability(
    runs: dict[str, str], grades: dict[tuple[str, str], dict], *, boundary: int = 0
) -> dict[str, dict]:
    """Per case, how concentrated each model's answers are, and the mean across models.

    Answers are counted as the entities the judge grouped them into, so a case does not look
    variable merely because a model phrased the same diagnosis several ways.
    """
    per_case: dict[str, dict[str, Counter]] = {}
    for slug, run in runs.items():
        path = store.run_dir(run) / "rollout_scores.parquet"
        for row in pq.read_table(
            path, columns=["case_id", "boundary_index", "raw_answer", "valid"]
        ).to_pylist():
            if row["boundary_index"] != boundary or not row["valid"] or not row["raw_answer"]:
                continue
            key = (row["case_id"], row["raw_answer"].strip())
            entity = grades.get(key, {}).get("canonical") or row["raw_answer"].strip()
            per_case.setdefault(row["case_id"], {}).setdefault(slug, Counter())[entity] += 1

    out = {}
    for case, models in per_case.items():
        shares, entities = {}, {}
        for slug, counts in models.items():
            total = sum(counts.values())
            shares[slug] = max(counts.values()) / total if total else None
            entities[slug] = len(counts)
        usable = [v for v in shares.values() if v is not None]
        out[case] = {
            "case_id": case,
            "modal_share": shares,
            "entities": entities,
            "mean_modal_share": sum(usable) / len(usable) if usable else None,
            "n_models": len(usable),
        }
    return out


def outcomes(
    runs: dict[str, str], grades: dict[tuple[str, str], dict], *, boundary: int = 0
) -> dict[str, dict[str, list[tuple[int, bool]]]]:
    """Per model, per case, each screening sample as (rollout_index, judged correct).

    The index is carried so a selection can name which sample to sweep rather than only how
    many went each way. Only samples the judge returned a verdict for are counted, so a case
    can carry fewer than the configured screen_rollouts.
    """
    out: dict[str, dict[str, list[tuple[int, bool]]]] = {}
    for slug, run in runs.items():
        per_case: dict[str, list[tuple[int, bool]]] = {}
        path = store.run_dir(run) / "rollout_scores.parquet"
        for row in pq.read_table(
            path, columns=["case_id", "boundary_index", "rollout_index", "raw_answer", "valid"]
        ).to_pylist():
            if row["boundary_index"] != boundary or not row["valid"] or not row["raw_answer"]:
                continue
            verdict = grades.get((row["case_id"], row["raw_answer"].strip()), {}).get("is_correct")
            if verdict is not None:
                per_case.setdefault(row["case_id"], []).append(
                    (int(row["rollout_index"]), bool(verdict))
                )
        out[slug] = {c: sorted(v) for c, v in per_case.items()}
    return out


def buildable(
    judged: dict[str, dict[str, list[tuple[int, bool]]]],
    runs: dict[str, str],
    *,
    verbose: bool = True,
) -> dict[str, dict[str, list[tuple[int, bool]]]]:
    """`judged` with the samples that cannot be rebuilt into a sweepable trace removed.

    A sample only reaches the sweep if `prepare_sample` can turn it back into a trace whose
    sentences address exact token spans, and the same gates that excluded cases during the
    screen apply again: reasoning that does not extract, or that is not a verbatim slice of the
    rebuilt response. Filtering here rather than at sweep time means a case is not chosen on the
    strength of a sample that was never going to run, and a case keeps its place when another
    sample of the same outcome does build.

    Loading a tokenizer per model is enough; no engine is needed.
    """
    from transformers import AutoTokenizer

    out: dict[str, dict[str, list[tuple[int, bool]]]] = {}
    for slug, per_case in judged.items():
        cfg = load_model(slug)
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
        records = {r.get("pmcid"): r for r in load_existing(responses_path(slug))}
        bases = {t.case_id: t for t in store.load_traces(runs[slug])}
        kept: dict[str, list[tuple[int, bool]]] = {}
        dropped = 0
        for case, samples in per_case.items():
            base = bases.get(case)
            if base is None:
                dropped += len(samples)
                continue
            rollouts = {r.rollout_index: r for r in store.load_rollouts(runs[slug], case)
                        if r.boundary_index == 0}
            survivors = []
            for index, correct in samples:
                sample = rollouts.get(index)
                if sample is None:
                    dropped += 1
                    continue
                try:
                    boundaries.prepare_sample(
                        base, sample.prefix_token_count, sample.continuation,
                        {**records.get(case, {}), "pmcid": case}, cfg, tokenizer,
                    )
                except boundaries.BoundaryError:
                    dropped += 1
                    continue
                survivors.append((index, correct))
            if survivors:
                kept[case] = survivors
        out[slug] = kept
        if verbose:
            before = sum(len(v) for v in per_case.values())
            print(f"  {slug:<32} {before - dropped:>6}/{before:<6} samples build "
                  f"({dropped} dropped)", flush=True)
    return out


def select(
    judged: dict[str, dict[str, list[tuple[int, bool]]]],
    *,
    core_min: int = 6,
    floor: int = 100,
    min_each: int = 1,
    exclude: Iterable[str] = (),
) -> dict:
    """Which cases to sweep in which model.

    A case qualifies for a model when that model's screening samples include at least
    `min_each` correct and `min_each` wrong, since the sweep needs both arms from one case to
    compare them without difficulty differing between them. Each row names the lowest-numbered
    sample of each outcome, so which two traces get swept is fixed by the screening data.

    Cases qualifying for `core_min` models or more are swept in every model that qualifies for
    them; that shared set is what lets models be compared on identical material. A model short
    of `floor` then takes further cases from its own qualifying list, most widely shared first,
    so even the additions overlap other models where they can.

    Ordering is (-depth, case_id) throughout, so the result is fixed by the screening data with
    no random draw to reproduce.
    """
    exclude = set(exclude)
    models = list(judged)
    qualifies = {
        slug: {
            case: v for case, v in per_case.items()
            if case not in exclude
            and sum(1 for _, ok in v if ok) >= min_each
            and sum(1 for _, ok in v if not ok) >= min_each
        }
        for slug, per_case in judged.items()
    }
    every = set().union(*(set(q) for q in qualifies.values())) if qualifies else set()
    depth = {case: sum(case in qualifies[s] for s in models) for case in every}
    rank = lambda case: (-depth[case], case)  # noqa: E731

    core = sorted((c for c in every if depth[c] >= core_min), key=rank)
    core_set = set(core)
    chosen = {s: [c for c in core if c in qualifies[s]] for s in models}
    for slug in models:
        short = floor - len(chosen[slug])
        if short > 0:
            rest = sorted((c for c in qualifies[slug] if c not in core_set), key=rank)
            chosen[slug].extend(rest[:short])

    rows = []
    for slug in models:
        for case in sorted(chosen[slug]):
            v = qualifies[slug][case]
            right = [i for i, ok in v if ok]
            wrong = [i for i, ok in v if not ok]
            rows.append({
                "model": slug, "case_id": case, "depth": depth[case],
                "in_core": case in core_set,
                "correct_index": right[0], "wrong_index": wrong[0],
                "n_correct": len(right), "n_wrong": len(wrong), "n_graded": len(v),
            })
    return {
        "core_min": core_min, "floor": floor, "min_each": min_each, "models": models,
        "case_ids": sorted({c for v in chosen.values() for c in v}),
        "core": core,
        "by_model": {s: sorted(v) for s, v in chosen.items()},
        "rows": rows,
        "excluded": sorted(exclude),
    }


def write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path
