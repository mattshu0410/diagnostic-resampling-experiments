"""Answer equivalence and correctness for rollouts.

A sweep produces hundreds of thousands of rollouts but only tens of distinct diagnoses per
case, so answers are judged rather than rollouts. Distinct answers are grouped into clinical
entities first, then one representative per entity is scored, and every rollout giving an
answer inherits its entity's verdict. Grading before grouping would let two answers in the
same entity fall either side of the correctness threshold.

The judgement itself is `generate.grade`'s: describe both diagnoses in the context of the
case, then rate the descriptions for similarity. Only the addressing differs.

A sweep case carries hundreds of distinct answers, more than one call can group, so grouping
runs in rounds over batches and narrows the case each time.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from dctax.config import ModelConfig, load_prompt
from dctax.generate.grade import (
    COMPARE_PROMPT,
    CORRECT_AT,
    DESCRIBE_PROMPT,
    Description,
    Similarity,
    case_text,
)
from dctax.rollouts import store
from dctax.utils import llm

BUCKET_PROMPT = "bucket_answers"

#: In-flight judge requests. Grading is the longest stage after a sweep and sits between the
#: rollouts existing and any result existing, so it runs wider than the library default.
CONCURRENCY = 32

#: Diagnoses per grouping call. The judge holds alignment over a batch this size; at several
#: hundred it truncates its reply and misassigns what it does return.
BATCH = 64

#: Grouping rounds before a case is left as it stands.
MAX_ROUNDS = 8

#: Example members shown beside a name, so a round groups on content rather than on a label.
EXEMPLARS = 3

_PUNCTUATION = re.compile(r"[^\w\s]")
_EXAMPLES = re.compile(r"\s*\(e\.g\..*\)\s*$", re.S)

GRADE_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("raw_answer", pa.string()),
        ("canonical", pa.string()),
        ("similarity", pa.float32()),
        ("is_correct", pa.bool_()),
        ("n_rollouts", pa.int32()),
    ]
)


class Bucket(BaseModel):
    """Diagnoses naming one clinical entity."""

    canonical: str
    members: list[str]


class AnswerBuckets(BaseModel):
    """Every candidate diagnosis for a case, grouped by the entity it names."""

    buckets: list[Bucket]


def _assign(candidates: Sequence[str], reply: AnswerBuckets | None) -> dict[str, str]:
    """Canonical entity per candidate, from one bucketing reply.

    The judge is asked to echo each input verbatim into exactly one bucket, and does not
    always oblige: it repeats members across buckets and invents ones that were never sent.
    Anything outside the input is discarded and the first assignment of a repeat wins, so a
    duplicate cannot land an answer in two entities. An answer the judge omits becomes its
    own entity rather than disappearing, since a dropped answer would shrink the case's
    answer distribution and inflate every divergence computed from it.
    """
    unique = list(dict.fromkeys(a.strip() for a in candidates if a and a.strip()))
    wanted = set(unique)
    mapping: dict[str, str] = {}
    for bucket in reply.buckets if reply else []:
        name = bucket.canonical.strip() or next(iter(bucket.members), "")
        for member in bucket.members:
            member = member.strip()
            if member in wanted and member not in mapping:
                mapping[member] = name or member
    for answer in unique:
        mapping.setdefault(answer, answer)
    return mapping


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def _partition(names: Sequence[str], size: int, *, strided: bool) -> list[list[str]]:
    """Split `names` into groups of at most `size`, contiguously or by stride.

    Names are sorted, so a contiguous split puts spelling variants of one diagnosis in one
    group. Repeating that split on names that survived it is a fixed point: the same names
    meet again and nothing further merges. A strided split pairs names the contiguous one
    never places together.
    """
    if not names:
        return []
    groups = max(1, -(-len(names) // size))
    if strided:
        return [list(names[i::groups]) for i in range(groups)]
    return [list(names[i : i + size]) for i in range(0, len(names), size)]


def _examples(name: str, members: Sequence[str]) -> list[str]:
    """Up to EXEMPLARS members, spread across the sorted list, excluding the name itself.

    Spread rather than most frequent: rollout counts would tell the judge which names are
    already large and invite it to group towards them.
    """
    others = sorted({m for m in members if m != name})
    if len(others) <= EXEMPLARS:
        return others
    step = (len(others) - 1) / (EXEMPLARS - 1)
    return [others[round(i * step)] for i in range(EXEMPLARS)]


def _shown(name: str, members: Sequence[str]) -> str:
    """A name as the judge sees it: itself, plus examples of what it already covers."""
    examples = _examples(name, members)
    return f"{name} (e.g. {'; '.join(examples)})" if examples else name


def _collapse(
    cases: Sequence[str],
    contexts: dict[str, str],
    answers: dict[str, Sequence[str]],
    *,
    llm_cfg: llm.LLMConfig | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, str]]:
    """Canonical entity per answer, for every case.

    Exact matches collapse first, then each round groups the surviving names in batches and
    replaces every name with the canonical the judge returned for it, so a case narrows over
    several rounds. Batching alternates contiguous and strided partitions; a round that adds
    nothing twice running ends the case, and a case small enough for one batch ends after it,
    since that batch is what merges names first seen apart.

    Every case is batched into the same round, so width comes from the whole set rather than
    from one case at a time.
    """
    label: dict[str, dict[str, str]] = {}
    for case in cases:
        seen: dict[str, str] = {}
        label[case] = {}
        for answer in dict.fromkeys(a.strip() for a in answers[case] if a and a.strip()):
            label[case][answer] = seen.setdefault(_normalise(answer), answer)

    previous, stalled = sum(len(set(v.values())) for v in label.values()), 0
    if verbose:
        print(f"  {sum(len(v) for v in label.values()):,} answers -> {previous:,} "
              f"after exact match", flush=True)

    for index in range(MAX_ROUNDS):
        members: dict[str, dict[str, list[str]]] = {}
        for case in cases:
            grouped: dict[str, list[str]] = {}
            for answer, name in label[case].items():
                grouped.setdefault(name, []).append(answer)
            members[case] = grouped

        work = []
        for case in cases:
            names = sorted(members[case], key=lambda t: (_normalise(t), t))
            for batch in _partition(names, BATCH, strided=bool(index % 2)):
                work.append((case, batch, [_shown(n, members[case][n]) for n in batch]))
        last = all(len(members[c]) <= BATCH for c in cases)

        replies = llm.map_structured(
            [{"case": contexts[case], "diagnoses": shown} for case, _, shown in work],
            system=load_prompt(BUCKET_PROMPT),
            schema=AnswerBuckets,
            cfg=llm_cfg,
            progress=verbose,
        )

        merged: dict[str, dict[str, str]] = {case: {} for case in cases}
        failed = 0
        for (case, batch, shown), reply in zip(work, replies):
            if reply is None:
                failed += 1
                continue
            assigned = _assign(shown, reply)
            for name, text in zip(batch, shown):
                merged[case][name] = _EXAMPLES.sub("", assigned.get(text, name)).strip() or name
        for case in cases:
            label[case] = {a: merged[case].get(n, n) for a, n in label[case].items()}

        total = sum(len(set(v.values())) for v in label.values())
        if verbose:
            note = f", {failed} of {len(work)} calls failed" if failed else ""
            print(f"  round {index + 1}: {len(work)} batches -> {total:,} names{note}",
                  flush=True)
        if last:
            break
        stalled = stalled + 1 if total == previous else 0
        if stalled >= 2:
            if verbose:
                print(f"  stopped: two rounds added nothing, {total:,} names stand",
                      flush=True)
            break
        previous = total

    return label


def bucket_answers(
    case: str, candidates: Sequence[str], *, llm_cfg: llm.LLMConfig | None = None
) -> dict[str, str]:
    """Canonical entity for each candidate diagnosis."""
    if not any(a and a.strip() for a in candidates):
        return {}
    return _collapse(
        ["case"], {"case": case}, {"case": list(candidates)}, llm_cfg=llm_cfg
    )["case"]


def grade_answers(
    case: str, gold: str, candidates: Sequence[str], *, llm_cfg: llm.LLMConfig | None = None
) -> dict[str, float]:
    """Similarity to `gold` for each candidate, describing both sides before comparing."""
    unique = list(dict.fromkeys(a.strip() for a in candidates if a and a.strip()))
    if not unique:
        return {}

    described = llm.map_structured(
        [{"case": case, "diagnosis": d} for d in [gold, *unique]],
        system=load_prompt(DESCRIBE_PROMPT),
        schema=Description,
        cfg=llm_cfg,
        progress=False,
    )
    true_side, predicted = described[0], described[1:]
    if true_side is None:
        return {}

    scored = [i for i, d in enumerate(predicted) if d is not None]
    replies = llm.map_structured(
        [
            {
                "case": case,
                "predicted_diagnosis": predicted[i].description,
                "true_diagnosis": true_side.description,
            }
            for i in scored
        ],
        system=load_prompt(COMPARE_PROMPT),
        schema=Similarity,
        cfg=llm_cfg,
        progress=False,
    )
    return {unique[i]: r.score for i, r in zip(scored, replies) if r is not None}


def grade_run(
    runs: Sequence[str],
    records: dict[str, dict],
    *,
    llm_cfg: llm.LLMConfig | None = None,
    out: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Bucket and grade every distinct answer across `runs`, sharing entities per case.

    Answers from every model are bucketed together, so a case's entities mean the same thing
    in each model's answer distribution and the distributions stay comparable.
    """
    llm_cfg = llm_cfg or llm.LLMConfig(concurrency=CONCURRENCY)
    answers_by_case: dict[str, dict[str, int]] = {}
    for run in runs:
        table = pq.read_table(
            store.run_dir(run) / "rollout_scores.parquet",
            columns=["case_id", "raw_answer", "valid"],
        ).to_pylist()
        for row in table:
            if row["valid"] and row["raw_answer"]:
                case = answers_by_case.setdefault(row["case_id"], {})
                answer = row["raw_answer"].strip()
                case[answer] = case.get(answer, 0) + 1

    if verbose:
        total = sum(len(v) for v in answers_by_case.values())
        print(f"grading {total:,} distinct answers over {len(answers_by_case)} cases "
              f"from {len(runs)} runs", flush=True)

    cases = [c for c in sorted(answers_by_case) if c in records]
    contexts = {c: case_text(records[c].get("question", "")) for c in cases}

    # Every stage batches across cases. Bucketing one case at a time would leave the request
    # concurrency idle and turn a quarter-hour into most of an afternoon.
    if verbose:
        print(f"  bucketing {len(cases)} cases", flush=True)
    canonical = _collapse(
        cases,
        contexts,
        {case: list(answers_by_case[case]) for case in cases},
        llm_cfg=llm_cfg,
        verbose=verbose,
    )

    entities = {c: list(dict.fromkeys(canonical[c].values())) for c in cases}
    if verbose:
        print(f"  {sum(len(answers_by_case[c]) for c in cases):,} answers -> "
              f"{sum(len(entities[c]) for c in cases):,} entities; grading", flush=True)

    described = llm.map_structured(
        [{"case": contexts[c], "diagnosis": d}
         for c in cases for d in [records[c].get("gold_answer", ""), *entities[c]]],
        system=load_prompt(DESCRIBE_PROMPT),
        schema=Description,
        cfg=llm_cfg,
    )
    offsets, cursor = {}, 0
    for case in cases:
        offsets[case] = cursor
        cursor += 1 + len(entities[case])

    pairs = [
        (case, entity, described[offsets[case]], described[offsets[case] + 1 + i])
        for case in cases
        for i, entity in enumerate(entities[case])
        if described[offsets[case]] is not None and described[offsets[case] + 1 + i] is not None
    ]
    compared = llm.map_structured(
        [
            {"case": contexts[c], "predicted_diagnosis": pred.description,
             "true_diagnosis": true.description}
            for c, _, true, pred in pairs
        ],
        system=load_prompt(COMPARE_PROMPT),
        schema=Similarity,
        cfg=llm_cfg,
    )
    scores: dict[tuple[str, str], float] = {
        (case, entity): reply.score
        for (case, entity, _, _), reply in zip(pairs, compared)
        if reply is not None
    }

    rows = []
    for case in cases:
        for answer, count in answers_by_case[case].items():
            name = canonical[case].get(answer, answer)
            score = scores.get((case, name))
            rows.append(
                {
                    "case_id": case,
                    "raw_answer": answer,
                    "canonical": name,
                    "similarity": score,
                    "is_correct": None if score is None else score >= CORRECT_AT,
                    "n_rollouts": count,
                }
            )

    out = out or store.run_dir(runs[0]) / "answer_grades.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=GRADE_SCHEMA), temporary, compression="zstd")
    temporary.replace(out)
    if verbose:
        graded = sum(r["is_correct"] is not None for r in rows)
        entities = len({(r["case_id"], r["canonical"]) for r in rows})
        print(f"{len(rows):,} answers -> {entities:,} entities, {graded:,} graded -> {out.name}",
              flush=True)
    return out


def load_grades(path: Path) -> dict[tuple[str, str], dict]:
    """Grades keyed by (case_id, raw_answer)."""
    return {(r["case_id"], r["raw_answer"]): r for r in pq.read_table(path).to_pylist()}
