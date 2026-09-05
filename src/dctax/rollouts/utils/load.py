"""Reading a finished resampling run into the shapes an analysis wants.

An analysis needs three things from a run: the judge's entity for every answer, a light
description of each base trace, and the answers produced at each boundary. None of them are
what the store hands back. `store.load_traces` rebuilds full `BaseTrace` objects including
`input_ids`, which is hundreds of megabytes across a roster and which no analysis indexes
into; `rollout_scores.parquet` is per rollout, while the unit of analysis is the boundary.

So the loaders here return summaries rather than records, and fold the two filters every
analysis applies - a rollout must be valid, and its answer must resolve to a judged entity -
into `boundary_entities`, so no analysis has to remember to apply them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from dctax.config import data_root
from dctax.rollouts import store

#: (case_id, raw answer) -> the canonical entity the judge grouped that answer into.
EntityLookup = dict[tuple[str, str], str]


@dataclass(frozen=True)
class TraceSummary:
    """One base trace, with its boundaries located in the token sequence.

    `boundary_tokens[b]` is the token a rollout at boundary b generates from. There are
    n_sentences + 1 of them: boundary b cuts before sentence b, and the final boundary holds
    every sentence and so begins where the last one ends.
    """

    trace_id: str
    case_id: str
    model: str
    base_is_correct: bool | None
    raw_answer: str
    prompt_token_count: int
    n_sentences: int
    total_tokens: int
    boundary_tokens: tuple[int, ...]
    sentences: tuple[str, ...]

    @property
    def n_boundaries(self) -> int:
        return self.n_sentences + 1

    @property
    def reasoning_start(self) -> int:
        """First generated token: every prefix keeps the prompt, so depth is measured here."""
        return self.prompt_token_count

    @property
    def reasoning_end(self) -> int:
        return self.boundary_tokens[-1]

    @property
    def reasoning_tokens(self) -> int:
        return self.reasoning_end - self.reasoning_start

    def token_fraction(self, boundary_index: int) -> float:
        """How deep a boundary sits in the generated reasoning: 0 at its start, 1 at its end.

        Normalised over the generation rather than the whole sequence, so a model with a long
        rendered prompt is not reported as converging earlier than one with a short prompt.
        """
        return (
            self.boundary_tokens[boundary_index] - self.reasoning_start
        ) / self.reasoning_tokens


def grades_path(experiment: str) -> Path:
    """Where `grade_run` writes the shared judgements for an experiment."""
    return data_root() / "rollouts" / f"grades_{experiment}.parquet"


def entity_lookup(experiment: str | Path) -> EntityLookup:
    """Canonical entity per (case_id, raw answer), for answers the judge grouped.

    An answer the judge never returned an entity for is absent rather than mapped to itself:
    a missing entity means the answer was never placed, and treating it as its own entity
    would add a spurious category to every distribution taken over these.
    """
    path = experiment if isinstance(experiment, Path) else grades_path(experiment)
    return {
        (row["case_id"], row["raw_answer"]): row["canonical"]
        for row in pq.read_table(
            path, columns=["case_id", "raw_answer", "canonical"]
        ).to_pylist()
        if row["canonical"]
    }


def trace_summaries(run: str) -> list[TraceSummary]:
    """Every base trace of a run, without the token ids."""
    out: list[TraceSummary] = []
    with (store.run_dir(run) / "traces.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            sentences = payload["sentences"]
            out.append(
                TraceSummary(
                    trace_id=payload["trace_id"],
                    case_id=payload["case_id"],
                    model=payload["model"],
                    base_is_correct=payload.get("is_correct"),
                    raw_answer=(payload.get("raw_answer") or "").strip(),
                    prompt_token_count=payload["prompt_token_count"],
                    n_sentences=len(sentences),
                    total_tokens=len(payload["input_ids"]),
                    boundary_tokens=tuple(
                        [s["token_start"] for s in sentences] + [sentences[-1]["token_end"]]
                    ),
                    sentences=tuple(s["text"] for s in sentences),
                )
            )
    return out


def boundary_answers(run: str) -> dict[tuple[str, int], list[str]]:
    """Raw answers of the valid rollouts at each (trace_id, boundary_index)."""
    table = pq.read_table(
        store.run_dir(run) / "rollout_scores.parquet",
        columns=["trace_id", "boundary_index", "raw_answer", "valid"],
    )
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for trace_id, boundary, answer, valid in zip(
        table.column("trace_id").to_pylist(),
        table.column("boundary_index").to_pylist(),
        table.column("raw_answer").to_pylist(),
        table.column("valid").to_pylist(),
    ):
        if valid and answer:
            out[(trace_id, int(boundary))].append(answer.strip())
    return out


def boundary_entities(
    run: str, entities: EntityLookup, cases: dict[str, str] | None = None
) -> dict[tuple[str, int], list[str]]:
    """Judged entities of the graded rollouts at each (trace_id, boundary_index).

    A rollout reaches this only if it finished cleanly with a readable answer and that answer
    was placed in an entity, which is what "successfully graded" means everywhere downstream.
    A boundary with no such rollout is absent rather than empty, so a caller that indexes it
    has to decide what a missing boundary means instead of silently reading zero.

    `cases` maps trace_id to case_id; it is derived from the run's traces when not given.
    """
    if cases is None:
        cases = {t.trace_id: t.case_id for t in trace_summaries(run)}
    out: dict[tuple[str, int], list[str]] = {}
    for (trace_id, boundary), answers in boundary_answers(run).items():
        case_id = cases.get(trace_id)
        if case_id is None:
            continue
        found = [entities[(case_id, a)] for a in answers if (case_id, a) in entities]
        if found:
            out[(trace_id, boundary)] = found
    return out
