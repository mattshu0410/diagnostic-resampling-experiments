"""Grading a model's final diagnosis against the case's gold answer.

The two diagnoses are never compared as strings. Each is described in the context of the case,
then the two descriptions are rated for similarity from 0 to 10, and a response counts as
correct at `CORRECT_AT` or above. A response whose answer cannot be read scores 0 without any
model call; a record the judge never scored raises rather than being written as one.

Reads the records `generate.responses` writes and adds `extracted_answer`, `similarity_score`,
`is_correct`, `_true_description` and `_predicted_description` to each.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from dctax.config import ModelConfig, data_root, load_prompt
from dctax.generate.responses import responses_path, write
from dctax.utils import answers, llm

CORRECT_AT = 8.0
DESCRIBE_PROMPT = "describe_diagnosis"
COMPARE_PROMPT = "compare_diagnoses"

CASE_BLOCK = re.compile(
    r"-{40}\nCASE PRESENTATION\n-{40}\n(.*?)\n-{40}\nOUTPUT TEMPLATE\n-{40}",
    re.DOTALL,
)


class Description(BaseModel):
    """One diagnosis described in the context of its case."""

    description: str


class Similarity(BaseModel):
    """How close a predicted diagnosis is to the true one."""

    score: float = Field(ge=0.0, le=10.0)


def graded_path(slug: str) -> Path:
    """Where one model's graded responses are written."""
    return data_root() / "responses" / f"responses_{slug}.graded.json"


def case_text(question: str) -> str:
    """The case presentation out of the prompt the model was given."""
    found = CASE_BLOCK.search(question or "")
    return found.group(1).strip() if found else ""


def grade(
    records: list[dict],
    cfg: ModelConfig,
    *,
    llm_cfg: llm.LLMConfig | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Grade every record, describing both diagnoses before rating their similarity."""
    read = [answers.extract(r.get("full_response", ""), cfg.answer_fallbacks) for r in records]
    cases = [case_text(r.get("question", "")) for r in records]
    gradable = [i for i, answer in enumerate(read) if answer]
    if verbose:
        print(f"{cfg.slug}: grading {len(gradable)} of {len(records)}, "
              f"{len(records) - len(gradable)} with no readable answer", flush=True)

    # Both descriptions share a prompt, so they go out as one batch of 2n payloads.
    payloads = [{"case": cases[i], "diagnosis": records[i].get("gold_answer", "")} for i in gradable]
    payloads += [{"case": cases[i], "diagnosis": read[i]} for i in gradable]
    described = llm.map_structured(
        payloads, system=load_prompt(DESCRIBE_PROMPT), schema=Description, cfg=llm_cfg
    )
    true_side, predicted_side = described[: len(gradable)], described[len(gradable):]

    scored: list[Similarity | None] = [None] * len(gradable)
    pairs = [i for i in range(len(gradable)) if true_side[i] and predicted_side[i]]
    if pairs:
        replies = llm.map_structured(
            [
                {
                    "case": cases[gradable[i]],
                    "predicted_diagnosis": predicted_side[i].description,
                    "true_diagnosis": true_side[i].description,
                }
                for i in pairs
            ],
            system=load_prompt(COMPARE_PROMPT),
            schema=Similarity,
            cfg=llm_cfg,
        )
        for i, reply in zip(pairs, replies):
            scored[i] = reply

    # A judge that never returned would otherwise be indistinguishable from a wrong answer.
    unscored = [records[gradable[i]].get("pmcid", "?") for i in range(len(gradable))
                if scored[i] is None]
    if unscored:
        raise RuntimeError(
            f"{cfg.slug}: {len(unscored)} of {len(gradable)} records went unscored, "
            f"first few {unscored[:5]}. Rerun to retry them; cached judgements are reused."
        )

    graded = [dict(record) for record in records]
    for slot, index in enumerate(gradable):
        graded[index]["_true_description"] = (
            true_side[slot].description if true_side[slot] else ""
        )
        graded[index]["_predicted_description"] = (
            predicted_side[slot].description if predicted_side[slot] else ""
        )
        if scored[slot] is not None:
            graded[index]["similarity_score"] = scored[slot].score
    for index, answer in enumerate(read):
        # Only an unreadable answer reaches the output as a zero.
        graded[index]["extracted_answer"] = answer
        graded[index].setdefault("similarity_score", 0.0)
        graded[index]["is_correct"] = graded[index]["similarity_score"] >= CORRECT_AT

    if verbose:
        right = sum(1 for r in graded if r["is_correct"])
        print(f"  {right}/{len(graded)} correct at similarity >= {CORRECT_AT}", flush=True)
    return graded


def run(
    cfg: ModelConfig,
    *,
    path: Path | None = None,
    out: Path | None = None,
    llm_cfg: llm.LLMConfig | None = None,
    verbose: bool = True,
) -> tuple[list[dict], Path]:
    """Grade one model's stored responses and write them alongside."""
    path = path or responses_path(cfg.slug)
    with path.open() as handle:
        records = json.load(handle)
    graded = grade(records, cfg, llm_cfg=llm_cfg, verbose=verbose)
    out = out or graded_path(cfg.slug)
    write(out, graded)
    return graded, out
