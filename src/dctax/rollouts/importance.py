"""Sentence importance from boundary-conditioned rollouts.

Scoring runs in two stages. `score_rollouts` extracts an answer and a replacement similarity
for every rollout once, since that is the expensive part; `importance_table` then derives the
per-sentence metrics from those, so a different threshold or divergence costs no embedding.

Every metric is the effect of KEEPING a sentence relative to removing it, and the answer
counts both sides were computed from are stored so any other convention stays recoverable.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dctax.config import ModelConfig, load_model
from dctax.rollouts import store
from dctax.rollouts.schema import BaseTrace, Rollout
from dctax.rollouts.semantic import SemanticMatcher
from dctax.utils import answers, traces

#: An opening longer than this is not a sentence: the splitter has been handed a generation that
#: never terminated. Only applied to rollouts that also hit the token cap.
OVERSIZED_CHARS = 8000

THRESHOLD = 0.8
SMOOTHING = 0.5
HEAD_CHARS = 2000
MATCH_N = 16
MATCH_N_SMALL = 10
ANCHOR_KEEP_MAX = 3.0
ANCHOR_RATIO = 2.0

_FILLER = re.compile(r"\b(the|a|an|likely|most|diagnosis|of|is|with|due|to)\b")
_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation and filler so trivial rewordings share a bucket."""
    return " ".join(_FILLER.sub(" ", _PUNCT.sub(" ", text.lower())).split())


class Scorer(Protocol):
    """Reads a continuation's answer, the bucket it belongs to, and any verdict on it."""

    def extract(self, continuation: str) -> str: ...

    def bucket(self, continuation: str) -> str | None: ...

    def is_correct(self, continuation: str) -> bool | None: ...


@dataclass
class NormalisedStringScorer:
    """Lexical bucketing, until a clinical grader replaces it.

    It splits clinically equivalent phrasings, so distinct buckets are an upper bound on
    genuine disagreement and every divergence derived from them is an upper bound too.
    """

    cfg: ModelConfig

    def extract(self, continuation: str) -> str:
        return answers.extract(continuation, self.cfg.answer_fallbacks)

    def bucket(self, continuation: str) -> str | None:
        return normalise_answer(self.extract(continuation)) or None

    def is_correct(self, continuation: str) -> bool | None:
        return None


def _split_head(payload: tuple[str, str]) -> tuple[str, int]:
    """First sentence of a head and how many the head yielded. Runs in a worker process."""
    head, rules = payload
    parts = traces.split_into_sentences(head, rules)
    return (parts[0] if parts else ""), len(parts)


def _first_sentences(continuations: Sequence[str], rules: str) -> list[str]:
    """first_sentence over many continuations, splitting heads across processes.

    Only the head crosses the process boundary; the rare continuation whose head yields no
    boundary is re-split whole in this process.
    """
    heads = [(c.strip()[:HEAD_CHARS], rules) for c in continuations]
    with ProcessPoolExecutor() as pool:
        split = list(pool.map(_split_head, heads, chunksize=64))

    out = []
    for (first, count), continuation in zip(split, continuations):
        text = continuation.strip()
        if count < 2 and len(text) > HEAD_CHARS:
            first = first_sentence(text, rules=rules)
        out.append(first)
    return out


def first_sentence(continuation: str, head: int = HEAD_CHARS,
                   rules: str = "clinical") -> str:
    """The sentence a rollout opens with, which is its replacement for the removed one.

    Only the opening of the continuation is split, since splitting tens of thousands of full
    continuations costs far more than it returns. A head that yields no boundary is re-split
    in full, so a sentence longer than the window is still returned whole.
    """
    text = continuation.strip()
    if not text:
        return ""
    parts = traces.split_into_sentences(text[:head], rules)
    if len(parts) < 2 and len(text) > head:
        parts = traces.split_into_sentences(text, rules)
    return parts[0] if parts else ""


def _distribution(counts: dict[str, int], support: Sequence[str], alpha: float) -> dict[str, float]:
    total = sum(counts.get(a, 0) + alpha for a in support)
    return {a: (counts.get(a, 0) + alpha) / total for a in support}


def smoothed_kl(p_counts, q_counts, alpha: float = SMOOTHING,
                vocab: Sequence[str] | None = None) -> float | None:
    """KL(P || Q) in bits, additively smoothed over the support.

    Support defaults to the union of both arms, which makes the smoothing mass alpha*K vary with
    how many answers happened to appear at this boundary; the divergence then partly tracks K
    rather than the shift between the arms. Passing `vocab` fixes the support - the case's whole
    answer vocabulary - so K is constant across a trace and sentences become comparable.
    """
    support = sorted(set(vocab) | set(p_counts) | set(q_counts)) if vocab is not None \
        else sorted(set(p_counts) | set(q_counts))
    if not support or not sum(p_counts.values()) or not sum(q_counts.values()):
        return None
    p, q = _distribution(p_counts, support, alpha), _distribution(q_counts, support, alpha)
    return sum(p[a] * log2(p[a] / q[a]) for a in support)


def total_variation(p_counts, q_counts) -> float | None:
    """Total variation distance, half the L1 distance between the two distributions.

    Bounded in [0, 1] and invariant to the ambient support: an entity neither arm produced
    leaves it unchanged, so two arms realising different answer sets are directly comparable
    with no smoothing constant and no shared vocabulary to fix.
    """
    support = set(p_counts) | set(q_counts)
    total_p, total_q = sum(p_counts.values()), sum(q_counts.values())
    if not support or not total_p or not total_q:
        return None
    return 0.5 * sum(
        abs(p_counts.get(a, 0) / total_p - q_counts.get(a, 0) / total_q) for a in support
    )


def jensen_shannon(p_counts, q_counts) -> float | None:
    """JSD in bits, bounded in [0, 1]. Needs no smoothing: the midpoint is never zero."""
    support = sorted(set(p_counts) | set(q_counts))
    if not support or not sum(p_counts.values()) or not sum(q_counts.values()):
        return None
    p, q = _distribution(p_counts, support, 0.0), _distribution(q_counts, support, 0.0)
    mid = {a: (p[a] + q[a]) / 2 for a in support}
    half = lambda d: sum(d[a] * log2(d[a] / mid[a]) for a in support if d[a] > 0)  # noqa: E731
    return 0.5 * half(p) + 0.5 * half(q)


def effective_answers(counts: dict[str, int]) -> float | None:
    """2^H over an answer distribution."""
    total = sum(counts.values())
    if not total:
        return None
    return 2 ** -sum((n / total) * log2(n / total) for n in counts.values() if n)


def _matched_arms(
    keep: Sequence[dict],
    counterfactual: Sequence[dict],
    grades: Grades,
    case_id: str,
    index: int,
    n: int = MATCH_N,
    vocab: Sequence[str] | None = None,
) -> dict:
    """Both arms subsampled to n rollouts.

    2^H estimated from k samples cannot exceed k, so a concentrated-keep test passes more often
    at small k. Arms of unequal size are therefore not comparable; these columns fix both at n.
    Seeded on (case_id, index), so the draw is reproducible and independent of row order.

    Two KLs are returned. matched_kl_cf_keep smooths over the answers the two arms happened to
    produce here; matched_kl_cf_keep_fixed smooths over every answer the case produced anywhere.
    Matching the arms removes the sample-size bias but not the support bias - a boundary where
    the arms span four answers and one where they span twenty still smooth differently - so only
    the second is comparable across sentences on both counts. The first is kept beside it to
    show how much the support alone moved things.
    """
    empty = dict.fromkeys(
        ("matched_n", "matched_effective_keep", "matched_effective_counterfactual",
         "matched_collapse_ratio", "matched_is_anchor", "matched_kl_cf_keep",
         "matched_kl_cf_keep_fixed"))
    keep_entities = [e for e in (_entity(r, grades) for r in keep if r["valid"]) if e]
    cf_entities = [e for e in (_entity(r, grades) for r in counterfactual if r["valid"]) if e]
    if len(keep_entities) < n or len(cf_entities) < n:
        return empty

    seed = int(hashlib.sha256(f"{case_id}:{index}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    keep_counts = Counter(rng.sample(keep_entities, n))
    cf_counts = Counter(rng.sample(cf_entities, n))
    keep_effective = effective_answers(keep_counts)
    cf_effective = effective_answers(cf_counts)
    ratio = cf_effective / keep_effective if keep_effective else None
    return {
        "matched_n": n,
        "matched_effective_keep": keep_effective,
        "matched_effective_counterfactual": cf_effective,
        "matched_collapse_ratio": ratio,
        "matched_is_anchor": bool(
            ratio is not None and keep_effective <= ANCHOR_KEEP_MAX and ratio >= ANCHOR_RATIO
        ),
        "matched_kl_cf_keep": smoothed_kl(cf_counts, keep_counts),
        "matched_kl_cf_keep_fixed": smoothed_kl(cf_counts, keep_counts, vocab=vocab),
    }


SCORE_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("trace_id", pa.string()),
        ("boundary_index", pa.int32()),
        ("rollout_index", pa.int32()),
        ("raw_answer", pa.string()),
        ("bucket", pa.string()),
        ("is_correct", pa.bool_()),
        ("first_sentence", pa.string()),
        ("similarity", pa.float32()),
        ("finish_reason", pa.string()),
        ("valid", pa.bool_()),
    ]
)


def score_rollouts(
    run: str, scorer: Scorer, matcher: SemanticMatcher, *, verbose: bool = True
) -> Path:
    """Extract answer and replacement similarity for every rollout, once."""
    traces_by_id = {t.trace_id: t for t in store.load_traces(run)}
    cases = sorted({p.parent.name.split("=", 1)[1] for p in store.part_files(run)})
    # Replacements must be split the same way the originals were.
    rules = load_model(next(iter(traces_by_id.values())).model).sentence_rules \
        if traces_by_id else "clinical"
    if verbose:
        print(f"  scoring {len(cases)} cases from {len(traces_by_id)} traces", flush=True)

    rows: list[dict] = []
    withheld_total = 0
    for position, case in enumerate(cases, start=1):
        rollouts = _case_rollouts(run, case)
        if not rollouts:
            continue
        rows_here, withheld = _score_case(rollouts, traces_by_id, rules, scorer, matcher)
        rows.extend(rows_here)
        withheld_total += withheld
        if verbose and (position % 10 == 0 or position == len(cases)):
            print(f"    {position}/{len(cases)} cases, {len(rows):,} rollouts scored", flush=True)
    if verbose:
        print(f"  embedded and compared {len(rows):,} pairs"
              + (f", {withheld_total} withheld" if withheld_total else ""), flush=True)
    return _write_scores(run, rows)


def _case_rollouts(run: str, case: str) -> list:
    """One case's rollouts, without the token ids the scorer never reads."""
    from dctax.rollouts.schema import Rollout

    out = []
    for path in sorted((store.run_dir(run) / "rollouts" / f"case_id={case}").glob("part-*.parquet")):
        table = pq.read_table(path, columns=[
            "case_id", "trace_id", "boundary_index", "rollout_index", "prefix_token_count",
            "prefix_sha256", "continuation", "finish_reason", "seed"])
        for row in table.to_pylist():
            out.append(Rollout(output_token_ids=(), **row))
    return out


def _score_case(rollouts, traces_by_id, rules, scorer, matcher):
    """Rows for one case, and how many rollouts were withheld from similarity."""
    originals = []
    for rollout in rollouts:
        trace = traces_by_id[rollout.trace_id]
        # The final boundary removes nothing, so it has no sentence to compare against.
        removed = (
            trace.sentences[rollout.boundary_index]
            if rollout.boundary_index < len(trace.sentences)
            else None
        )
        originals.append(removed.text if removed else "")

    firsts = _first_sentences([r.continuation for r in rollouts], rules)

    # A rollout that ran to the token cap has no answer and never reaches either arm, and its
    # "first sentence" is usually a repetition loop tens of thousands of tokens long, which the
    # embedding endpoint rejects outright and which would fail the whole batch it sits in. Its
    # similarity is withheld rather than computed, and every one is named so the exclusion is
    # visible rather than inferred from a gap.
    degenerate = [
        i for i, rollout in enumerate(rollouts)
        if rollout.finish_reason == "length" and len(firsts[i] or "") > OVERSIZED_CHARS
    ]
    if degenerate:
        print(f"  {len(degenerate)} degenerate rollouts withheld from similarity:", flush=True)
        for i in degenerate:
            rollout = rollouts[i]
            print(f"    {rollout.case_id[:12]} boundary {rollout.boundary_index} "
                  f"rollout {rollout.rollout_index}: {len(firsts[i]):,} chars, "
                  f"finish_reason={rollout.finish_reason}", flush=True)
    withheld = set(degenerate)
    keep = [i for i in range(len(rollouts)) if i not in withheld]

    similarity = np.full(len(rollouts), np.nan, dtype=np.float32)
    if keep:
        similarity[keep] = matcher.pairwise([originals[i] for i in keep],
                                            [firsts[i] for i in keep])
    rows = []
    for rollout, first, sim in zip(rollouts, firsts, similarity):
        raw = scorer.extract(rollout.continuation)
        bucket = scorer.bucket(rollout.continuation)
        rows.append(
            {
                "case_id": rollout.case_id,
                "trace_id": rollout.trace_id,
                "boundary_index": rollout.boundary_index,
                "rollout_index": rollout.rollout_index,
                "raw_answer": raw or None,
                "bucket": bucket,
                "is_correct": scorer.is_correct(rollout.continuation),
                "first_sentence": first or None,
                "similarity": None if sim != sim else float(sim),
                "finish_reason": rollout.finish_reason,
                "valid": bucket is not None and rollout.finish_reason == "stop",
            }
        )
    return rows, len(degenerate)


def _write_scores(run: str, rows: list[dict]) -> Path:
    path = store.run_dir(run) / "rollout_scores.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCORE_SCHEMA), temporary, compression="zstd")
    temporary.replace(path)
    return path


def load_scores(run: str) -> list[dict]:
    return pq.read_table(store.run_dir(run) / "rollout_scores.parquet").to_pylist()


Grades = dict[tuple[str, str], dict]


def _entity(row: dict, grades: Grades) -> str | None:
    """The judged entity a row's answer belongs to.

    There is deliberately no lexical fallback. Matching answer strings splits one diagnosis
    across every phrasing of it — 22 buckets where the judge finds 2 — which inflates the
    entropy of the distribution and every divergence taken from it, while looking perfectly
    healthy in the output.
    """
    verdict = grades.get((row["case_id"], (row["raw_answer"] or "").strip()))
    return verdict["canonical"] if verdict else None


def _counts(rows: Sequence[dict], grades: Grades) -> dict[str, int]:
    entities = [_entity(r, grades) for r in rows if r["valid"]]
    return dict(Counter(e for e in entities if e))


def _accuracy(rows: Sequence[dict], grades: Grades) -> float | None:
    judged = []
    for row in rows:
        if not row["valid"]:
            continue
        verdict = grades.get((row["case_id"], (row["raw_answer"] or "").strip()))
        if verdict is not None and verdict["is_correct"] is not None:
            judged.append(bool(verdict["is_correct"]))
    return sum(judged) / len(judged) if judged else None


def require_graded(rows: Sequence[dict], grades: Grades) -> None:
    """Refuse to score when answers the judge never returned would silently vanish."""
    missing = {
        (r["case_id"], (r["raw_answer"] or "").strip())
        for r in rows
        if r["valid"] and (r["case_id"], (r["raw_answer"] or "").strip()) not in grades
    }
    if missing:
        raise RuntimeError(
            f"{len(missing)} distinct answers have no judgement, first few "
            f"{sorted(missing)[:3]}. Rerun grading; cached judgements are reused."
        )


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def score_sentence(
    trace: BaseTrace,
    index: int,
    by_boundary: dict[int, list[dict]],
    grades: Grades,
    *,
    threshold: float = THRESHOLD,
    vocab: Sequence[str] | None = None,
) -> dict:
    """Metrics for one sentence, as the effect of keeping it rather than removing it."""
    remove = by_boundary.get(index, [])
    keep = by_boundary.get(index + 1, [])
    scored = [r for r in remove if r["similarity"] is not None]
    counterfactual = [r for r in scored if r["similarity"] < threshold]
    # The other side of the same boundary: rollouts that regenerated a sentence close enough
    # to the original to count as continuing it. Both arms are drawn from the same boundary,
    # so they differ only in what the model wrote in place of the removed sentence.
    similar = [r for r in scored if r["similarity"] >= threshold]

    remove_counts, keep_counts = _counts(remove, grades), _counts(keep, grades)
    cf_counts = _counts(counterfactual, grades)
    similar_counts = _counts(similar, grades)
    replacements = [r["first_sentence"] for r in scored if r["first_sentence"]]
    sentence = trace.sentences[index]

    # Levels, not just their difference: a delta cannot be un-subtracted, and convergence is
    # defined on the level. Stored so the convergence cut can be varied without rescoring.
    remove_accuracy = _accuracy(remove, grades)
    keep_accuracy = _accuracy(keep, grades)
    counterfactual_accuracy = _accuracy(counterfactual, grades)

    return {
        "model": trace.model,
        "case_id": trace.case_id,
        "sentence_index": index,
        "sentence_text": sentence.text,
        "sentence_tokens": sentence.token_end - sentence.token_start,
        "position": index / len(trace.sentences),
        "n_sentences": len(trace.sentences),
        "base_is_correct": trace.is_correct,
        "remove_n": len(remove),
        "keep_n": len(keep),
        "counterfactual_n": len(counterfactual),
        "similar_n": len(similar),
        "remove_invalid": sum(not r["valid"] for r in remove),
        "keep_invalid": sum(not r["valid"] for r in keep),
        "remove_accuracy": remove_accuracy,
        "keep_accuracy": keep_accuracy,
        "counterfactual_accuracy": counterfactual_accuracy,
        "resampling_accuracy_delta": _delta(keep_accuracy, remove_accuracy),
        "counterfactual_accuracy_delta": _delta(keep_accuracy, counterfactual_accuracy),
        "resampling_kl_keep_remove": smoothed_kl(keep_counts, remove_counts),
        "resampling_kl_remove_keep": smoothed_kl(remove_counts, keep_counts),
        "resampling_jsd": jensen_shannon(keep_counts, remove_counts),
        "counterfactual_kl_keep_cf": smoothed_kl(keep_counts, cf_counts),
        "counterfactual_kl_cf_keep": smoothed_kl(cf_counts, keep_counts),
        "counterfactual_jsd": jensen_shannon(keep_counts, cf_counts),
        # Same divergences over the case's fixed answer vocabulary, so the smoothing mass does
        # not change with the number of answers seen at this boundary. JSD is unsmoothed and so
        # is unaffected by the support, and is not recomputed.
        "counterfactual_kl_cf_keep_fixed": smoothed_kl(cf_counts, keep_counts, vocab=vocab),
        "counterfactual_kl_keep_cf_fixed": smoothed_kl(keep_counts, cf_counts, vocab=vocab),
        "resampling_kl_remove_keep_fixed": smoothed_kl(remove_counts, keep_counts, vocab=vocab),
        "case_vocab_size": len(vocab) if vocab is not None else None,
        # Within one boundary: how far the answers diverge when the model replaces the
        # sentence with something different, against when it regenerates something close.
        "counterfactual_tvd": total_variation(similar_counts, cf_counts),
        "alternative_trajectory_fraction": len(counterfactual) / len(scored) if scored else None,
        "unique_replacement_fraction": (
            len(set(replacements)) / len(replacements) if replacements else None
        ),
        "mean_replacement_similarity": (
            sum(r["similarity"] for r in scored) / len(scored) if scored else None
        ),
        **_matched_arms(keep, counterfactual, grades, trace.case_id, index, vocab=vocab),
        **{f"{k}10": v for k, v in _matched_arms(
            keep, counterfactual, grades, trace.case_id, index,
            n=MATCH_N_SMALL, vocab=vocab).items()},
        "keep_counts": json.dumps(keep_counts, sort_keys=True),
        "remove_counts": json.dumps(remove_counts, sort_keys=True),
        "counterfactual_counts": json.dumps(cf_counts, sort_keys=True),
        "similar_counts": json.dumps(similar_counts, sort_keys=True),
        "threshold": threshold,
    }


def median_similarity(scores: Sequence[dict]) -> float:
    """Median replacement similarity over a run's scored rollouts."""
    values = sorted(r["similarity"] for r in scores if r["similarity"] is not None)
    if not values:
        raise ValueError("no scored rollouts to take a median similarity from")
    return values[len(values) // 2]


def importance_table(
    run: str, grades: Grades, *, threshold: float | None = None, verbose: bool = True
) -> Path:
    """One row per sentence, joinable to assignments.csv on (model, case_id, sentence_index).

    Answer distributions are taken over the judge's entities, so grading must have run over
    every answer first.

    `threshold` splits each boundary's rollouts into those that regenerated something close
    to the removed sentence and those that did not. Passing None takes the run's own median
    replacement similarity, which is what the models differ on: a fixed cut puts 95% of one
    model's rollouts in the counterfactual arm and 35% of another's, so the arms are not
    comparable across models. The value used is written into every row.
    """
    traces_by_id = {t.trace_id: t for t in store.load_traces(run)}
    scores = load_scores(run)
    require_graded(scores, grades)

    dynamic = threshold is None
    if dynamic:
        threshold = median_similarity(scores)
    if verbose:
        print(f"  similarity cut {threshold:.4f}"
              + (" (this run's median)" if dynamic else " (fixed)"), flush=True)

    grouped: dict[str, dict[int, list[dict]]] = {}
    for row in scores:
        grouped.setdefault(row["trace_id"], {}).setdefault(row["boundary_index"], []).append(row)

    # Rollouts scoring withheld a similarity for cannot enter the counterfactual arm, and a
    # boundary that loses several is measured over fewer samples than its neighbours. Reported
    # per boundary rather than as a total, so a boundary that lost most of its arm is visible.
    missing: dict[tuple[str, int], int] = {}
    for trace_id, by_boundary in grouped.items():
        for boundary, rows_at in by_boundary.items():
            n = sum(1 for r in rows_at if r["similarity"] is None)
            if n:
                missing[(trace_id, boundary)] = n
    if missing and verbose:
        total = sum(missing.values())
        print(f"  {total} rollouts have no similarity, over {len(missing)} boundaries; "
              f"those boundaries measure the counterfactual arm on fewer samples", flush=True)
        for (trace_id, boundary), n in sorted(missing.items(), key=lambda kv: -kv[1])[:10]:
            case = next((t.case_id for t in traces_by_id.values()
                         if t.trace_id == trace_id), trace_id[:12])
            print(f"    {case[:12]} boundary {boundary}: {n} of "
                  f"{len(grouped[trace_id][boundary])} rollouts", flush=True)

    rows = []
    for trace_id, by_boundary in grouped.items():
        trace = traces_by_id[trace_id]
        # Every canonical answer this trace produced anywhere, so the KL support is the same at
        # every boundary of the trace.
        vocab = sorted({a for rs in by_boundary.values() for a in _counts(rs, grades)})
        rows.extend(
            score_sentence(trace, i, by_boundary, grades, threshold=threshold, vocab=vocab)
            for i in range(len(trace.sentences))
        )

    name = "importance_tmedian" if dynamic else f"importance_t{threshold}"
    path = store.run_dir(run) / f"{name}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)
    if verbose:
        print(f"  {len(rows):,} sentences -> {path.name}", flush=True)
    return path
