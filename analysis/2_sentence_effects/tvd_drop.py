"""How much each sentence moves the trace toward the distribution it settles on.

`analysis/1_convergence/tvd.py` tracks one curve per trace: at boundary i, the worst total
variation distance between that boundary's answer distribution and any later boundary's.
It starts high, because early on the trace still disagrees with where it ends up, and falls
to nothing once the trace has committed.

Sentence i sits between boundary i and boundary i + 1, so

    tvd_drop(i) = curve(i) - curve(i + 1)

is what writing that sentence did to the disagreement. Positive means the sentence moved the
trace toward its eventual distribution; negative means it opened the question back up. Unlike
`counterfactual_tvd`, which asks what would have happened had the sentence been written
differently, this is a property of the trace as it was actually written.

Two gates, matching the convergence analysis so both describe the same traces:

    the base trace's answer has to grade to an entity
    every boundary needs more than MIN_GRADED graded rollouts

and then two more, because a drop is only defined where there is a descent to attribute:

    traces that never commit are dropped, having no point to descend to
    traces already committed at boundary 0 are dropped, having no descent

    DCTAX_DATA_ROOT=/path/to/artifacts python analysis/2_sentence_effects/tvd_drop.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts.utils import (  # noqa: E402
    boundary_entities,
    entity_lookup,
    forward_curves,
    trace_summaries,
    tvd_matrix,
)

OUT = HERE / "out"

#: A boundary needs more than this many graded rollouts, as in `analysis/1_convergence`.
MIN_GRADED = 10

#: The trace has committed once its worst forward disagreement is inside this band.
BAND = 0.2


def commitment(curve: list[float], band: float = BAND) -> int | None:
    """The first boundary whose worst forward disagreement is inside the band.

    None when no boundary qualifies. `analysis/1_convergence/tvd.py` falls back to the last
    boundary so that every trace gets a depth; here the distinction matters, since a trace
    that never settles has no descent and is dropped rather than counted as a late one.
    """
    return next((i for i, v in enumerate(curve) if v <= band), None)


def analyse(run: str, entities: dict) -> tuple[list[dict], dict]:
    """One row per sentence of every trace that has a descent, plus the exclusion tally."""
    traces = trace_summaries(run)
    graded = boundary_entities(run, entities, cases={t.trace_id: t.case_id for t in traces})

    rows: list[dict] = []
    tally = {"traces": len(traces), "no_base_entity": 0, "thin_boundary": 0,
             "never_commits": 0, "commits_at_zero": 0, "no_descent": 0, "kept": 0}

    for t in traces:
        base_entity = entities.get((t.case_id, t.raw_answer)) if t.raw_answer else None
        if base_entity is None:
            tally["no_base_entity"] += 1
            continue

        arms = [graded.get((t.trace_id, b), []) for b in range(t.n_boundaries)]
        if any(len(a) <= MIN_GRADED for a in arms):
            tally["thin_boundary"] += 1
            continue

        vocabulary = sorted({e for arm in arms for e in arm} | {base_entity})
        curve, _ = forward_curves(tvd_matrix(arms, vocabulary))

        commit = commitment(curve)
        if commit is None:
            tally["never_commits"] += 1
            continue
        if commit == 0:
            tally["commits_at_zero"] += 1
            continue

        # Reported rather than gated on: the curve can end lower than it started without
        # having descended monotonically, and the per-sentence drops are still defined.
        descent = curve[0] - curve[commit]
        if descent <= 0:
            tally["no_descent"] += 1

        # The last sentence has no boundary after it to compare against.
        for i in range(len(curve) - 1):
            rows.append(
                {
                    "model": t.model,
                    "case_id": t.case_id,
                    "trace_id": t.trace_id,
                    "base_is_correct": t.base_is_correct,
                    "sentence_index": i,
                    "n_sentences": t.n_sentences,
                    "token_position": t.token_fraction(i),
                    "tvd_before": curve[i],
                    "tvd_after": curve[i + 1],
                    "tvd_drop": curve[i] - curve[i + 1],
                    "commit_boundary": commit,
                    "commit_position": t.token_fraction(commit),
                    "pre_commit": int(i < commit),
                    "descent": descent,
                }
            )
        tally["kept"] += 1

    return rows, tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    experiment = load_resample(args.config)
    entities = entity_lookup(experiment.name)
    print(f"{args.config}: {len(experiment.models)} models, artifacts at {data_root()}\n")

    rows: list[dict] = []
    header = (f'{"model":<30} {"traces":>7} {"no base":>8} {"thin":>6} '
              f'{"never":>6} {"at 0":>6} {"kept":>6}')
    print(header)
    for model in experiment.models:
        mine, tally = analyse(experiment.sweep_run(model), entities)
        rows.extend(mine)
        print(f'{model:<30} {tally["traces"]:>7} {tally["no_base_entity"]:>8} '
              f'{tally["thin_boundary"]:>6} {tally["never_commits"]:>6} '
              f'{tally["commits_at_zero"]:>6} {tally["kept"]:>6}', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tvd_drop.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    traces = {r["trace_id"] for r in rows}
    drops = [r["tvd_drop"] for r in rows]
    pre = [r["tvd_drop"] for r in rows if r["pre_commit"]]
    print(f"\n{len(rows):,} sentences over {len(traces)} traces")
    print(f"  mean drop            {sum(drops) / len(drops):>8.4f}")
    print(f"  mean drop pre-commit {sum(pre) / len(pre):>8.4f}  ({len(pre):,} sentences)")
    print(f"  positive drops       {100 * sum(d > 0 for d in drops) / len(drops):>7.1f}%")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
