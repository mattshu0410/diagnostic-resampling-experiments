"""How the commitment rule behaves as the band delta is varied.

`convergence.py` and `tvd.py` take a trace to have committed at the first boundary whose worst
forward total variation distance falls within `BAND` = 0.2. That value was carried over from
the proportion-based rule rather than derived, so this sweeps it and records, for each trace
and each delta, whether the trace commits, where, and what its answer distribution looks like
at that point.

Two kinds of column come out, and they answer different questions:

    depth                 where commitment lands, which is what delta moves
    top_share, second_share, effective_n   how concentrated the distribution is there

The second kind is not what delta controls. TVD measures how much a distribution *moves*
between one position and every later one; how many diagnoses are in play at a single position
is a separate property. Any relationship between them is a fact about these traces rather
than something the rule enforces, which is exactly why it is worth measuring.

    DCTAX_DATA_ROOT=/path/to/artifacts python analysis/1_convergence/delta_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from convergence import MIN_GRADED, OUT, write_csv  # noqa: E402
from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts.utils import (  # noqa: E402
    boundary_entities,
    entity_lookup,
    forward_curves,
    trace_summaries,
    tvd_matrix,
)

#: Bands to sweep, in the increments the appendix table reports.
DELTAS = tuple(round(0.05 * k, 2) for k in range(1, 11))


def concentration(arm: list[str]) -> dict:
    """How the rollouts at one boundary are spread over answers.

    `top_share` and `second_share` are the mass on the most and second most common answers,
    which say directly what a reader wants to know. `effective_n` is 2 raised to the Shannon
    entropy: the number of equally likely answers that would leave the model this uncertain,
    so 1.0 means one answer with all the mass and 2.0 means a genuine coin flip between two.
    It is reported because it summarises the whole tail in one number, where the two shares
    do not.
    """
    counts = np.array(sorted(Counter(arm).values(), reverse=True), dtype=float)
    p = counts / counts.sum()
    return {
        "n_distinct": len(p),
        "top_share": float(p[0]),
        "second_share": float(p[1]) if len(p) > 1 else 0.0,
        "top_two_share": float(p[:2].sum()),
        "n_over_10pct": int((p >= 0.10).sum()),
        "effective_n": float(2 ** -(p * np.log2(p)).sum()),
    }


def analyse(run: str, entities: dict) -> list[dict]:
    """One row per trace and delta, for every trace passing the same gates as `tvd.py`."""
    traces = trace_summaries(run)
    graded = boundary_entities(run, entities, cases={t.trace_id: t.case_id for t in traces})

    rows: list[dict] = []
    for t in traces:
        base = entities.get((t.case_id, t.raw_answer)) if t.raw_answer else None
        if base is None:
            continue
        arms = [graded.get((t.trace_id, b), []) for b in range(t.n_boundaries)]
        if any(len(a) <= MIN_GRADED for a in arms):
            continue

        vocabulary = sorted({e for arm in arms for e in arm} | {base})
        curve, _ = forward_curves(tvd_matrix(arms, vocabulary))

        for delta in DELTAS:
            commit = next((i for i, v in enumerate(curve) if v <= delta), None)
            row = {
                "model": t.model, "case_id": t.case_id, "trace_id": t.trace_id,
                "base_is_correct": t.base_is_correct, "n_sentences": t.n_sentences,
                "delta": delta, "commits": int(commit is not None),
                "commit_boundary": commit if commit is not None else -1,
                "depth": t.token_fraction(commit) if commit is not None else float("nan"),
                "tvd_at_commit": curve[commit] if commit is not None else float("nan"),
            }
            row.update(concentration(arms[commit]) if commit is not None
                       else dict.fromkeys(concentration(arms[0]), float("nan")))
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    experiment = load_resample(args.config)
    entities = entity_lookup(experiment.name)
    print(f"{args.config}: {len(experiment.models)} models, artifacts at {data_root()}")
    print(f"sweeping delta over {DELTAS[0]} to {DELTAS[-1]}\n")

    rows: list[dict] = []
    for model in experiment.models:
        mine = analyse(experiment.sweep_run(model), entities)
        rows.extend(mine)
        print(f"  {model:<30} {len(mine) // len(DELTAS):>5} traces", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "delta_sweep.csv"
    write_csv(path, rows)

    print(f'\n{"delta":>7} {"committing":>11} {"median depth":>13} {"top share":>10}')
    for delta in DELTAS:
        mine = [r for r in rows if r["delta"] == delta]
        ok = [r for r in mine if r["commits"]]
        depths = sorted(r["depth"] for r in ok)
        tops = sorted(r["top_share"] for r in ok)
        print(f"{delta:>7.2f} {100 * len(ok) / len(mine):>10.1f}% "
              f"{depths[len(depths) // 2]:>13.3f} {tops[len(tops) // 2]:>10.2f}")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
