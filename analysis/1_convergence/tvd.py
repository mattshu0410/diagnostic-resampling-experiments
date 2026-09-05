"""Does the whole answer distribution settle, or only the eventual answer's share?

`convergence.py` tracks one number per boundary: the share of rollouts giving the base
trace's own answer. That is a one-dimensional projection, so it can sit still while the rest
of the mass churns between rival diagnoses - a boundary reading {A:8, B:12} and a later one
reading {A:8, C:12} look identical to it.

This measures the whole distribution instead. For boundary i and every later boundary j, take
the total variation distance

    TVD(P_i, P_j) = 0.5 * sum_a |P_i(a) - P_j(a)|

over the trace's own answer vocabulary: every entity any of its rollouts produced, plus the
base trace's answer. TVD ignores entities neither side used, so unlike a smoothed KL it does
not move with the size of that vocabulary.

Each boundary is then summarised over its forward comparisons - the worst case, which is what
a convergence rule would threshold, and the median, whose distance from the worst case says
how variable the disagreement is at that depth. The final boundary has nothing after it and
is not reported.

    DCTAX_DATA_ROOT=/path/to/artifacts python analysis/1_convergence/tvd.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from convergence import (  # noqa: E402
    BAND,
    BINS,
    BIN_LABELS,
    MIN_GRADED,
    OUT,
    boxplot,
    composition_bar,
    convergence_bin,
    mean_curves,
    model_palette,
    pairwise_tests,
    trajectory_plot,
    write_csv,
)
from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts.utils import (  # noqa: E402
    boundary_entities,
    entity_lookup,
    trace_summaries,
    tvd_matrix,
)

#: Fractions of the total decline the sweep asks for.
DROP_FRACTIONS = tuple(round(0.05 * k, 2) for k in range(1, 21))


def drop_widths(curve: list[float], depths: list[float], commit: int) -> list[dict] | None:
    """How far back from the commitment point each fraction of the decline reaches.

    The decline is measured on the worst-case forward TVD curve, from boundary 0 to the
    commitment boundary: after commitment the curve is already inside the band, so anything
    further is not part of becoming committed. Writing F for the fraction of that decline a
    boundary has achieved, the window always ends at the commitment point, so

        w(q) = d(commit) - d(a),   a = max { a : F(commit) - F(a) >= q }

    Anchoring the window rather than searching for the narrowest one anywhere is what keeps a
    transient dip from being mined: the curve can fall below its commitment value early and
    rebound, and a free search would place a narrow window on that dip and call the trace
    sudden. An anchored window has to span the rebound, so the dip scores negatively and the
    scan passes over it.

    w(1) is usually less than the full [0, commit] span rather than equal to it, because the
    curve rises above its boundary-0 value at some point in 84-93% of traces: boundary 0 is
    the prompt-only distribution, which often already resembles the answer the model settles
    on, and the model then wanders and disagrees with its own later boundaries more than the
    prompt did. So w(1) reads as the last point at which the trace was at least as far from
    its committed distribution as it was before writing anything - the depth from which the
    descent effectively ran.

    Returns None when there is no descent to characterise - the trace was already committed at
    its first boundary, or its worst-case TVD never falls.
    """
    if commit <= 0 or commit >= len(curve):
        return None
    total = curve[0] - curve[commit]
    if total <= 0:
        return None

    reached = (np.asarray(curve[: commit + 1]) - curve[commit]) / total
    d = np.asarray(depths[: commit + 1])
    span = d[commit] - d[0]

    out = []
    for q in DROP_FRACTIONS:
        candidates = np.nonzero(reached >= q - 1e-12)[0]
        if not candidates.size:
            continue
        a = int(candidates[-1])
        w = float(d[commit] - d[a])
        out.append({"drop_fraction": q, "width": w,
                    "width_of_descent": w / span if span > 0 else float("nan")})
    return out


def analyse(run: str, entities: dict) -> tuple[list[dict], list[dict], dict]:
    """Per-trace rows, per-boundary rows and the exclusion tally for one model's sweep.

    The same two gates as `convergence.py`, so the two analyses describe the same traces.
    """
    traces = trace_summaries(run)
    graded = boundary_entities(run, entities, cases={t.trace_id: t.case_id for t in traces})

    per_trace: list[dict] = []
    per_boundary: list[dict] = []
    tally = {"traces": len(traces), "no_base_entity": 0, "thin_boundary": 0, "kept": 0}

    for t in traces:
        base_entity = entities.get((t.case_id, t.raw_answer)) if t.raw_answer else None
        if base_entity is None:
            tally["no_base_entity"] += 1
            continue

        arms = [graded.get((t.trace_id, b), []) for b in range(t.n_boundaries)]
        if any(len(a) <= MIN_GRADED for a in arms):
            tally["thin_boundary"] += 1
            continue

        # The base trace's own answer joins the vocabulary even when no rollout produced it.
        # TVD is unchanged by an entity both sides give zero mass, so this only matters for
        # reporting the vocabulary size honestly.
        vocabulary = sorted({e for arm in arms for e in arm} | {base_entity})
        matrix = tvd_matrix(arms, vocabulary)

        forward_max, forward_median = [], []
        for i in range(t.n_boundaries - 1):
            forward = matrix[i, i + 1:]
            forward_max.append(float(forward.max()))
            forward_median.append(float(np.median(forward)))
            per_boundary.append(
                {
                    "model": t.model,
                    "case_id": t.case_id,
                    "trace_id": t.trace_id,
                    "base_is_correct": t.base_is_correct,
                    "boundary_index": i,
                    "n_forward": int(forward.size),
                    "tvd_max_forward": forward_max[-1],
                    "tvd_median_forward": forward_median[-1],
                    "tvd_next": float(matrix[i, i + 1]),
                    "token_frac": t.token_fraction(i),
                }
            )

        # First boundary whose worst forward disagreement is inside the band; the final
        # boundary has nothing after it, so it always qualifies and is the fallback.
        converge = next((i for i, v in enumerate(forward_max) if v <= BAND),
                        t.n_boundaries - 1)
        per_trace.append(
            {
                "model": t.model,
                "case_id": t.case_id,
                "trace_id": t.trace_id,
                "base_is_correct": t.base_is_correct,
                "n_sentences": t.n_sentences,
                "n_boundaries": t.n_boundaries,
                "vocabulary_size": len(vocabulary),
                "converge_boundary_tvd": converge,
                "converge_token_frac_tvd": t.token_fraction(converge),
                "tvd_max_first": forward_max[0] if forward_max else 0.0,
                "tvd_max_at_convergence": (
                    forward_max[converge] if converge < len(forward_max) else 0.0
                ),
                "tvd_median_first": forward_median[0] if forward_median else 0.0,
            }
        )
        tally["kept"] += 1

    return per_trace, per_boundary, tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    experiment = load_resample(args.config)
    entities = entity_lookup(experiment.name)
    models = list(experiment.models)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{args.config}: {len(models)} models, artifacts at {data_root()}\n")

    per_trace: list[dict] = []
    per_boundary: list[dict] = []
    print(f'{"model":<30} {"traces":>7} {"no base entity":>15} {"thin boundary":>14} {"kept":>6}')
    for model in models:
        trace_rows, boundary_rows, tally = analyse(experiment.sweep_run(model), entities)
        per_trace.extend(trace_rows)
        per_boundary.extend(boundary_rows)
        print(f'{model:<30} {tally["traces"]:>7} {tally["no_base_entity"]:>15} '
              f'{tally["thin_boundary"]:>14} {tally["kept"]:>6}', flush=True)

    write_csv(OUT / "tvd_per_trace.csv", per_trace)
    write_csv(OUT / "tvd_per_boundary.csv", per_boundary)

    # Drop-width sweep, rebuilt from the per-boundary curves already in hand.
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for r in per_boundary:
        by_trace[r["trace_id"]].append(r)
    commit_of = {r["trace_id"]: r["converge_boundary_tvd"] for r in per_trace}
    meta = {r["trace_id"]: r for r in per_trace}
    drop_rows: list[dict] = []
    no_descent = Counter()
    for trace_id, rows in by_trace.items():
        rows.sort(key=lambda r: r["boundary_index"])
        found = drop_widths([r["tvd_max_forward"] for r in rows],
                            [r["token_frac"] for r in rows], commit_of[trace_id])
        if found is None:
            no_descent[meta[trace_id]["model"]] += 1
            continue
        for row in found:
            drop_rows.append({
                "model": meta[trace_id]["model"],
                "case_id": meta[trace_id]["case_id"],
                "trace_id": trace_id,
                "base_is_correct": meta[trace_id]["base_is_correct"],
                "converge_token_frac_tvd": meta[trace_id]["converge_token_frac_tvd"],
                **row,
            })
    write_csv(OUT / "tvd_drop_width.csv", drop_rows)
    print("\ntraces with no descent to characterise (committed at boundary 0, or "
          "worst-case TVD never falls):")
    for model in models:
        print(f'  {model:<30} {no_descent[model]:>4}')

    palette = model_palette(models)
    common = dict(
        x_key="drop_fraction",
        y_key="width_of_descent",
        summarise_by="model",
        summarise=mean_curves,
        diagonal=True,
        xlabel="fraction of the total decline captured",
        ylabel="shortest window achieving it\n(fraction of the descent region)",
    )
    trajectory_plot(
        [(m, [r for r in drop_rows if r["model"] == m]) for m in models],
        OUT / "tvd_drop_width_by_model.png",
        suptitle="How concentrated the decline is within the descent   "
                 "(line: mean +/- SE; dotted: evenly spread)",
        **common,
    )
    trajectory_plot(
        [("all models", drop_rows)],
        OUT / "tvd_drop_width.png",
        columns=1,
        colour_key="model",
        palette=palette,
        panel_size=(7.6, 5.4),
        **common,
    )

    tests = pairwise_tests(per_trace, models, depth_key="converge_token_frac_tvd")
    write_csv(OUT / "tvd_pairwise_tests.csv", tests)
    boxplot(per_trace, models, OUT / "tvd_boxplot.png", tests,
            depth_key="converge_token_frac_tvd",
            title="Where the answer distribution stabilises")

    trajectory_plot(
        [(m, [r for r in per_boundary if r["model"] == m]) for m in models],
        OUT / "tvd_trajectories.png",
        suptitle="Total variation distance from every later boundary  "
                 "(solid: worst case, dashed: median)",
        y_key="tvd_max_forward",
        summarise_by="model",
        extra_curves=(("tvd_median_forward", "--"),),
        hline=BAND,
        ylabel="TVD to all subsequent\nboundaries of the same trace",
    )
    trajectory_plot(
        [(m, [r for r in per_boundary if r["model"] == m]) for m in models],
        OUT / "tvd_trajectories_by_model.png",
        suptitle="Total variation distance from every later boundary  "
                 "(solid: worst case, dashed: median)",
        y_key="tvd_max_forward",
        colour_key="model",
        palette=model_palette(models),
        summarise_by="model",
        extra_curves=(("tvd_median_forward", "--"),),
        ylabel="TVD to all subsequent\nboundaries of the same trace",
    )

    # The same depth bands as the composition figure, so a band's trajectories can be read
    # beside its share of the traces.
    band_of = {r["trace_id"]: convergence_bin(r["converge_token_frac_tvd"])
               for r in per_trace}
    write_csv(
        OUT / "tvd_composition.csv",
        composition_bar(per_trace, models, model_palette(models),
                        OUT / "tvd_composition_bar.png",
                        depth_key="converge_token_frac_tvd",
                        title="Where the answer distribution stabilises, "
                              "and which models sit in each band"),
    )
    for band in BINS:
        rows = [r for r in per_boundary if band_of.get(r["trace_id"]) == band]
        if not rows:
            continue
        trajectory_plot(
            [(f"commitment depth {BIN_LABELS[band]}", rows)],
            OUT / f"tvd_band_{band.replace('%', 'pct').replace('-', '_')}.png",
            y_key="tvd_max_forward",
            colour_key="model",
            palette=model_palette(models),
            columns=1,
            legend=True,
            summarise_by="model",
            statistic="mean",
            hline=BAND,
            panel_size=(7.6, 5.4),
            ylabel="TVD to all subsequent\nboundaries of the same trace",
        )

    def q(values, p):
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    print(f'\nTVD to all later boundaries, at the FIRST boundary of each trace')
    print(f'{"model":<30} {"n":>4} {"worst median":>13} {"median median":>14} '
          f'{"vocab size":>11}')
    for model in models:
        rows = [r for r in per_trace if r["model"] == model]
        if not rows:
            continue
        print(f'{model:<30} {len(rows):>4} '
              f'{q([r["tvd_max_first"] for r in rows], .5):>13.2f} '
              f'{q([r["tvd_median_first"] for r in rows], .5):>14.2f} '
              f'{q([r["vocabulary_size"] for r in rows], .5):>11.0f}')

    print(f'\nConvergence depth under TVD (band {BAND}), as a fraction of reasoning tokens')
    print(f'{"model":<30} {"n":>4} {"p10":>6} {"q1":>6} {"median":>7} {"q3":>6} {"p90":>6}')
    for model in models:
        values = [r["converge_token_frac_tvd"] for r in per_trace if r["model"] == model]
        if not values:
            continue
        print(f'{model:<30} {len(values):>4} {q(values, .1):>6.2f} {q(values, .25):>6.2f} '
              f'{q(values, .5):>7.2f} {q(values, .75):>6.2f} {q(values, .9):>6.2f}')

    significant = [t for t in tests if t["significant"]]
    print(f"\nPairwise Welch t-tests, BH corrected over {len(tests)} pairs: "
          f"{len(significant)} significant at q<0.05")
    print("Pairs that do NOT differ:")
    for t in tests:
        if not t["significant"]:
            print(f'  {t["model_a"]:<30} vs {t["model_b"]:<30} q={t["p_welch_bh"]:.3f}')
    for t in tests:
        if not t["agrees_with_mannwhitney"]:
            print(f'  borderline: {t["model_a"]:<24} vs {t["model_b"]:<24} '
                  f'welch q={t["p_welch_bh"]:.4f}, MWU q={t["p_mannwhitney_bh"]:.4f}')

    print(f"\n{len(per_trace)} traces, {len(per_boundary):,} boundaries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
