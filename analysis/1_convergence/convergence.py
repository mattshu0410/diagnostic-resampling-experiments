"""Where in a trace does the answer converge?

Every boundary of a base trace already has rollouts. For each one, take the share of graded
rollouts whose answer falls in the same judged entity as the base trace's own final answer.
That gives a curve p(0..M) over the trace's M+1 boundaries: how strongly the model is already
committed to the answer it eventually gave.

Convergence is the earliest boundary c from which that curve no longer moves - for every
boundary b at or after c, |p(b) - p(c)| <= BAND - reported as a fraction of the trace's
reasoning tokens, so a 400-sentence trace and a 12-sentence one are on the same axis.

A trace is kept only when its base answer resolves to a judged entity and every boundary
carries more than MIN_GRADED graded rollouts, so no curve is read off a thin boundary.

    DCTAX_DATA_ROOT=/path/to/artifacts python analysis/1_convergence/convergence.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from functools import partial
from statistics import stdev
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts.utils import boundary_entities, entity_lookup, trace_summaries  # noqa: E402

#: Every boundary of a kept trace needs strictly more than this many graded rollouts.
MIN_GRADED = 10

#: The curve may not move further than this from its value at the convergence point.
BAND = 0.2

OUT = HERE / "out"


def convergence_index(curve: list[float], band: float = BAND) -> int:
    """Earliest c with |p(b) - p(c)| <= band for every b >= c.

    Always defined: the last boundary satisfies it with nothing after it to violate it, so a
    trace that never settles is reported as converging at its very end rather than as missing.
    """
    for c, value in enumerate(curve):
        if all(abs(later - value) <= band for later in curve[c:]):
            return c
    return len(curve) - 1


def stable_band_index(curve: list[float], band: float = BAND) -> int:
    """Earliest c whose whole tail spans no more than `band` (max - min <= band).

    A stricter reading of "no longer moves": `convergence_index` measures every later point
    against p(c) and so tolerates a tail spanning 2 * band. Carried alongside so the choice
    between the two readings can be made from the output rather than by rerunning.
    """
    for c in range(len(curve)):
        tail = curve[c:]
        if max(tail) - min(tail) <= band:
            return c
    return len(curve) - 1


def analyse(run: str, entities: dict) -> tuple[list[dict], list[dict], dict]:
    """Per-trace rows, per-boundary rows and the exclusion tally for one model's sweep."""
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

        curve = [sum(e == base_entity for e in a) / len(a) for a in arms]
        per_boundary.extend(
            {
                "model": t.model,
                "case_id": t.case_id,
                "trace_id": t.trace_id,
                "base_is_correct": t.base_is_correct,
                "boundary_index": b,
                "n_graded": len(arms[b]),
                "n_match_base_entity": sum(e == base_entity for e in arms[b]),
                "p_base_entity": curve[b],
                "token_pos": t.boundary_tokens[b],
                "token_frac": t.token_fraction(b),
            }
            for b in range(t.n_boundaries)
        )

        c = convergence_index(curve)
        strict = stable_band_index(curve)
        per_trace.append(
            {
                "model": t.model,
                "case_id": t.case_id,
                "trace_id": t.trace_id,
                "base_is_correct": t.base_is_correct,
                "base_entity": base_entity,
                "n_sentences": t.n_sentences,
                "n_boundaries": t.n_boundaries,
                "reasoning_tokens": t.reasoning_tokens,
                "total_tokens": t.total_tokens,
                "converge_boundary": c,
                "converge_boundary_frac": c / (t.n_boundaries - 1),
                "converge_token": t.boundary_tokens[c],
                "converge_token_frac": t.token_fraction(c),
                "p_at_convergence": curve[c],
                "p_first": curve[0],
                "p_final": curve[-1],
                "converge_boundary_strict": strict,
                "converge_token_frac_strict": t.token_fraction(strict),
                "curve": json.dumps([round(x, 4) for x in curve]),
            }
        )
        tally["kept"] += 1

    return per_trace, per_boundary, tally


def stars(p: float) -> str:
    """Significance marker, or an empty string when the pair does not survive correction."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def pairwise_tests(per_trace: list[dict], models: list[str],
                   depth_key: str = "converge_token_frac") -> list[dict]:
    """Every pair of models compared on convergence depth, Benjamini-Hochberg corrected.

    Welch's t-test rather than Student's: the groups differ in both size (70 to 173) and
    spread, and Welch costs nothing when they do not. Mann-Whitney is carried alongside as a
    robustness column, since convergence depth is a bounded, skewed quantity that a t-test is
    not ideally suited to - the two should agree, and where they do not the pair is worth
    treating as borderline rather than significant.
    """
    from itertools import combinations

    from scipy.stats import false_discovery_control, mannwhitneyu, ttest_ind

    values = {m: [r[depth_key] for r in per_trace if r["model"] == m] for m in models}
    present = [m for m in models if values[m]]
    pairs = list(combinations(present, 2))

    welch = [ttest_ind(values[a], values[b], equal_var=False) for a, b in pairs]
    utest = [mannwhitneyu(values[a], values[b]) for a, b in pairs]
    welch_bh = false_discovery_control([r.pvalue for r in welch], method="bh")
    utest_bh = false_discovery_control([r.pvalue for r in utest], method="bh")

    rows = []
    for (a, b), w, u, wq, uq in zip(pairs, welch, utest, welch_bh, utest_bh):
        mean_a = sum(values[a]) / len(values[a])
        mean_b = sum(values[b]) / len(values[b])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n_a": len(values[a]),
                "n_b": len(values[b]),
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "mean_diff": round(mean_a - mean_b, 4),
                "t": round(float(w.statistic), 4),
                "df": round(float(w.df), 2),
                "p_welch": float(w.pvalue),
                "p_welch_bh": float(wq),
                "stars": stars(wq),
                "significant": bool(wq < 0.05),
                "p_mannwhitney": float(u.pvalue),
                "p_mannwhitney_bh": float(uq),
                "agrees_with_mannwhitney": bool((wq < 0.05) == (uq < 0.05)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boxplot(per_trace: list[dict], models: list[str], path: Path,
            tests: list[dict] | None = None,
            depth_key: str = "converge_token_frac",
            title: str = "Where the answer stops moving") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [m for m in models if any(r["model"] == m for r in per_trace)]
    data = [[r[depth_key] for r in per_trace if r["model"] == m] for m in present]

    fig, ax = plt.subplots(figsize=(11, 11.5))
    drawn = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "#111111", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45,
                    "markerfacecolor": "#555555", "markeredgecolor": "none"},
    )
    for box in drawn["boxes"]:
        box.set_facecolor("#9ecae1")
        box.set_alpha(0.85)
        box.set_edgecolor("#31688e")

    # Jitter every trace over its box: the boxes alone hide how bimodal some models are.
    rng = random.Random(0)
    for i, values in enumerate(data, start=1):
        ax.scatter([i + rng.uniform(-0.13, 0.13) for _ in values], values,
                   s=7, alpha=0.3, color="#31688e", zorder=3, linewidths=0)
        ax.annotate(f"n={len(values)}", (i, 1.03), ha="center", fontsize=8, color="#444444")

    top = 1.08
    if tests:
        position = {m: i + 1 for i, m in enumerate(present)}
        marked = [t for t in tests if t["significant"]
                  and t["model_a"] in position and t["model_b"] in position]
        # One row each, ordered 1-2, 1-3, ... 1-n, 2-3, ... so a given pair can be found by
        # counting rather than by hunting for the bracket that happens to span it.
        marked.sort(key=lambda t: sorted((position[t["model_a"]], position[t["model_b"]])))
        spans = [tuple(sorted((position[t["model_a"]], position[t["model_b"]])))
                 for t in marked]
        levels = list(range(len(marked)))
        base, step, tick = 1.08, 0.045, 0.010
        for (lo, hi), level, test in zip(spans, levels, marked):
            y = base + level * step
            ax.plot([lo, lo, hi, hi], [y, y + tick, y + tick, y],
                    linewidth=0.9, color="#444444", clip_on=False, zorder=5)
            ax.text(hi + 0.10, y, test["stars"], ha="left", va="center",
                    fontsize=8, color="#222222", clip_on=False, zorder=5)
        top = base + (max(levels) + 1) * step + 0.03 if marked else top

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([m.replace("deepseek-r1-distill-", "ds-") for m in present],
                       rotation=30, ha="right")
    ax.set_ylabel("convergence point\n(fraction of the trace's reasoning tokens)")
    ax.set_ylim(-0.05, top)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0.5, color="#bbbbbb", linestyle="--", linewidth=1, zorder=0)
    ax.set_title(f"{title}   (band {BAND}, "
                 f">{MIN_GRADED} graded rollouts at every boundary)", fontsize=12)
    if tests:
        ax.text(0.5, -0.10,
                "Welch t-test on every pair, Benjamini-Hochberg corrected across all "
                f"{len(tests)} pairs.   * q<0.05   ** q<0.01   *** q<0.001   "
                "(non-significant pairs unmarked)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")


#: Where a trace's convergence point falls, as depth d into its own reasoning.
#:
#: Quarters, with the last closed at 1. d = 1 means the trace was still moving at its final
#: boundary rather than settling late - the convergence rule always admits the last boundary,
#: so that is where "never settles" lands - and it now sits inside the top quarter rather than
#: standing alone. `converge_boundary == n_boundaries - 1` in the per-trace CSV still isolates
#: those 60 traces when they need separating.
BINS = ("0-25%", "25-50%", "50-75%", "75-100%")

#: How each bin is written on an axis. The keys stay ASCII so they can name a file and a
#: CSV column; only the display form carries the interval the bin actually means.
BIN_LABELS = {
    "0-25%": "0 \u2264 d < 0.25",
    "25-50%": "0.25 \u2264 d < 0.5",
    "50-75%": "0.5 \u2264 d < 0.75",
    "75-100%": "0.75 \u2264 d \u2264 1",
}


def convergence_bin(d: float) -> str:
    """Which of BINS a convergence depth d falls in."""
    if d < 0.25:
        return "0-25%"
    if d < 0.5:
        return "25-50%"
    return "50-75%" if d < 0.75 else "75-100%"


def model_palette(models: list[str]) -> dict[str, str]:
    """A stable colour per model, shared by every figure so they compose."""
    import matplotlib.pyplot as plt

    wheel = plt.get_cmap("tab10").colors
    return {m: wheel[i % len(wheel)] for i, m in enumerate(models)}


def interpolated_curves(
    rows: list[dict],
    group_key: str,
    y_key: str,
    n_points: int = 41,
    min_traces: int = 5,
    x_key: str = "token_frac",
    statistic: str = "median",
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """Summary curve per group, each trace interpolated onto a shared depth grid.

    For curves whose x values differ from trace to trace - the depth trajectories, where each
    trace has boundaries only at its own depths. When every trace already shares an x grid,
    use `mean_curves`, which needs no interpolation.

    Returns {group: (grid, medians, standard errors)}. Interpolating rather than bucketing
    keeps the population behind the summary constant: a trace only has boundaries at its own
    depths, so an 11-boundary trace would be absent from roughly half of any fixed set of
    buckets and the median would be taken over a different subset of traces at every x.

    The cost is that a short trace is resampled onto more points than it has boundaries, so
    its curve looks smoother than it was measured. `np.interp` holds the end values rather
    than extrapolating, which only matters over the few hundredths of depth before a trace's
    first boundary and after its last.

    The error is sd/sqrt(n) across traces at each grid point. Under `statistic="mean"` that
    is the standard error of the line being drawn. Under `statistic="median"` it is not: it
    still says how well supported the line is, but it is the error of a different statistic
    from the one it surrounds, so do not report it as a confidence interval for the median.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")

    import numpy as np

    grid = np.linspace(0.0, 1.0, n_points)
    curves: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        curves[r[group_key]][r["trace_id"]].append((r[x_key], r[y_key]))

    out: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for group, traces in curves.items():
        if len(traces) < min_traces:
            continue
        stacked = []
        for points in traces.values():
            points.sort()
            stacked.append(np.interp(grid, [x for x, _ in points], [y for _, y in points]))
        block = np.vstack(stacked)
        out[group] = (
            grid.tolist(),
            (np.median if statistic == "median" else np.mean)(block, axis=0).tolist(),
            (block.std(axis=0, ddof=1) / np.sqrt(block.shape[0])).tolist(),
        )
    return out


def mean_curves(
    rows: list[dict],
    group_key: str,
    y_key: str,
    x_key: str,
    min_traces: int = 5,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """Mean curve per group, at the x values the rows already share.

    For curves measured on a common grid, where interpolation would invent points between
    real measurements and hold the first value out to the edge of the axis. Returns
    {group: (x values, means, standard errors)}.

    The mean rather than the median, so the band is the standard error of the statistic it is
    drawn around. It describes how well the summary is located, not how far the traces spread
    - which is far wider, and is what the individual lines behind it show.
    """
    collected: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        collected[r[group_key]][r[x_key]].append(r[y_key])

    out: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for group, at_x in collected.items():
        xs, means, errors = [], [], []
        for x in sorted(at_x):
            values = at_x[x]
            if len(values) < min_traces:
                continue
            xs.append(x)
            means.append(sum(values) / len(values))
            errors.append(stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0)
        if xs:
            out[group] = (xs, means, errors)
    return out


def trajectory_plot(
    panels: list[tuple[str, list[dict]]],
    path: Path,
    *,
    colour_key: str | None = None,
    palette: dict | None = None,
    columns: int = 4,
    suptitle: str = "",
    legend: bool = False,
    alpha: float = 0.09,
    linewidth: float = 0.5,
    summarise_by: str | None = None,
    summarise_min_traces: int = 5,
    summarise=None,
    statistic: str = "median",
    panel_size: tuple[float, float] = (4.6, 3.7),
    y_key: str = "p_base_entity",
    x_key: str = "token_frac",
    diagonal: bool = False,
    xlabel: str = "depth into the reasoning\n(fraction of generated tokens, prompt excluded)",
    extra_curves: tuple[tuple[str, str], ...] = (),
    hline: float | None = None,
    ylabel: str = "rollouts matching the\nbase trace's answer entity",
) -> None:
    """One line per base trace, over the depth of its own reasoning.

    `panels` is (title, per-boundary rows) pairs - the caller decides what a panel is, so the
    same function draws one panel per model, one per convergence bin, or a single panel.
    Lines take their colour from `colour_key` through `palette` when both are given.

    `summarise_by` names a row field to summarise over: one line is drawn per distinct
    value of that field, from `interpolated_curves`, so each trace counts once however
    many boundaries it has.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(panel_size[0] * columns, panel_size[1] * rows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, (title, panel_rows) in zip(axes.flat, panels):
        traces: dict[str, list[tuple[float, float]]] = {}
        colours: dict[str, object] = {}
        for r in panel_rows:
            traces.setdefault(r["trace_id"], []).append((r[x_key], r[y_key]))
            if colour_key:
                colours[r["trace_id"]] = (palette or {}).get(r[colour_key], "#31688e")
        for trace_id, points in traces.items():
            points.sort()
            ax.plot([x for x, _ in points], [y for _, y in points],
                    linewidth=linewidth, alpha=alpha,
                    color=colours.get(trace_id, "#31688e"), solid_capstyle="round")
        if summarise_by:
            summariser = summarise or partial(interpolated_curves, statistic=statistic)
            for key, (xs, ys, se) in summariser(
                panel_rows, summarise_by, y_key, x_key=x_key,
                min_traces=summarise_min_traces
            ).items():
                colour = (palette or {}).get(key, "#08306b")
                ax.fill_between(xs, [y - e for y, e in zip(ys, se)],
                                [y + e for y, e in zip(ys, se)],
                                color=colour, alpha=0.30, linewidth=0, zorder=6)
                ax.plot(xs, ys, color=colour, linewidth=2.0, zorder=7,
                        solid_capstyle="round")
            # Line-only overlays, to show a second summary of the same panel.
            for field, style in extra_curves:
                for key, (xs, ys, _) in (
                    summarise or partial(interpolated_curves, statistic=statistic)
                )(
                    panel_rows, summarise_by, field, x_key=x_key,
                    min_traces=summarise_min_traces
                ).items():
                    ax.plot(xs, ys, color=(palette or {}).get(key, "#08306b"),
                            linewidth=1.6, linestyle=style, zorder=7,
                            solid_capstyle="round")

        if diagonal:
            ax.plot([0, 1], [0, 1], color="#c0392b", linestyle=":", linewidth=1.2, zorder=5)
        if hline is not None:
            ax.axhline(hline, color="#c0392b", linestyle=":", linewidth=1.2, zorder=5)
        ax.set_title(f"{title}   (n={len(traces)})", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)

    if legend and palette:
        from matplotlib.lines import Line2D

        present = [m for m in palette if any(
            r.get(colour_key) == m for _, rs in panels for r in rs)]
        fig.legend(
            [Line2D([0], [0], color=palette[m], linewidth=3) for m in present],
            present, loc="lower center", ncol=min(len(present), 4),
            frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0.08 if legend and palette else 0, 1, 1))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def composition_bar(per_trace: list[dict], models: list[str], palette: dict,
                    path: Path, depth_key: str = "converge_token_frac",
                    title: str = "Where convergence happens, "
                                 "and which models sit in each band") -> list[dict]:
    """One stacked bar per convergence bin, split by the model the traces came from.

    `depth_key` names the column holding the convergence depth, so the same bar serves the
    base-entity-share analysis and the TVD one. The two disagree on roughly 6% of traces, so
    a bar must be built from the same depth a figure beside it was banded on.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {b: {m: 0 for m in models} for b in BINS}
    for r in per_trace:
        counts[convergence_bin(r[depth_key])][r["model"]] += 1

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bottoms = [0] * len(BINS)
    for model in models:
        heights = [counts[b][model] for b in BINS]
        ax.bar(range(len(BINS)), heights, bottom=bottoms, width=0.62,
               color=palette[model], edgecolor="white", linewidth=0.6, label=model)
        bottoms = [a + b for a, b in zip(bottoms, heights)]

    for i, total in enumerate(bottoms):
        ax.annotate(f"{total}\n{100 * total / len(per_trace):.0f}%", (i, total),
                    ha="center", va="bottom", fontsize=10, color="#333333",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels([BIN_LABELS[b] for b in BINS])
    ax.set_xlabel("convergence depth d\n(fraction of the trace's reasoning tokens)")
    ax.set_ylabel("base traces")
    ax.set_ylim(0, max(bottoms) * 1.30)
    ax.set_title(f"{title}\n{len(per_trace)} base traces", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return [{"bin": b, "total": sum(counts[b].values()),
             **{m: counts[b][m] for m in models}} for b in BINS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    experiment = load_resample(args.config)
    entities = entity_lookup(experiment.name)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{args.config}: {len(experiment.models)} models, artifacts at {data_root()}")
    print(f"{len(entities):,} graded (case, answer) pairs\n")

    per_trace: list[dict] = []
    per_boundary: list[dict] = []
    print(f'{"model":<30} {"traces":>7} {"no base entity":>15} {"thin boundary":>14} {"kept":>6}')
    for model in experiment.models:
        trace_rows, boundary_rows, tally = analyse(experiment.sweep_run(model), entities)
        per_trace.extend(trace_rows)
        per_boundary.extend(boundary_rows)
        print(f'{model:<30} {tally["traces"]:>7} {tally["no_base_entity"]:>15} '
              f'{tally["thin_boundary"]:>14} {tally["kept"]:>6}')

    tests = pairwise_tests(per_trace, list(experiment.models))

    write_csv(OUT / "convergence_per_trace.csv", per_trace)
    write_csv(OUT / "convergence_per_boundary.csv", per_boundary)
    write_csv(OUT / "convergence_pairwise_tests.csv", tests)

    models = list(experiment.models)
    palette = model_palette(models)
    boxplot(per_trace, models, OUT / "convergence_boxplot.png", tests)

    # One panel per model, as before, with that model's median over 0.05 depth buckets.
    trajectory_plot(
        [(m, [r for r in per_boundary if r["model"] == m]) for m in models],
        OUT / "convergence_trajectories.png",
        suptitle="Commitment to the eventual answer, one line per base trace",
        summarise_by="model",
    )

    # Composition of each convergence band, and the same bands as separate trajectory
    # figures so they can be laid out alongside the bar.
    bin_of = {r["trace_id"]: convergence_bin(r["converge_token_frac"])
              for r in per_trace}
    composition = composition_bar(per_trace, models, palette,
                                  OUT / "convergence_composition_bar.png")
    write_csv(OUT / "convergence_composition.csv", composition)
    for band in BINS:
        rows = [r for r in per_boundary if bin_of.get(r["trace_id"]) == band]
        if not rows:
            continue
        trajectory_plot(
            [(f"convergence depth {BIN_LABELS[band]}", rows)],
            OUT / f"convergence_trajectories_{band.replace('%', 'pct').replace('-', '_')}.png",
            colour_key="model", palette=palette, columns=1, legend=True,
            summarise_by="model", panel_size=(7.6, 5.4),
        )

    def q(values, p):
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    print("\nConvergence point, as a fraction of the trace's reasoning tokens")
    print(f'{"model":<30} {"n":>4} {"p10":>6} {"q1":>6} {"median":>7} {"q3":>6} {"p90":>6} '
          f'{"median strict":>14}')
    for model in experiment.models:
        values = [r["converge_token_frac"] for r in per_trace if r["model"] == model]
        if not values:
            continue
        strict = [r["converge_token_frac_strict"] for r in per_trace if r["model"] == model]
        print(f'{model:<30} {len(values):>4} {q(values, .1):>6.2f} {q(values, .25):>6.2f} '
              f'{q(values, .5):>7.2f} {q(values, .75):>6.2f} {q(values, .9):>6.2f} '
              f'{q(strict, .5):>14.2f}')

    significant = [t for t in tests if t["significant"]]
    disagree = [t for t in tests if not t["agrees_with_mannwhitney"]]
    print(f"\nPairwise Welch t-tests, BH corrected over {len(tests)} pairs: "
          f"{len(significant)} significant at q<0.05")
    print("Pairs that do NOT differ:")
    for t in tests:
        if not t["significant"]:
            print(f'  {t["model_a"]:<30} vs {t["model_b"]:<30} q={t["p_welch_bh"]:.3f}')
    if disagree:
        print("Pairs where Mann-Whitney disagrees with the t-test (treat as borderline):")
        for t in disagree:
            print(f'  {t["model_a"]:<30} vs {t["model_b"]:<30} '
                  f'welch q={t["p_welch_bh"]:.4f}, MWU q={t["p_mannwhitney_bh"]:.4f}')

    print(f"\n{len(per_trace)} traces, {len(per_boundary):,} boundaries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
