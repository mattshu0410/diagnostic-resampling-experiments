"""What the model would answer, and what its reasoning names, around TVD convergence.

`tvd.py` finds the boundary where each trace's answer distribution stops moving. This aligns
every trace it kept on that boundary (event time 0) and follows, RADIUS boundaries either side:

    forced answers   whether the answer forced at the boundary is the trace's final answer,
                     whether it changed from the previous boundary, and how settled it is
                     over the last WINDOW boundaries
    differentials    how many of the case's diagnoses the last WINDOW sentences name
    activations      how far the hidden state moves into each sentence

Forced answers are counted as rollouts are: a valid answer placed in an entity counts, anything
else is missing rather than wrong. Every forced sample contributes a value per boundary; samples
are averaged within a trace and traces are bootstrapped.

The activations are each sentence's hidden state, mean-pooled over its tokens at one layer per
model, the layer named in each file. They are read from DCTAX_DATA_ROOT, or else fetched from
the dataset, which is gated: request access, then `hf auth login`.

    python analysis/3_forced_answers/forced_answer.py --config screen1000
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "2_sentence_effects"))

import diagnosis_mentions as mentions  # noqa: E402
from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts import store  # noqa: E402
from dctax.rollouts.utils import entity_lookup, trace_summaries  # noqa: E402

OUT = HERE / "out"
DATASET = "matthewshu/medcasereasoning-cot-activations"
WINDOW, RADIUS = 5, 20
METRICS = (
    "activation_velocity",
    "forced_matches_final",
    "forced_change_from_previous",
    "forced_stable_w5",
    "forced_unique_w5",
    "ddx_unique_all_w5",
    "ddx_unique_w5",
    "ddx_sentence_hits_w5",
)


def forced_labels(runs: list[str], grades: dict) -> dict[str, list[list[str]]]:
    """Per trace, one entity sequence over its boundaries for each forced sample.

    A boundary whose answer was invalid or never placed in an entity holds "".
    """
    sequences: dict[str, list[list[str]]] = defaultdict(list)
    for run in runs:
        table = pq.read_table(
            store.run_dir(run) / "forced_answers.parquet",
            columns=["case_id", "trace_id", "boundary_index", "rollout_index", "raw_answer", "valid"],
        ).to_pandas()
        table["label"] = [
            grades.get((case, answer.strip()), "") if valid else ""
            for case, answer, valid in zip(table.case_id, table.raw_answer.fillna(""), table.valid)
        ]
        table = table.sort_values(["trace_id", "rollout_index", "boundary_index"])
        for (trace, _), group in table.groupby(["trace_id", "rollout_index"], sort=False):
            sequences[trace].append(group.label.tolist())
    return sequences


def boundary_metrics(traces: dict, sequences: dict, metadata: pd.DataFrame,
                     entities: dict) -> pd.DataFrame:
    """One row per trace, boundary and forced sample, for every trace tvd.py dated."""
    case_entities = defaultdict(set)
    for (case, _), name in entities.items():
        case_entities[case].add(mentions.content_words(name))
    rows = []
    for trace_id, convergence in zip(metadata.trace_id, metadata.converge_boundary_tvd):
        trace, samples = traces[trace_id], sequences.get(trace_id, [])
        n = trace.n_sentences
        if not samples or any(len(labels) != n + 1 for labels in samples):
            raise ValueError(f"{trace_id}: expected forced answers at all {n + 1} boundaries")
        final = entities[trace.case_id, trace.raw_answer]
        final_words = mentions.content_words(final)
        others = case_entities[trace.case_id] - {final_words, ()}
        # Mentions are lexical, so a sentence ruling a diagnosis out still counts as naming it.
        other_hits, all_hits = [], []
        for sentence in trace.sentences:
            flat = " ".join(mentions.normalise(sentence))
            other = {words for words in others if mentions.names(flat, words)}
            other_hits.append(other)
            all_hits.append(other | ({final_words} if mentions.names(flat, final_words) else set()))
        for labels in samples:
            for b in range(n + 1):
                window = [label for label in labels[max(0, b - WINDOW + 1): b + 1] if label]
                recent = slice(max(0, b - WINDOW), b)
                rows.append({
                    "trace_id": trace_id, "event_time": b - convergence,
                    "forced_matches_final": float(labels[b] == final) if labels[b] else np.nan,
                    "forced_change_from_previous": (
                        float(labels[b] != labels[b - 1]) if b and labels[b] and labels[b - 1]
                        else np.nan
                    ),
                    "forced_stable_w5": float(len(set(window)) == 1) if len(window) == WINDOW else np.nan,
                    "forced_unique_w5": len(set(window)) if b >= WINDOW - 1 and window else np.nan,
                    "ddx_unique_all_w5": len(set().union(*all_hits[recent])) if b >= WINDOW else np.nan,
                    "ddx_unique_w5": len(set().union(*other_hits[recent])) if b >= WINDOW else np.nan,
                    "ddx_sentence_hits_w5": sum(map(bool, other_hits[recent])) if b >= WINDOW else np.nan,
                })
    return pd.DataFrame(rows)


def activations_dir(experiment: str) -> Path:
    """The TVD activations, local if present, else downloaded once from the dataset."""
    relative = Path("activations/tvd") / experiment / "raw"
    if any((data_root() / relative).glob("activations_*.pkl")):
        return data_root() / relative
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(DATASET, repo_type="dataset",
                                 allow_patterns=f"recommended_decoding/{relative.as_posix()}/*")
    return Path(snapshot) / "recommended_decoding" / relative


def cosine_changes(raw_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Consecutive centered cosine distances, standardized within each trace.

    The step into sentence i lands on boundary i + 1, the first prefix holding sentence i, so
    it lines up with the answer forced at that boundary.
    """
    convergence = dict(zip(metadata.trace_id, metadata.converge_boundary_tvd))
    rows, found = [], set()
    for path in sorted(raw_dir.glob("activations_*.pkl")):
        with path.open("rb") as handle:
            vectors, _, keys, running_mean = pickle.load(handle)
        vectors = np.asarray(vectors, np.float32) - np.asarray(running_mean, np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        groups = defaultdict(list)
        for row, (trace, sentence) in enumerate(keys):
            groups[trace].append((int(sentence), row))
        found.update(groups)
        for trace, pairs in groups.items():
            if trace not in convergence or len(pairs) < 2:
                continue
            pairs.sort()
            sentence, order = np.asarray(pairs).T
            if np.any(np.diff(sentence) != 1):
                raise ValueError(f"{path.name}: {trace} has nonconsecutive activation sentences")
            distance = 1 - np.sum(vectors[order[1:]] * vectors[order[:-1]], axis=1)
            z = (distance - distance.mean()) / (distance.std() + 1e-8)
            rows.extend({"trace_id": trace, "event_time": int(i + 1 - convergence[trace]),
                         "activation_velocity": float(value)} for i, value in zip(sentence[1:], z))
    missing = set(convergence) - found
    if missing:
        raise ValueError(f"{len(missing)} TVD traces have no activations, e.g. {sorted(missing)[:3]}")
    return pd.DataFrame(rows)


def event_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Trace-bootstrapped mean of one metric at each event time within RADIUS."""
    part = frame.loc[frame.event_time.between(-RADIUS, RADIUS)].dropna(subset=[target])
    rows = []
    for event, at in part.groupby("event_time", sort=True):
        values = at.groupby("trace_id", sort=False)[target].mean().to_numpy(float)
        digest = hashlib.sha256(f"convergence-event|{target}|{int(event)}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32 - 1))
        draws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(500)]
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append({"target": target, "event_time": int(event), "mean": values.mean(),
                     "ci_low": low, "ci_high": high, "n_traces": len(values)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    experiment = load_resample(args.config)
    runs = [experiment.sweep_run(model) for model in experiment.models]
    entities = entity_lookup(experiment.name)
    metadata = pd.read_csv(HERE.parent / "1_convergence/out/tvd_per_trace.csv")
    traces = {t.trace_id: t for run in runs for t in trace_summaries(run)}
    sequences = forced_labels(runs, entity_lookup(f"{experiment.name}_forced"))
    metrics = boundary_metrics(traces, sequences, metadata, entities)
    changes = cosine_changes(activations_dir(experiment.name), metadata)
    summary = pd.concat([event_summary(changes if target == "activation_velocity" else metrics, target)
                         for target in METRICS], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "event_summary.csv", index=False)
    print(f"{metrics.trace_id.nunique():,} traces, "
          f"{metrics.forced_matches_final.notna().mean():.1%} of forced answers counted "
          f"-> {OUT / 'event_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
