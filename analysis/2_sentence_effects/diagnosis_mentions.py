"""Which sentences of a trace name a diagnosis, and whether it is the one the trace gave.

One row per sentence, with two flags:

    mentions_final   the sentence names the diagnosis the base trace ended on
    mentions_other   the sentence names some other diagnosis the case produced

A diagnosis is reduced to its content words and a sentence counts as naming it when all of
them appear. A sentence matching both is recorded as final, since a sentence weighing the
eventual answer against a rival is doing the former.

The final diagnosis comes from the base trace's own answer. The other diagnoses are the
case's remaining judged entities, so the same reduction is applied across every bucket the
case produced rather than to a hand-written list.

`first_mention` is not stored: it is the first sentence with `mentions_final`, derived
downstream, so the same file serves an analysis that wants every mention and one that wants
only the first.

    python analysis/2_sentence_effects/diagnosis_mentions.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dctax.config import data_root, load_resample  # noqa: E402
from dctax.rollouts.utils import trace_summaries  # noqa: E402

OUT = HERE / "out"

#: Words carrying no diagnostic identity: laterality, the generic nouns that attach to any
#: diagnosis, severity and course, and connectives. Removed before matching, so
#: "Left-sided subphrenic abscess" and "subphrenic abscess" reduce alike.
STOPWORDS = {
    "l", "r", "left", "right", "bilateral", "unilateral", "sided", "side",
    "upper", "lower", "proximal", "distal", "anterior", "posterior",
    "syndrome", "disease", "disorder", "condition", "infection", "reaction",
    "deficiency", "insufficiency", "failure", "injury", "lesion", "tumour", "tumor",
    "carcinoma", "cancer", "malignancy", "neoplasm", "abnormality",
    "acute", "chronic", "severe", "mild", "moderate", "primary", "secondary",
    "early", "late", "recurrent", "congenital", "acquired", "idiopathic",
    "the", "a", "an", "of", "and", "or", "with", "due", "to", "in", "on", "for",
    "type", "stage", "grade", "class", "possible", "probable", "likely", "suspected",
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit, drop empty tokens."""
    return [t for t in _NON_WORD.split((text or "").lower()) if t]


def content_words(name: str) -> tuple[str, ...]:
    """The content words of a diagnosis, in order, duplicates removed."""
    out: list[str] = []
    for token in normalise(name):
        if token not in STOPWORDS and token not in out:
            out.append(token)
    return tuple(out)


def names(flat: str, words: tuple[str, ...]) -> bool:
    """Whether a normalised sentence contains every word of a diagnosis.

    Matching is on the flattened sentence rather than token by token, so a word still counts
    when it appears inside a longer form: "Beta-thalassemia" and "beta thalassemia" reduce to
    the same text, and "thalassemia" is found in either.
    """
    return bool(words) and all(word in flat for word in words)


def case_entities(experiment_name: str) -> dict[str, set[str]]:
    """Every judged entity name each case produced."""
    path = data_root() / "rollouts" / f"grades_{experiment_name}.parquet"
    table = pq.read_table(path, columns=["case_id", "canonical"]).to_pylist()
    out: dict[str, set[str]] = defaultdict(set)
    for row in table:
        if row["canonical"]:
            out[row["case_id"]].add(row["canonical"])
    return out


def build(config_name: str) -> list[dict]:
    """One row per sentence of every base trace."""
    experiment = load_resample(config_name)
    entities = case_entities(experiment.name)

    rows: list[dict] = []
    for model in experiment.models:
        for trace in trace_summaries(experiment.sweep_run(model)):
            final = content_words(trace.raw_answer)
            # Every other entity the case produced, minus any that reduce to nothing or to
            # exactly the final answer's words.
            others = [
                w for w in (content_words(e) for e in sorted(entities.get(trace.case_id, ())))
                if w and w != final
            ]
            for index, sentence in enumerate(trace.sentences):
                flat = " ".join(normalise(sentence))
                is_final = names(flat, final)
                # Precedence: a sentence naming both is recorded as naming the final answer.
                is_other = (not is_final) and any(names(flat, w) for w in others)
                rows.append(
                    {
                        "model": trace.model,
                        "case_id": trace.case_id,
                        "trace_id": trace.trace_id,
                        "base_is_correct": trace.base_is_correct,
                        "sentence_index": index,
                        "n_sentences": trace.n_sentences,
                        "mentions_final": int(is_final),
                        "mentions_other": int(is_other),
                        "base_answer": trace.raw_answer,
                        "final_words": " ".join(final),
                        "n_other_entities": len(others),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="screen1000", help="under configs/resampling/")
    args = parser.parse_args()

    rows = build(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "diagnosis_mentions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_trace[r["trace_id"]].append(r)
    with_final = [t for t in by_trace.values() if any(r["mentions_final"] for r in t)]

    print(f"{len(rows):,} sentences over {len(by_trace)} traces")
    print(f"  mention the final diagnosis : {sum(r['mentions_final'] for r in rows):>7,} "
          f"({100 * sum(r['mentions_final'] for r in rows) / len(rows):.1f}%)")
    print(f"  mention another diagnosis   : {sum(r['mentions_other'] for r in rows):>7,} "
          f"({100 * sum(r['mentions_other'] for r in rows) / len(rows):.1f}%)")
    print(f"  traces with any final mention: {len(with_final)} of {len(by_trace)} "
          f"({100 * len(with_final) / len(by_trace):.1f}%)\n")

    print(f'{"model":<30} {"sentences":>10} {"final %":>8} {"other %":>8} '
          f'{"traces w/ mention":>18} {"median first pos":>17}')
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        traces = defaultdict(list)
        for r in mine:
            traces[r["trace_id"]].append(r)
        firsts = []
        for t in traces.values():
            hit = [r for r in t if r["mentions_final"]]
            if hit:
                firsts.append(min(r["sentence_index"] for r in hit) / t[0]["n_sentences"])
        firsts.sort()
        print(f'{model:<30} {len(mine):>10,} '
              f'{100 * sum(r["mentions_final"] for r in mine) / len(mine):>7.1f}% '
              f'{100 * sum(r["mentions_other"] for r in mine) / len(mine):>7.1f}% '
              f'{len(firsts):>8} of {len(traces):<7} '
              f'{firsts[len(firsts) // 2] if firsts else float("nan"):>17.2f}')
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
