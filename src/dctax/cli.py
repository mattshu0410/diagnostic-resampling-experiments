"""Command line entry point.

    dctax cohort         --cohort <name>
    dctax generate       --model <slug> --cohort <name>
    dctax pool           --config <name>
    dctax resample       --config <name> --model <slug> --stage screen|sweep
    dctax score          --config <name> --model <slug> --stage screen|sweep
    dctax select         --config <name>
    dctax grade-rollouts --config <name> --stage screen|sweep
    dctax importance     --config <name>

Resampling runs in that order: draw one pool shared by every model, screen it at boundary 0,
grade the answers, select cases, sweep every boundary, then score and grade again before
importance. Grading is shared across models so a case's answer entities mean the same thing
in each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from dctax import config
from dctax.data import cohort as cohorts
from dctax.generate import responses as generation
from dctax.rollouts import generate as resampling
from dctax.rollouts import grade as grading
from dctax.rollouts import importance as scoring
from dctax.rollouts import screen as screening
from dctax.rollouts import semantic
from dctax.utils import llm


def _responses_path(slug: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return config.data_root() / "responses" / f"responses_{slug}.json"


def _cohort(args: argparse.Namespace) -> int:
    cfg = config.load_cohort(args.cohort)
    print(f"{cfg.name}: {len(cfg.sources)} sources")
    cases = cohorts.build(cfg, args.datasets)
    out = cohorts.write(cfg, cases, args.out)
    print(f"{len(cases)} cases -> {out}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    cfg = config.load_model(args.model)
    cohort = config.load_cohort(args.cohort)
    cases = cohorts.load(cohort.name) if cohorts.path_for(cohort.name).exists() \
        else cohorts.build(cohort)
    if args.limit:
        cases = cases[: args.limit]

    print(f"{cfg.slug}: {len(cases)} cases, {cfg.decoding.max_new_tokens} max new tokens")
    engine, tokenizer = generation.load_model(cfg)
    records = generation.generate(cfg, cases, cohort, engine, tokenizer, path=args.out)
    print(f"{len(records)} responses -> {args.out or generation.responses_path(cfg.slug)}")
    return 0


def _pool_path(experiment: config.ResampleConfig) -> Path:
    return config.data_root() / "rollouts" / f"pool_{experiment.name}.json"


def _selection_path(experiment: config.ResampleConfig) -> Path:
    return config.data_root() / "rollouts" / f"cases_{experiment.name}.json"


def _grades_path(experiment: config.ResampleConfig) -> Path:
    return config.data_root() / "rollouts" / f"grades_{experiment.name}.parquet"


def _pool(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    payload = screening.build_pool(
        experiment.models,
        cohort=experiment.cohort,
        total=experiment.pool_size,
        seed=experiment.seed,
    )
    out = screening.write(_pool_path(experiment), payload)
    digest = hashlib.sha256(",".join(payload["case_ids"]).encode()).hexdigest()[:16]
    print(f"{len(payload['case_ids'])} cases -> {out}")
    print(f"case_ids sha256 {digest}  (identical on any machine from the cohort and seed alone)")
    return 0


def _resample(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    screening_pass = args.stage == "screen"
    payload = json.loads(
        (_pool_path(experiment) if screening_pass else _selection_path(experiment)).read_text()
    )
    cases = payload["case_ids"]
    sample_source, sample_picks = None, None
    if not screening_pass and "by_model" in payload:
        # The sweep takes the screened samples this model qualified on, so the cases and the
        # traces within them both differ by model.
        cases = payload["by_model"].get(args.model, [])
        sample_source = experiment.screen_run(args.model)
        sample_picks = {
            row["case_id"]: [(row["correct_index"], True), (row["wrong_index"], False)]
            for row in payload["rows"] if row["model"] == args.model
        }
        print(f"{len(cases)} cases selected for {args.model}, two traces each", flush=True)
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        if not 1 <= index <= count:
            raise ValueError(f"--shard {args.shard}: index must be within 1..{count}")
        # Strided rather than contiguous, so each shard gets a mix of long and short traces.
        cases = cases[index - 1 :: count]
        print(f"shard {index} of {count}: {len(cases)} cases", flush=True)
    totals = resampling.run(
        model=args.model,
        run_name=(experiment.screen_run if screening_pass else experiment.sweep_run)(args.model),
        n_cases=len(cases) * (1 if sample_picks is None else 2),
        n_rollouts=experiment.screen_rollouts if screening_pass else experiment.sweep_rollouts,
        seed=experiment.seed,
        case_ids=cases,
        sample_source=sample_source,
        sample_picks=sample_picks,
        only_boundaries=[0] if screening_pass else None,
        # The screen resamples one boundary, so its cost does not scale with trace length and
        # the sweep's length cap would drop long traces before they are ever screened.
        max_sentences=None if screening_pass else experiment.max_sentences,
        max_new_tokens=experiment.max_new_tokens,
        engine_kwargs={"max_model_len": args.max_model_len,
                       "gpu_memory_utilization": args.gpu_memory_utilization},
    )
    print(totals)
    return 0


def _score(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    run = (experiment.screen_run if args.stage == "screen" else experiment.sweep_run)(args.model)
    cfg = config.load_model(args.model)
    matcher = semantic.SemanticMatcher(device=args.device)
    scoring.score_rollouts(run, scoring.NormalisedStringScorer(cfg), matcher)
    return 0


def _grade_rollouts(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    runs = [
        (experiment.screen_run if args.stage == "screen" else experiment.sweep_run)(slug)
        for slug in experiment.models
    ]
    records = {r["pmcid"]: r for r in generation.load_existing(
        generation.responses_path(experiment.models[0]))}
    out = grading.grade_run(
        runs, records, out=_grades_path(experiment),
        llm_cfg=llm.LLMConfig(concurrency=args.concurrency),
    )
    print(f"grades -> {out}")
    return 0


def _select(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    runs = {slug: experiment.screen_run(slug) for slug in experiment.models}
    grades = grading.load_grades(_grades_path(experiment))

    capped: set[str] = set()
    for run in runs.values():
        capped |= screening.capped_cases(run, experiment.max_new_tokens)

    judged = screening.outcomes(runs, grades)
    if not args.skip_validation:
        judged = screening.buildable(judged, runs)
    payload = screening.select(
        judged,
        core_min=experiment.core_min,
        floor=experiment.floor,
        min_each=experiment.min_each,
        exclude=capped,
    )
    out = screening.write(_selection_path(experiment), payload)
    per_model = Counter(row["model"] for row in payload["rows"])
    print(f"{len(capped)} cases excluded for reaching the token cap")
    print(f"core of {len(payload['core'])} cases shared by >= {experiment.core_min} models")
    for slug in experiment.models:
        print(f"  {slug:<32} {per_model.get(slug, 0):>4} cases")
    print(f"{len(payload['case_ids'])} cases, {len(payload['rows'])} assignments, "
          f"{2 * len(payload['rows'])} traces -> {out}")
    return 0


def _importance(args: argparse.Namespace) -> int:
    experiment = config.load_resample(args.config)
    grades = grading.load_grades(_grades_path(experiment))
    # None asks each run for its own median replacement similarity; a number fixes the cut.
    threshold = None if args.threshold == "median" else float(args.threshold)
    for slug in experiment.models:
        print(experiment.sweep_run(slug), flush=True)
        scoring.importance_table(experiment.sweep_run(slug), grades, threshold=threshold)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dctax")
    sub = parser.add_subparsers(dest="command", required=True)


    cohort_cmd = sub.add_parser("cohort", help="assemble the cases from their source datasets")
    cohort_cmd.add_argument(
        "--cohort", default="main2073", help="cohort config under configs/cohorts/"
    )
    cohort_cmd.add_argument("--datasets", type=Path, help="override the dataset directory")
    cohort_cmd.add_argument("--out", type=Path, help="override the output file")
    cohort_cmd.set_defaults(func=_cohort)

    generate_cmd = sub.add_parser("generate", help="generate chain-of-thought responses")
    generate_cmd.add_argument("--model", required=True, help="model slug under configs/models/")
    generate_cmd.add_argument(
        "--cohort", default="main2073", help="cohort config under configs/cohorts/"
    )
    generate_cmd.add_argument("--limit", type=int, help="use only the first N cases")
    generate_cmd.add_argument("--out", type=Path, help="override the responses file")
    generate_cmd.set_defaults(func=_generate)

    pool_cmd = sub.add_parser("pool", help="draw one screening pool shared by every model")
    pool_cmd.add_argument("--config", required=True, help="under configs/resampling/")
    pool_cmd.set_defaults(func=_pool)

    resample_cmd = sub.add_parser("resample", help="generate rollouts at sentence boundaries")
    resample_cmd.add_argument("--config", required=True, help="under configs/resampling/")
    resample_cmd.add_argument("--model", required=True, help="model slug under configs/models/")
    resample_cmd.add_argument(
        "--stage", choices=("screen", "sweep"), required=True,
        help="screen resamples boundary 0 over the pool; sweep every boundary of the selection",
    )
    resample_cmd.add_argument(
        "--shard", help="i/n to take every nth case, for splitting one model across GPUs"
    )
    resample_cmd.add_argument("--max-model-len", type=int, default=32768)
    resample_cmd.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    resample_cmd.set_defaults(func=_resample)

    score_cmd = sub.add_parser("score", help="extract answers and replacement similarity")
    score_cmd.add_argument("--config", required=True)
    score_cmd.add_argument("--model", required=True)
    score_cmd.add_argument("--stage", choices=("screen", "sweep"), required=True)
    score_cmd.add_argument("--device", help="torch device for the similarity model")
    score_cmd.set_defaults(func=_score)

    grade_cmd = sub.add_parser("grade-rollouts", help="group and judge every distinct answer")
    grade_cmd.add_argument("--config", required=True)
    grade_cmd.add_argument("--stage", choices=("screen", "sweep"), required=True)
    grade_cmd.add_argument(
        "--concurrency", type=int, default=grading.CONCURRENCY, help="in-flight judge requests"
    )
    grade_cmd.set_defaults(func=_grade_rollouts)

    select_cmd = sub.add_parser("select", help="pick the cases to sweep from the screen")
    select_cmd.add_argument("--config", required=True)
    select_cmd.add_argument(
        "--skip-validation", action="store_true",
        help="do not check that every screened sample rebuilds into a sweepable trace; faster, "
             "but the sweep then skips cases the selection counted on",
    )
    select_cmd.set_defaults(func=_select)

    importance_cmd = sub.add_parser("importance", help="per-sentence importance from rollouts")
    importance_cmd.add_argument("--config", required=True)
    importance_cmd.add_argument(
        "--threshold", default="median",
        help="similarity cut splitting each boundary's rollouts: 'median' for the run's own "
             "median replacement similarity, or a number for a fixed cut",
    )
    importance_cmd.set_defaults(func=_importance)


    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
