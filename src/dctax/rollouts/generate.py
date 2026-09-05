"""Batched vLLM continuations from every sentence boundary of a trace.

Decoding comes from the model config, never from the resampling settings, so a rollout is
sampled the same way the trace it intervenes on was. Prompts are token ids sliced from the
stored sequence; no text is re-rendered or re-encoded.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Iterator, Sequence

from dctax.config import ModelConfig, load_model
from dctax.generate import responses as generation
from dctax.rollouts import boundaries, store
from dctax.rollouts.schema import BaseTrace, Rollout, token_hash

#: Room a boundary must leave for its continuation, or it is skipped.
MIN_NEW_TOKENS = 256

#: Prompts per engine call, so long traces checkpoint instead of finishing all at once.
BATCH_SIZE = 64

#: Spreads seeds across boundaries so no two boundaries share a sampling stream.
_SEED_STRIDE = 1_000_003

#: Keeps derived seeds inside the range samplers accept.
_SEED_MODULUS = 2**31 - 1


def assert_decoding(cfg: ModelConfig, expected: dict | None) -> None:
    """Fail when declared decoding differs from the model's, which owns these settings."""
    if not expected:
        return
    actual = asdict(cfg.decoding)
    wrong = {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
    if wrong:
        raise ValueError(
            f"{cfg.slug}: resampling config asserts {wrong} as (declared, actual); "
            f"change configs/models/{cfg.slug}.yaml rather than overriding it here"
        )
    unknown = sorted(set(expected) - set(actual))
    if unknown:
        raise ValueError(f"{cfg.slug}: unknown decoding settings {unknown}")


def context_length(engine, override: int | None = None) -> int:
    """The engine's context window, which bounds prefix plus continuation."""
    if override is not None:
        return override
    model_config = getattr(getattr(engine, "llm_engine", None), "model_config", None)
    length = getattr(model_config, "max_model_len", None)
    if length is None:
        raise RuntimeError("could not read max_model_len from the engine; pass max_model_len")
    return int(length)


def _manifest(cfg: ModelConfig, tokenizer) -> dict:
    """Provenance that every invocation of a run must share."""
    import vllm

    return {
        "model": cfg.slug,
        "hf_id": cfg.hf_id,
        "dtype": cfg.dtype,
        "decoding": asdict(cfg.decoding),
        "expected_trace_format": cfg.expected_trace_format,
        "answer_fallbacks": list(cfg.answer_fallbacks),
        "tokenizer": str(getattr(tokenizer, "name_or_path", cfg.hf_id)),
        "vllm_version": vllm.__version__,
    }


def _traces_to_sweep(
    records: list[dict],
    case_ids: Sequence[str] | None,
    cfg,
    tokenizer,
    run_name: str,
    totals: dict,
    *,
    sample_source: str | None,
    sample_picks: dict[str, Sequence[tuple[int, bool | None]]] | None,
) -> Iterator[BaseTrace]:
    """The traces this run sweeps, skipping and logging any that cannot be addressed.

    Without `sample_picks` that is one trace per case, built from the stored response, which is
    what every run before the paired design did.

    With it, each case contributes the named boundary-0 samples of an earlier run instead. Those
    are alternative traces the model produced for the same case, so a sweep can compare a trace
    that reached the right answer against one that did not without the case differing between
    them. `trace_id` is a content hash, so each sample is distinct in the store without anything
    else changing.
    """
    if sample_picks is None:
        for record in _select(records, case_ids):
            try:
                yield boundaries.prepare_trace(record, cfg, tokenizer)
            except boundaries.BoundaryError as exc:
                store.log_skips(run_name, [exc])
                totals["skipped"] += 1
        return

    if sample_source is None:
        raise ValueError("sample_picks needs sample_source naming the run they come from")
    by_case = {r.get("pmcid") or r.get("question_id"): r for r in records}
    bases = {t.case_id: t for t in store.load_traces(sample_source)}
    for case in (case_ids if case_ids is not None else sample_picks):
        picks = sample_picks.get(case)
        base = bases.get(case)
        if not picks or base is None:
            store.log_skips(run_name, [boundaries.BoundaryError(case, "no screened trace")])
            totals["skipped"] += 1
            continue
        rollouts = {r.rollout_index: r for r in store.load_rollouts(sample_source, case)
                    if r.boundary_index == 0}
        for index, correct in picks:
            sample = rollouts.get(index)
            if sample is None:
                store.log_skips(
                    run_name, [boundaries.BoundaryError(case, f"sample {index} missing")]
                )
                totals["skipped"] += 1
                continue
            record = {**by_case.get(case, {}), "pmcid": case, "is_correct": correct}
            try:
                yield boundaries.prepare_sample(
                    base, sample.prefix_token_count, sample.continuation, record, cfg, tokenizer
                )
            except boundaries.BoundaryError as exc:
                store.log_skips(run_name, [exc])
                totals["skipped"] += 1


def _select(records: list[dict], case_ids: Sequence[str] | None) -> list[dict]:
    """Records for `case_ids` in that order, or every record unchanged."""
    if case_ids is None:
        return records
    by_case: dict[str, dict] = {}
    for record in records:
        key = record.get("pmcid") or record.get("question_id")
        by_case.setdefault(key, record)
    missing = [c for c in case_ids if c not in by_case]
    if missing:
        raise ValueError(f"{len(missing)} case ids are not in the responses file: {missing[:3]}")
    return [by_case[c] for c in case_ids]


def _seed(base: int | None, trace: BaseTrace, boundary_index: int, first_index: int) -> int | None:
    """Sampling seed for one request, distinct per trace, boundary and top-up."""
    if base is None:
        return None
    stream = int(trace.trace_id[:8], 16)
    return (base + stream + _SEED_STRIDE * boundary_index + first_index) % _SEED_MODULUS


def _to_rollouts(
    trace: BaseTrace, boundary_index: int, first_index: int, output, seed, tokenizer
) -> list[Rollout]:
    """Rollout records for one request's completions.

    continuation is decoded from the emitted token ids rather than taken from the engine's
    own text. vLLM 0.26 leaves byte-level markers in `completion.text` for the Llama-3
    tokenizer, which survives answer-tag extraction but destroys sentence similarity;
    decoding here keeps text and ids describing the same thing for every tokenizer.
    """
    prefix = trace.prefix_ids(boundary_index)
    digest = token_hash(prefix)
    rollouts = []
    for offset, completion in enumerate(output.outputs):
        token_ids = tuple(int(t) for t in completion.token_ids)
        rollouts.append(
            Rollout(
                case_id=trace.case_id,
                trace_id=trace.trace_id,
                boundary_index=boundary_index,
                rollout_index=first_index + offset,
                prefix_token_count=len(prefix),
                prefix_sha256=digest,
                output_token_ids=token_ids,
                continuation=tokenizer.decode(token_ids, skip_special_tokens=False),
                finish_reason=completion.finish_reason,
                seed=seed,
            )
        )
    return rollouts


def run(
    *,
    model: str,
    run_name: str,
    n_cases: int,
    n_rollouts: int,
    seed: int | None = 0,
    case_ids: Sequence[str] | None = None,
    sample_source: str | None = None,
    sample_picks: dict[str, Sequence[tuple[int, bool | None]]] | None = None,
    only_boundaries: Sequence[int] | None = None,
    max_sentences: int | None = None,
    expect_decoding: dict | None = None,
    max_model_len: int | None = None,
    max_new_tokens: int | None = None,
    min_new_tokens: int = MIN_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
    engine_kwargs: dict | None = None,
) -> dict:
    """Generate `n_rollouts` continuations at every boundary of `n_cases` traces.

    `n_cases` counts traces swept, which equals cases only when one trace is taken per case.
    Passing `sample_picks` sweeps the named samples of `sample_source` instead of the stored
    response, so a case can contribute more than one.
    """
    cfg = load_model(model)
    assert_decoding(cfg, expect_decoding)

    records = generation.load_existing(generation.responses_path(cfg.slug))
    if not records:
        raise RuntimeError(f"no responses at {generation.responses_path(cfg.slug)}")

    engine, tokenizer = generation.load_model(cfg, **(engine_kwargs or {}))
    context = context_length(engine, max_model_len)
    base_params = generation.sampling_params(cfg)

    store.write_manifest(run_name, _manifest(cfg, tokenizer))
    store.append_run(
        run_name,
        {
            "n_cases": n_cases,
            "n_rollouts": n_rollouts,
            "seed": seed,
            "case_ids": list(case_ids) if case_ids is not None else None,
            "only_boundaries": list(only_boundaries) if only_boundaries is not None else None,
            "max_sentences": max_sentences,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "max_model_len": context,
        },
    )

    stored = store.progress(run_name)
    already = store.trace_ids(run_name)
    totals = {"cases": 0, "skipped": 0, "boundaries": 0, "rollouts": 0, "over_context": 0}

    # Pending work is collected across every case first, so one engine call can hold prompts
    # from many traces. Batching per case starves the GPU when a case owes few boundaries.
    work: list[tuple[BaseTrace, store.Pending]] = []
    prepared = _traces_to_sweep(
        records, case_ids, cfg, tokenizer, run_name, totals,
        sample_source=sample_source, sample_picks=sample_picks,
    )
    for trace in prepared:
        if totals["cases"] >= n_cases:
            break

        if max_sentences is not None and len(trace.sentences) > max_sentences:
            store.log_skips(
                run_name,
                [
                    boundaries.BoundaryError(
                        trace.case_id, f"{len(trace.sentences)} sentences over max {max_sentences}"
                    )
                ],
            )
            totals["skipped"] += 1
            continue

        totals["cases"] += 1
        if trace.trace_id not in already:
            store.append_traces(run_name, [trace])
            already.add(trace.trace_id)

        owed = store.pending(trace, stored, n_rollouts)
        if only_boundaries is not None:
            wanted = set(only_boundaries)
            owed = [p for p in owed if p.boundary_index in wanted]
        fits = [p for p in owed if len(trace.prefix_ids(p.boundary_index)) + min_new_tokens <= context]
        totals["over_context"] += len(owed) - len(fits)
        work.extend((trace, item) for item in fits)

    print(
        f"{run_name}: {totals['cases']} cases, {len(work)} boundaries pending, "
        f"{sum(item.n_needed for _, item in work):,} rollouts to generate",
        flush=True,
    )

    ceiling = min(cfg.decoding.max_new_tokens, max_new_tokens or cfg.decoding.max_new_tokens)
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        prompts, params, meta = [], [], []
        for trace, item in batch:
            prefix = trace.prefix_ids(item.boundary_index)
            item_seed = _seed(seed, trace, item.boundary_index, item.first_rollout_index)
            request = copy.copy(base_params)
            request.n = item.n_needed
            request.max_tokens = min(ceiling, context - len(prefix))
            request.seed = item_seed
            prompts.append({"prompt_token_ids": list(prefix)})
            params.append(request)
            meta.append((trace, item, item_seed))

        outputs = engine.generate(prompts, params)

        # A part file holds one case, so split a mixed batch before writing.
        by_case: dict[str, list[Rollout]] = {}
        for output, (trace, item, item_seed) in zip(outputs, meta):
            by_case.setdefault(trace.case_id, []).extend(
                _to_rollouts(
                    trace, item.boundary_index, item.first_rollout_index, output, item_seed, tokenizer
                )
            )
        for produced in by_case.values():
            store.append_rollouts(run_name, produced)

        totals["boundaries"] += len(batch)
        totals["rollouts"] += sum(len(v) for v in by_case.values())
        print(
            f"  {min(start + batch_size, len(work))}/{len(work)} boundaries, "
            f"{totals['rollouts']:,} rollouts",
            flush=True,
        )

    print(
        f"\n{run_name}: {totals['cases']} cases, {totals['boundaries']} boundaries, "
        f"{totals['rollouts']} rollouts, {totals['skipped']} traces skipped, "
        f"{totals['over_context']} boundaries over context",
        flush=True,
    )
    return totals
