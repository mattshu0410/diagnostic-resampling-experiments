"""Forced answers at every sentence boundary of a sweep's base traces.

A rollout continues the reasoning from a boundary; a forced answer stops it there. The prompt
is the token-id slice a rollout would start from, followed by the model's own handoff from
reasoning to answer, so each boundary yields the diagnosis the model gives when made to answer
at that depth. Answers are read and validated as rollouts are, then placed in the sweep's
entities by `grade.grade_forced`, which leaves the rollout grades untouched.

    dctax force        --config screen1000 --model <slug>
    dctax grade-forced --config screen1000
"""
from __future__ import annotations

import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dctax.config import ResampleConfig, load_model
from dctax.generate import responses as generation
from dctax.rollouts import store
from dctax.rollouts.generate import _seed
from dctax.utils import answers

#: Enough for a diagnosis and its closing tag. Generation does not stop at the tag, so a model
#: that answers and keeps talking reaches the cap with its answer intact.
MAX_TOKENS = 64

#: Boundaries per engine call. Every prompt carries its whole prefix, so this bounds host memory.
CHUNK = 300

#: The reasoning-to-answer handoff each model emits, read off its own traces.
FORCING = {
    "deepseek-r1-distill-qwen-1.5b": "\n</think>\n\n<answer>",
    "deepseek-r1-distill-llama-8b": "\n</think>\n\n<answer>",
    "deepseek-r1-distill-qwen-14b": "\n</think>\n\n<answer>",
    "qwq-32b": "\n</think>\n\n<answer>\n",
    "glm-4.7-flash": "</think><answer>\n",
    "gemma-4-31b-it": "\n</think><answer>",
    # Huatuo hands off with a header rather than </think>, then writes its own answer tags.
    "huatuogpt-o1-8b": "\n\n## Final Response\n\n",
    # gpt-oss answers only after switching from the harmony analysis channel to final.
    "gpt-oss-20b": "<|end|><|start|>assistant<|channel|>final<|message|><think></think><answer>\n",
}

SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("trace_id", pa.string()),
        ("boundary_index", pa.int32()),
        ("rollout_index", pa.int32()),
        ("raw_answer", pa.string()),
        ("continuation", pa.string()),
        ("finish_reason", pa.string()),
        ("valid", pa.bool_()),
    ]
)


def run(experiment: ResampleConfig, model: str) -> Path:
    """`sweep_rollouts` forced answers at every boundary of every trace the sweep stored."""
    cfg = load_model(model)
    run_name = experiment.sweep_run(model)
    traces = store.load_traces(run_name)
    engine, tokenizer = generation.load_model(cfg)
    suffix = FORCING[model]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    base = generation.sampling_params(cfg)
    base.max_tokens, base.n = MAX_TOKENS, experiment.sweep_rollouts

    work = [(trace, b) for trace in traces for b in range(trace.n_boundaries)]
    rows = []
    # Rows are written once, at the end, so an interrupted run starts the model again.
    for start in range(0, len(work), CHUNK):
        batch = work[start : start + CHUNK]
        params = [copy.copy(base) for _ in batch]
        for request, (trace, b) in zip(params, batch):
            request.seed = _seed(experiment.seed, trace, b, 0)
        outputs = engine.generate(
            [{"prompt_token_ids": list(trace.prefix_ids(b)) + suffix_ids} for trace, b in batch],
            params,
            use_tqdm=False,
        )
        for (trace, b), output in zip(batch, outputs):
            for index, completion in enumerate(output.outputs):
                # Decoded from the ids rather than the engine's text, as rollouts are.
                text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
                raw = answers.extract(suffix + text, cfg.answer_fallbacks)
                # A rollout is valid when it stops on its own with an answer. The cap here is
                # short on purpose, so an answer whose tag closed has finished too.
                finished = completion.finish_reason == "stop" or "</answer>" in text
                rows.append({
                    "case_id": trace.case_id, "trace_id": trace.trace_id, "boundary_index": b,
                    "rollout_index": index, "raw_answer": raw or None, "continuation": text,
                    "finish_reason": completion.finish_reason, "valid": bool(raw) and finished,
                })
        print(f"  {min(start + CHUNK, len(work))}/{len(work)} boundaries", flush=True)

    path = store.run_dir(run_name) / "forced_answers.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")
    valid = sum(row["valid"] for row in rows)
    print(f"{run_name}: {len(traces)} traces, {len(work)} boundaries, {len(rows):,} forced "
          f"answers, {valid / len(rows):.1%} valid -> {path}", flush=True)
    return path
