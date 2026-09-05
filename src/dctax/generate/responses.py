"""Generating chain-of-thought responses with vLLM.

Each case is rendered through the cohort's prompt and then through the model's own chat
template, and the stored response is the prompt and the generation joined, decoded with special
tokens intact. Collection teacher-forces that exact string back through the model, so anything
dropped here moves the token spans that activations are pooled over.

    dctax generate --model qwq-32b --cohort main2073

Writing is incremental and a rerun resumes: cases already present in the output file are
skipped, which is also how one file comes to hold several cohorts.
"""
from __future__ import annotations

import json
from pathlib import Path

from dctax.config import CohortConfig, ModelConfig, data_root
from dctax.data.cohort import Case, questions

def responses_path(slug: str) -> Path:
    """Where one model's responses are read from and written to."""
    return data_root() / "responses" / f"responses_{slug}.json"


def load_existing(path: Path) -> list[dict]:
    """Records already generated, or an empty list."""
    if not path.exists():
        return []
    with path.open() as handle:
        return json.load(handle)


def build_prompts(tokenizer, texts: list[str], cfg: ModelConfig) -> list[str]:
    """Each question wrapped in the model's chat template, ready to generate from."""
    # setdefault, not a keyword: a config that names enable_thinking itself would otherwise
    # collide with it and raise.
    kwargs = dict(cfg.decoding.chat_template_kwargs)
    kwargs.setdefault("enable_thinking", True)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for text in texts
    ]


def sampling_params(cfg: ModelConfig):
    """vLLM sampling settings from the model's decoding config.

    skip_special_tokens is off so the generation carries the same markers the model emitted.
    """
    from vllm import SamplingParams

    decoding = cfg.decoding
    if not decoding.do_sample:
        return SamplingParams(
            max_tokens=decoding.max_new_tokens,
            temperature=0.0,
            skip_special_tokens=False,
            seed=decoding.seed,
        )
    return SamplingParams(
        max_tokens=decoding.max_new_tokens,
        temperature=decoding.temperature,
        top_p=decoding.top_p if decoding.top_p is not None else 1.0,
        top_k=decoding.top_k if decoding.top_k is not None else -1,
        repetition_penalty=(
            decoding.repetition_penalty if decoding.repetition_penalty is not None else 1.0
        ),
        skip_special_tokens=False,
        seed=decoding.seed,
    )


def load_model(cfg: ModelConfig, **kwargs):
    """The vLLM engine and tokenizer for one model."""
    from transformers import AutoTokenizer
    from vllm import LLM

    if cfg.tokenizer_mode is not None:
        kwargs["tokenizer_mode"] = cfg.tokenizer_mode
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    engine = LLM(model=cfg.hf_id, dtype=cfg.dtype or "auto", **kwargs)
    return engine, tokenizer


def complete(engine, tokenizer, texts: list[str], cfg: ModelConfig) -> list[str]:
    """Each question's prompt and generation joined, prompted as this model's tokenizer needs.

    A checkpoint carrying its own tokenizer_mode has no chat template to render, so the engine
    builds and tokenizes the prompt itself, leaving RequestOutput.prompt unset.
    """
    params = sampling_params(cfg)
    if cfg.tokenizer_mode is not None:
        messages = [[{"role": "user", "content": text}] for text in texts]
        outputs = engine.chat(messages, params)
        native = engine.get_tokenizer()
        return [native.decode(o.prompt_token_ids) + o.outputs[0].text for o in outputs]

    outputs = engine.generate(build_prompts(tokenizer, texts, cfg), params)
    return [o.prompt + o.outputs[0].text for o in outputs]


def record(case: Case, cohort: CohortConfig, question: str, response: str) -> dict:
    """One stored response, in the shape collection and grading read."""
    return {
        "original_message": {"role": "user", "content": question},
        "full_response": response,
        "question_id": f"{case.source}_{case.case_id}",
        "category": cohort.prompt,
        "question": question,
        "gold_answer": case.answer,
        "dataset_name": case.source,
        "pmcid": case.case_id,
    }


def generate(
    cfg: ModelConfig,
    cases: list[Case],
    cohort: CohortConfig,
    engine,
    tokenizer,
    *,
    path: Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Generate for every case not already in the output file, and write the result."""
    path = path or responses_path(cfg.slug)
    done = load_existing(path)
    seen = {r.get("pmcid") for r in done}
    todo = [c for c in cases if c.case_id not in seen]
    if verbose:
        print(f"{cfg.slug}: {len(todo)} to generate, {len(done)} already present", flush=True)
    if not todo:
        return done

    texts = questions(todo, cohort)
    responses = complete(engine, tokenizer, texts, cfg)

    for case, question, full in zip(todo, texts, responses):
        if question not in full:
            raise ValueError(f"{case.case_id}: the question is not present in the response")
        done.append(record(case, cohort, question, full))

    return write(path, done)


def write(path: Path, records: list[dict]) -> list[dict]:
    """Write records to `path` through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(records, handle)
    tmp.replace(path)
    return records
