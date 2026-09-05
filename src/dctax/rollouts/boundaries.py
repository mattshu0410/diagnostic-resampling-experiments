"""Turning a stored response into validated rollout boundaries.

Sentences are identified in the cleaned reasoning but addressed in the token sequence the
model was run on, so the two must agree exactly. Anything that would make a prefix cut
inexact raises BoundaryError rather than being recovered from.
"""
from __future__ import annotations

import re
from typing import Sequence

from dctax.config import ModelConfig
from dctax.generate.responses import build_prompts
from dctax.rollouts.schema import BaseTrace, SentenceBoundary, content_hash
from dctax.utils import answers
from dctax.utils.spans import char_span_to_token_span, encode_with_offsets
from dctax.utils.traces import extract_reasoning, split_into_sentences

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class BoundaryError(ValueError):
    """A trace whose sentences cannot be addressed as exact token cuts."""

    def __init__(self, case_id: str, reason: str) -> None:
        super().__init__(f"{case_id}: {reason}")
        self.case_id = case_id
        self.reason = reason


def _case_id(record: dict) -> str:
    return record.get("pmcid") or record.get("question_id") or "<unknown>"


def _prompt_token_count(
    full: str, question: str, offsets: list[tuple[int, int]], cfg: ModelConfig, tokenizer
) -> int:
    """Tokens covering the rendered prompt, which every prefix keeps."""
    case_id = "<prompt>"
    prompt = build_prompts(tokenizer, [question], cfg)[0]
    # gpt-oss's harmony template stamps the render date, which differs from generation day.
    if full[: len(prompt)] != prompt and _ISO_DATE.sub("<d>", full[: len(prompt)]) != _ISO_DATE.sub(
        "<d>", prompt
    ):
        raise BoundaryError(case_id, "rendered prompt is not a prefix of full_response")

    for index, (start, end) in enumerate(offsets):
        if start >= len(prompt):
            if index and offsets[index - 1][1] > len(prompt):
                raise BoundaryError(case_id, "a token spans the prompt boundary")
            return index
    raise BoundaryError(case_id, "no generation after the prompt")


def _locate(
    full: str,
    sentences: list[str],
    offsets: list[tuple[int, int]],
    start_at: int,
    region: tuple[int, int],
    case_id: str,
) -> list[SentenceBoundary]:
    """Locate each sentence, walking a cursor so repeats map to successive occurrences.

    Unlike utils.spans.locate_sentences there is no search from position zero when the cursor
    finds nothing, because that match can land in the prompt rather than the generation.
    """
    region_start, region_end = region
    cursor = start_at
    located: list[SentenceBoundary] = []

    for index, sentence in enumerate(sentences):
        char_start = full.find(sentence, cursor)
        if char_start < 0:
            raise BoundaryError(case_id, f"sentence {index} not found after char {cursor}")
        char_end = char_start + len(sentence)
        cursor = char_end

        token_start, token_end = char_span_to_token_span(offsets, char_start, char_end)
        if token_start is None:
            raise BoundaryError(case_id, f"sentence {index} covers no tokens")
        if token_start < region_start or token_end > region_end:
            raise BoundaryError(case_id, f"sentence {index} falls outside the reasoning")

        straddle = full[offsets[token_start][0] : char_start]
        if straddle and not straddle.isspace():
            raise BoundaryError(case_id, f"sentence {index} starts inside {straddle!r}")

        located.append(
            SentenceBoundary(index, sentence, char_start, char_end, token_start, token_end)
        )
    return located


def prepare_sample(
    base: BaseTrace, prefix_token_count: int, continuation: str,
    record: dict, cfg: ModelConfig, tokenizer,
) -> BaseTrace:
    """The trace one sampled continuation stands for, rebuilt from the prefix it grew from.

    The prefix text is sliced out of the base trace at the character where its token
    `prefix_token_count` begins, and never decoded back from ids: a decode and re-encode round
    trip shifts the join by a token often enough to matter.

    That count comes from the rollout itself rather than from `prompt_token_count`, because a
    boundary cuts before its sentence, not after the prompt. Anything the model emits ahead of
    its first sentence - gpt-oss opening an `analysis` channel, a `<think>` tag - belongs to the
    prefix, and dropping it leaves the trace format with nothing to key on.

    The result is a full response of the shape `prepare_trace` already takes, and its content
    hash gives the sample its own trace_id.
    """
    offsets = encode_with_offsets(base.full_response, tokenizer).offsets
    if not 0 < prefix_token_count < len(offsets):
        raise BoundaryError(
            base.case_id, f"prefix of {prefix_token_count} tokens is outside the base trace"
        )
    cut = offsets[prefix_token_count][0]
    return prepare_trace(
        {**record, "full_response": base.full_response[:cut] + continuation}, cfg, tokenizer
    )


def prepare_trace(record: dict, cfg: ModelConfig, tokenizer) -> BaseTrace:
    """Validated trace for `record`, or raise BoundaryError naming what failed."""
    case_id = _case_id(record)
    full = record.get("full_response") or ""
    if not full:
        raise BoundaryError(case_id, "empty full_response")

    encoding = encode_with_offsets(full, tokenizer)
    input_ids = tuple(encoding.input_ids[0].tolist())

    reasoning = extract_reasoning(full, record.get("question", ""), cfg.expected_trace_format)
    if not reasoning:
        raise BoundaryError(case_id, "no reasoning extracted")
    start_at = full.find(reasoning)
    if start_at < 0:
        raise BoundaryError(case_id, "reasoning is not verbatim in full_response")

    region = char_span_to_token_span(encoding.offsets, start_at, start_at + len(reasoning))
    if region[0] is None:
        raise BoundaryError(case_id, "reasoning covers no tokens")

    sentences = split_into_sentences(reasoning, cfg.sentence_rules)
    if not sentences:
        raise BoundaryError(case_id, "no sentences in reasoning")

    located = _locate(full, sentences, encoding.offsets, start_at, region, case_id)

    try:
        prompt_tokens = _prompt_token_count(
            full, record.get("question", ""), encoding.offsets, cfg, tokenizer
        )
        return BaseTrace(
            model=cfg.slug,
            case_id=case_id,
            trace_id=content_hash(full),
            full_response=full,
            prompt_token_count=prompt_tokens,
            input_ids=input_ids,
            sentences=tuple(located),
            raw_answer=answers.extract(full, cfg.answer_fallbacks) or None,
            is_correct=record.get("is_correct"),
        )
    except BoundaryError as exc:
        raise BoundaryError(case_id, exc.reason) from exc
    except ValueError as exc:
        raise BoundaryError(case_id, str(exc)) from exc


def prepare_traces(
    records: list[dict], cfg: ModelConfig, tokenizer
) -> tuple[list[BaseTrace], list[BoundaryError]]:
    """Every trace that validates, and the errors for those that do not."""
    prepared: list[BaseTrace] = []
    skipped: list[BoundaryError] = []
    for record in records:
        try:
            prepared.append(prepare_trace(record, cfg, tokenizer))
        except BoundaryError as exc:
            skipped.append(exc)
    return prepared, skipped
