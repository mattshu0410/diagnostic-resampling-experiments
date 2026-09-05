"""Records written by the resampling experiment.

Boundary B_i is the token index a rollout generates from, so a trace with M sentences has
M + 1 boundaries: removing S_i generates from B_i, keeping it generates from B_(i+1).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


def content_hash(text: str) -> str:
    """Identity of a trace, from its full response."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_hash(token_ids: Sequence[int]) -> str:
    """Identity of a prefix, from the exact token ids sent to the model."""
    joined = ",".join(str(i) for i in token_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SentenceBoundary:
    """One reasoning sentence, located in both the response text and its token sequence.

    token_start is the boundary generating a replacement for this sentence.
    """

    sentence_index: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"sentence {self.sentence_index}: char span "
                f"[{self.char_start}, {self.char_end}) is empty"
            )
        if self.token_end <= self.token_start:
            raise ValueError(
                f"sentence {self.sentence_index}: token span "
                f"[{self.token_start}, {self.token_end}) is empty"
            )

    def as_dict(self) -> dict:
        return {
            "sentence_index": self.sentence_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SentenceBoundary:
        return cls(**payload)


@dataclass(frozen=True)
class BaseTrace:
    """One stored response, with its sentences located in the sequence the model was run on.

    input_ids is that sequence: the rendered prompt and the generation, special tokens kept.
    Every rollout prefix is a slice of it. raw_answer and is_correct stay None until graded.
    """

    model: str
    case_id: str
    trace_id: str
    full_response: str
    prompt_token_count: int
    input_ids: tuple[int, ...]
    sentences: tuple[SentenceBoundary, ...]
    raw_answer: str | None = None
    is_correct: bool | None = None

    def __post_init__(self) -> None:
        if not self.sentences:
            raise ValueError(f"{self.case_id}: trace has no sentences")
        starts = [s.token_start for s in self.sentences]
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError(f"{self.case_id}: sentence token starts are not increasing")
        if starts[0] < self.prompt_token_count:
            raise ValueError(
                f"{self.case_id}: first sentence starts at token {starts[0]}, "
                f"inside the {self.prompt_token_count}-token prompt"
            )
        if self.sentences[-1].token_end > len(self.input_ids):
            raise ValueError(f"{self.case_id}: last sentence ends past the sequence")

    @property
    def n_boundaries(self) -> int:
        """One boundary before each sentence, plus one after the last."""
        return len(self.sentences) + 1

    def prefix_ids(self, boundary_index: int) -> tuple[int, ...]:
        """Token ids to generate from at `boundary_index`.

        Boundary i cuts before sentence i, so it holds every sentence up to i-1. The final
        boundary holds all of them and ends where the reasoning does.
        """
        if not 0 <= boundary_index < self.n_boundaries:
            raise IndexError(
                f"boundary {boundary_index} outside [0, {self.n_boundaries}) for {self.case_id}"
            )
        if boundary_index == len(self.sentences):
            return self.input_ids[: self.sentences[-1].token_end]
        return self.input_ids[: self.sentences[boundary_index].token_start]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "full_response": self.full_response,
            "prompt_token_count": self.prompt_token_count,
            "input_ids": list(self.input_ids),
            "sentences": [s.as_dict() for s in self.sentences],
            "raw_answer": self.raw_answer,
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BaseTrace:
        return cls(
            model=payload["model"],
            case_id=payload["case_id"],
            trace_id=payload["trace_id"],
            # Absent in the published archive, where the source case text is not
            # redistributed. Nothing that indexes sentences reads it.
            full_response=payload.get("full_response", ""),
            prompt_token_count=payload["prompt_token_count"],
            input_ids=tuple(payload["input_ids"]),
            sentences=tuple(SentenceBoundary.from_dict(s) for s in payload["sentences"]),
            raw_answer=payload.get("raw_answer"),
            is_correct=payload.get("is_correct"),
        )


@dataclass(frozen=True)
class Rollout:
    """One continuation generated from one boundary.

    output_token_ids is what the model emitted, so later analysis never has to re-encode
    continuation. The answer and its correctness are derived downstream, not stored here.
    """

    case_id: str
    trace_id: str
    boundary_index: int
    rollout_index: int
    prefix_token_count: int
    prefix_sha256: str
    output_token_ids: tuple[int, ...]
    continuation: str
    finish_reason: str | None = None
    seed: int | None = None

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity used to resume an interrupted run."""
        return (self.trace_id, self.boundary_index, self.rollout_index)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "boundary_index": self.boundary_index,
            "rollout_index": self.rollout_index,
            "prefix_token_count": self.prefix_token_count,
            "prefix_sha256": self.prefix_sha256,
            "output_token_ids": list(self.output_token_ids),
            "continuation": self.continuation,
            "finish_reason": self.finish_reason,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Rollout:
        return cls(
            case_id=payload["case_id"],
            trace_id=payload["trace_id"],
            boundary_index=payload["boundary_index"],
            rollout_index=payload["rollout_index"],
            prefix_token_count=payload["prefix_token_count"],
            prefix_sha256=payload["prefix_sha256"],
            output_token_ids=tuple(payload["output_token_ids"]),
            continuation=payload["continuation"],
            finish_reason=payload.get("finish_reason"),
            seed=payload.get("seed"),
        )


def rollouts_at_boundary(rollouts: Iterable[Rollout], boundary_index: int) -> list[Rollout]:
    """Rollouts generated from one boundary, in rollout order."""
    selected = [r for r in rollouts if r.boundary_index == boundary_index]
    return sorted(selected, key=lambda r: r.rollout_index)
