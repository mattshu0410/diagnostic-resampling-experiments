"""Mapping sentences to token spans.

Activations are produced per token, while the analysis unit is a sentence, so each sentence
must be located as a range of token positions in the sequence the model was run on.

`encode_with_offsets` returns the token ids and their character offsets from a single
tokenizer call. Both must come from the same call: the offsets index the ids, and any
difference in tokenization arguments between them shifts every span.
"""
from __future__ import annotations

from typing import NamedTuple


class Encoding(NamedTuple):
    """Token ids and the character range each token covers.

    input_ids  tensor of shape (1, n_tokens)
    offsets    list of (char_start, char_end) per token, aligned with input_ids
    """

    input_ids: object
    offsets: list[tuple[int, int]]


def encode_with_offsets(text: str, tokenizer, *, max_length: int | None = None) -> Encoding:
    """Tokenize `text` without adding special tokens, returning ids and character offsets.

    Special tokens are not added because the text already contains the chat template's own,
    rendered verbatim. Adding more would prepend a token absent from the sequence the offsets
    are meant to index.
    """
    kwargs = {
        "return_offsets_mapping": True,
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs.update(truncation=True, max_length=max_length)
    enc = tokenizer(text, **kwargs)
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
    return Encoding(enc["input_ids"], offsets)


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    """Token range covering the character range [char_start, char_end).

    Returns (token_start, token_end) with token_end exclusive, or (None, None) when no token
    overlaps the range. Tokens with an empty character range, such as special tokens, are
    skipped.
    """
    if char_start is None or char_end is None or char_end <= char_start:
        return None, None

    indices = [
        i for i, (s, e) in enumerate(offsets) if e > s and s < char_end and e > char_start
    ]
    if not indices:
        return None, None
    return min(indices), max(indices) + 1


def locate_sentences(
    text: str, sentences: list[str], offsets: list[tuple[int, int]], start_at: int = 0
) -> list[tuple[int, int] | None]:
    """Token span for each sentence, or None where no valid span exists.

    Sentences are searched for in order from a cursor that advances past each match, so
    repeated sentences map to successive occurrences rather than all to the first. The cursor
    starts at `start_at`, which should be where the reasoning begins, so that identical text
    in the prompt is not matched.
    """
    cursor = max(0, start_at)
    spans: list[tuple[int, int] | None] = []
    for sentence in sentences:
        pos = text.find(sentence, cursor)
        if pos >= 0:
            cursor = pos + len(sentence)
        else:
            pos = text.find(sentence)
        if pos < 0:
            spans.append(None)
            continue
        start, end = char_span_to_token_span(offsets, pos, pos + len(sentence))
        spans.append((start, end) if start is not None and end is not None and start < end else None)
    return spans
