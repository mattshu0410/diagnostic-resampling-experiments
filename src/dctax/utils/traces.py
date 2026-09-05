"""Reasoning-trace extraction and sentence splitting.

Each model wraps its chain of thought in a different envelope. `extract` tries them in a fixed
order and reports which one matched:

    envelope         models
    harmony          gpt-oss-20b
    answer_anchored  huatuogpt-o1-8b, qwq-32b
    think_block      deepseek-r1-distill-{qwen-1.5b, llama-8b, qwen-14b}, glm-4.7-flash
    gemma_thought    gemma-4-31b-it
    outside_tags     any model that reasons after the prompt's template but outside <think>
    thinking_section any model that emits a "## Thinking" header
    fallback         whole response, with answer blocks and chat markup removed

Order matters: huatuo and qwq also emit <think> blocks, but their reasoning begins at the
prompt's echoed answer placeholder, so answer_anchored is tried first. The deepseek distills
reach think_block because their chat prefix is stripped, which removes the echoed placeholder.

A model whose reasoning the order would misattribute names its envelope instead, through
`expected_trace_format` in its config. The named envelope is tried first and the order still
runs when it matches nothing. gemma needs this: it echoes the prompt's answer placeholder, so
the order reaches answer_anchored and returns the wrong span, while its reasoning is in a
thought channel that only gemma_thought reads.

Sentences are split under the ruleset a model's config names: medspaCy's PyRuSH clinical
sentencizer, or a markdown splitter for the outline-style traces; see `split_into_sentences`.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Extraction(NamedTuple):
    """Reasoning text and the name of the envelope it came from."""

    text: str
    trace_format: str


# --------------------------------------------------------------------------------------
# Shared cleanup
# --------------------------------------------------------------------------------------

# Terminal EOS and padding tokens, retained when responses are decoded with special tokens
# kept. Removing them lets the trailing-answer peel see a final "</answer>".
_TRAILING_SPECIAL = re.compile(
    r"(?:\s*(?:<｜end▁of▁sentence｜>|<\|(?:eot_id|end_of_text|endoftext|im_end|return|end)\|>))+\s*\Z"
)
# huatuogpt-o1-8b labels its answer section "## Final Response"; spacing, case and line
# endings vary, so NBSP and CRLF forms are matched.
_HUATUO_FINAL_RESPONSE = re.compile(
    r"(?m)^[ \t ]*##\s*Final\s+Response\b[ \t ]*\r?$|##\s*Final\s+Response\b\s*",
    flags=re.IGNORECASE,
)
# huatuogpt-o1-8b leaks Llama-3 turn scaffolding ahead of its reasoning, sometimes doubled.
# Its reasoning begins immediately after a near-leading "## Thinking".
_HUATUO_TURN_HEADER = re.compile(r"<\|start_header_id\|>.*?<\|end_header_id\|>", flags=re.DOTALL)
_HUATUO_THINKING_LEAD = re.compile(r"^.{0,40}?##\s*Thinking\s*\n+", flags=re.DOTALL)
_ANY_SPECIAL_TOKEN = re.compile(r"<\|[^>]*\|>")
_LEADING_ROLE_WORD = re.compile(r"^\s*assistant\b\s*")
# A dangling answer tag followed by a short bare answer at the very end, as in
# "</answer>\nPTSD\n</answer>" once the outer closer has been peeled.
_BARE_ANSWER_TAIL = re.compile(r"(?:</?answer>)\s*[^<>\n]{1,160}\Z", flags=re.IGNORECASE)


def _strip_trailing_answer(text: str) -> str:
    """Peel trailing answer markup off the end of a reasoning span.

    Removes one tag at a time from the end. A balanced <answer>...</answer> block is dropped
    whole; a dangling closer with no matching opener drops the closer plus any short bare
    answer wedged before it. The opener is the nearest preceding one with no closer between.
    """
    if not text:
        return text
    s = text.rstrip()
    original = s
    while s.endswith("</answer>"):
        close = s.rfind("</answer>")
        opener = s.rfind("<answer>", 0, close)
        if opener != -1 and "</answer>" not in s[opener + len("<answer>"):close]:
            s = s[:opener].rstrip()
        else:
            s = s[:close].rstrip()
            mb = _BARE_ANSWER_TAIL.search(s)
            if mb:
                bare = re.match(r"</?answer>", s[mb.start():])
                s = s[: mb.start() + bare.end()].rstrip()
    return s if s != original else text


def _cleanup(text: str | None) -> str:
    """Remove chat scaffolding, special tokens and section headers from a reasoning span."""
    if not text:
        return ""
    cleaned = _HUATUO_FINAL_RESPONSE.sub("", text)
    cleaned = cleaned.replace("## Final Response", "")
    cleaned = cleaned.replace("<｜Assistant｜><think>", "")
    cleaned = _HUATUO_TURN_HEADER.sub("", cleaned)
    cleaned = _ANY_SPECIAL_TOKEN.sub("", cleaned)
    cleaned = _HUATUO_THINKING_LEAD.sub("", cleaned)
    cleaned = _LEADING_ROLE_WORD.sub("", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


# --------------------------------------------------------------------------------------
# harmony envelope: gpt-oss-20b
# --------------------------------------------------------------------------------------

# The analysis channel carries the chain of thought; the user-facing answer is a separate
# final channel. With special tokens retained the channel markers survive verbatim. With them
# stripped, the channel names collapse onto the role word, leaving "assistantanalysis" and
# "assistantfinal" as the only boundary markers.
_HARMONY_TOKENS = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|<\|return\|>|\Z)",
    flags=re.DOTALL,
)
_HARMONY_GLUED = re.compile(r"assistantanalysis(.*?)assistantfinal", flags=re.DOTALL)
_HARMONY_GLUED_OPEN = re.compile(r"assistantanalysis(.*)\Z", flags=re.DOTALL)


def _extract_harmony(response: str) -> str | None:
    """Analysis-channel reasoning, or None when no harmony markers are present."""
    if "<|channel|>analysis<|message|>" not in response and "assistantanalysis" not in response:
        return None
    m = (
        _HARMONY_TOKENS.search(response)
        or _HARMONY_GLUED.search(response)
        or _HARMONY_GLUED_OPEN.search(response)
    )
    return m.group(1) if m is not None else None


# --------------------------------------------------------------------------------------
# answer_anchored envelope: huatuogpt-o1-8b, qwq-32b
# --------------------------------------------------------------------------------------

# These models echo the prompt's answer placeholder, then begin reasoning after it.
_PLACEHOLDER_ANSWER = re.compile(
    r"<answer>\s*\.\.\.the name of the disease/entity\.\.\.\s*</answer>", flags=re.IGNORECASE
)
# qwq-32b closes its reasoning with "</think>" followed by a tool-call block; the intervening
# whitespace varies, so the boundary is matched whitespace-tolerantly.
_QWQ_TOOL_CALL_CUT = re.compile(r"\n</think>\s*<tool_call>")


def _extract_answer_anchored(text: str) -> str | None:
    """Reasoning following the echoed answer placeholder, or None when there is no echo."""
    m = _PLACEHOLDER_ANSWER.search(text)
    if not m:
        return None
    tail = text[m.end():]
    cut = _QWQ_TOOL_CALL_CUT.search(tail)
    if cut:
        tail = tail[: cut.start()]
    tail = tail.replace("\n\n<think>\n", "")
    tail = _strip_trailing_answer(tail).strip()
    return tail or None


# --------------------------------------------------------------------------------------
# think_block envelope: deepseek-r1-distill-{qwen-1.5b, llama-8b, qwen-14b}
# --------------------------------------------------------------------------------------

# These traces carry a doubled assistant boundary: the first echoes the prompt's output
# template, the second is where generation begins. Anchoring on the last one drops the echo.
_DEEPSEEK_ASSISTANT_BOUNDARY = re.compile(
    r"</answer>\s*<[^>]*assistant[^>]*><think>\n", flags=re.IGNORECASE
)
_DEEPSEEK_ASSISTANT_SUFFIX = re.compile(
    r"</answer>\s*<[^>]*assistant[^>]*><think>\n?", flags=re.IGNORECASE
)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_TEMPLATE_PLACEHOLDERS = ("...", "...your internal reasoning for the diagnosis...")


def _strip_chat_prefix(text: str) -> str:
    """Drop everything up to the last assistant turn boundary."""
    if not text:
        return text
    matches = list(_DEEPSEEK_ASSISTANT_BOUNDARY.finditer(text))
    if not matches:
        return text
    last = matches[-1]
    think_pos = text.find("<think>\n", last.start())
    return text[think_pos:] if think_pos != -1 else text[last.end():]


def _think_blocks(text: str) -> list[str]:
    """Contents of every <think> block, excluding the prompt's own placeholder text."""
    return [
        t for t in (b.strip() for b in _THINK_BLOCK.findall(text))
        if t not in _TEMPLATE_PLACEHOLDERS
    ]


# --------------------------------------------------------------------------------------
# outside_tags / thinking_section / fallback
# --------------------------------------------------------------------------------------

_THINKING_SECTION = re.compile(
    r"##\s*Thinking\s*\n(.*?)(?:##|<answer>|\Z)", flags=re.IGNORECASE | re.DOTALL
)
_SHORT_ANSWER = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_RULE_LINE = re.compile(r"^\s*[-]+\s*$", flags=re.MULTILINE)
_CASE_HEADING = re.compile(r"^CASE PRESENTATION.*?(?=\n\n|\Z)", flags=re.DOTALL)


def _extract_outside_tags(text: str) -> str:
    """Reasoning emitted after the prompt's output template but outside any <think> block."""
    for marker in ("</answer>\n", "</answer>"):
        oti = text.find("OUTPUT TEMPLATE")
        if oti == -1:
            continue
        te = text.find(marker, oti)
        if te == -1:
            continue
        after = text[te + len(marker):]
        last_think = after.rfind("<think>")
        if last_think <= 100:
            continue
        ot = after[:last_think].strip()
        if "<|" in ot or "|>" in ot:
            continue
        ot = _RULE_LINE.sub("", ot)
        ot = _CASE_HEADING.sub("", ot).strip()
        if len(ot) >= 100:
            return ot
    return ""


# --------------------------------------------------------------------------------------
# gemma_thought / think_longest: models that also emit a restatement in the prompt's tags
# --------------------------------------------------------------------------------------

# gemma-4 writes its reasoning into a named channel and a shorter restatement into the
# prompt's <think> tags.
_GEMMA_THOUGHT = re.compile(r"<\|channel>thought\s*(.*?)(?:<channel\|>|\Z)", flags=re.DOTALL)


def _extract_gemma_thought(response: str) -> str | None:
    """Contents of gemma's thought channel, or None when the channel is absent."""
    m = _GEMMA_THOUGHT.search(response)
    if m is None:
        return None
    return m.group(1).strip() or None


def _extract_think_longest(response: str) -> str | None:
    """Longest <think> block, taken whole."""
    blocks = _think_blocks(response)
    if not blocks:
        return None
    return max(blocks, key=len).strip() or None


# ministral opens reasoning with its own [THINK] control token but closes with the </think>
# the prompt asked for, so the pair never matches; [/THINK] appears once in 2073 responses. In
# about one response in six it opens with a plain <think> instead, so that is accepted as an
# opener too. The prompt echoes <think> in its own instructions, so the search starts after the
# [/INST] that closes the prompt, never in it. The reasoning then runs from the opener to
# whichever of the restatement's <think>, a </think>, or the answer comes first -- or to the
# end when generation hit the token cap.
_MISTRAL_PROMPT_END = re.compile(r"\[/INST\]")
_BRACKET_THINK_OPEN = re.compile(r"\[THINK\]|<think>", flags=re.IGNORECASE)
_BRACKET_THINK_END = re.compile(r"\[/THINK\]|<think>|</think>|<answer>", flags=re.IGNORECASE)


def _extract_bracket_think(response: str) -> str | None:
    """Reasoning between ministral's opener and the restatement that follows it."""
    prompt_end = _MISTRAL_PROMPT_END.search(response)
    start = prompt_end.end() if prompt_end else 0
    opener = _BRACKET_THINK_OPEN.search(response, start)
    if opener is None:
        return None
    rest = response[opener.end():]
    end = _BRACKET_THINK_END.search(rest)
    return (rest[: end.start()] if end else rest).strip() or None


def _strip_short_answers(text: str) -> str:
    """Remove <answer> blocks shorter than 500 characters, leaving longer ones in place."""
    matches = list(_SHORT_ANSWER.finditer(text))
    if not matches:
        return text
    result = text
    for m in reversed(matches):
        if len(m.group(1)) < 500:
            result = result[: m.start()] + result[m.end():]
    return result


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def _select_harmony(response: str) -> str | None:
    return _extract_harmony(response)


def _select_answer_anchored(response: str) -> str | None:
    return _extract_answer_anchored(_strip_chat_prefix(response))


def _select_think_block(response: str) -> str | None:
    blocks = _think_blocks(_strip_chat_prefix(response))
    return _strip_trailing_answer(max(blocks, key=len)) if blocks else None


def _select_gemma_thought(response: str) -> str | None:
    return _extract_gemma_thought(response)


def _select_think_longest(response: str) -> str | None:
    return _extract_think_longest(_strip_chat_prefix(response))


def _select_bracket_think(response: str) -> str | None:
    return _extract_bracket_think(response)


FORMATS = {
    "harmony": _select_harmony,
    "answer_anchored": _select_answer_anchored,
    "think_block": _select_think_block,
    "gemma_thought": _select_gemma_thought,
    "think_longest": _select_think_longest,
    "bracket_think": _select_bracket_think,
}


def extract(response: str, question: str = "", trace_format: str | None = None) -> Extraction:
    """Extract the reasoning trace from a full response, with the envelope that matched.

    Naming a `trace_format` tries that envelope first. When it is not named, or yields
    nothing, the envelopes are tried in order and the first that matches is used.
    """
    if not response:
        return Extraction("", "empty")

    response = _TRAILING_SPECIAL.sub("", response)

    if trace_format is not None:
        if trace_format not in FORMATS:
            raise ValueError(f"unknown trace_format {trace_format!r}; known: {sorted(FORMATS)}")
        selected = FORMATS[trace_format](response)
        if selected:
            return Extraction(_cleanup(selected), trace_format)

    harmony = _extract_harmony(response)
    if harmony is not None:
        return Extraction(_cleanup(harmony), "harmony")

    text = _strip_chat_prefix(response)

    anchored = _extract_answer_anchored(text)
    if anchored is not None:
        return Extraction(_cleanup(anchored), "answer_anchored")

    candidates = [("think_block", t) for t in _think_blocks(text)]
    outside = _extract_outside_tags(text)
    if outside:
        candidates.append(("outside_tags", outside))
    if candidates:
        label, best = max(candidates, key=lambda c: len(c[1]))
        return Extraction(_cleanup(_strip_trailing_answer(best)), label)

    section = _THINKING_SECTION.search(text)
    if section:
        content = section.group(1).strip()
        if len(content) >= 50:
            return Extraction(_cleanup(_strip_trailing_answer(content)), "thinking_section")

    resp = text
    if question and question in resp:
        resp = resp.replace(question, "")
    stripped = _strip_short_answers(resp).strip()
    cleaned = _strip_trailing_answer(stripped if stripped else resp)
    cleaned = _DEEPSEEK_ASSISTANT_SUFFIX.sub("", cleaned)
    return Extraction(_cleanup(cleaned), "fallback")


def extract_reasoning(response: str, question: str = "", trace_format: str | None = None) -> str:
    """Reasoning trace text from a full response.

    Naming a `trace_format` tries that envelope first and falls back to the ordered chain when
    it matches nothing, so a model whose reasoning the chain would misattribute can name its
    own.
    """
    return extract(response, question, trace_format=trace_format).text


# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------

_PYRUSH_NLP = None

_WORD = re.compile(r"[0-9A-Za-z]")
_LINE_BREAK = re.compile(r"\n")
# A run of terminator and emphasis/bracket characters at a token boundary, in either order --
# ".**", "?)", "**.", "**?", ")?", '".', '"?'. PyRuSH ends a sentence on a terminator followed
# by a space, but not on one where markup sits between the terminator and the space, or between
# the word and the terminator, so these run two thoughts into one row. A boundary is added when
# the run carries both a terminator and a markup char (a bare terminator is left to PyRuSH, so
# its abbreviation handling for "E. coli" and "vs." is untouched) and is not an ellipsis.
_TERM_MARKUP = re.compile(r"""[.!?"')\]*`]+(?=\s|\Z)""")
_TERM_CHAR = re.compile(r"[.!?]")
_MARKUP_CHAR = re.compile(r"""["')\]*`]""")

# Boundaries PyRuSH draws that the outline undoes, so a fragment PyRuSH split off rejoins the
# sentence before it. A sentence never really ends on either of these:
#   _GENUS         a lone capital + period ("E.", "H.", "S.") when what follows starts
#                  lowercase -- an abbreviated genus, as in "E. coli", "S. aureus". The
#                  lowercase test keeps a true end like "... 140 U/L. Lipase ..." split, since
#                  that continues with a capital.
#   _ELLIPSIS_END  a trailing "..." or "…".
# A "vs." is left to PyRuSH, which splits it, as the clinical rules do.
_GENUS = re.compile(r"(?:^|[^A-Za-z])[A-Z]\.\Z")
_ELLIPSIS_END = re.compile(r"(?:\.\.+|…)\Z")

# A terminator followed by whitespace and then either an opening bracket/emphasis char -- "(",
# "[", "*", "`", a quote -- or a capital letter: the start of a new sentence, parenthetical or
# emphasised span that PyRuSH runs into the one before. PyRuSH ends on "." and "?" before a
# capital but misses them before a bracket ("midbrain. (Resolved ...", "criteria. *Bold ..."),
# misses a period after a non-word token ("T1DM. However"), and never ends on "!" at all
# ("TTP! The patient ..."); this covers all three the same way. A terminator preceded by a dot
# is skipped so an ellipsis is not a boundary, a decimal ("2.5 (") never matches because its dot
# has no following whitespace, and a lowercase follower is left alone so "E. coli" stays whole.
_TERM_OPEN = re.compile(r"""(?<!\.)[.!?](?=\s+[(\[*`"'A-Z])""")
# A piece that is only a list marker -- "1.", "a)", "*   2." -- keeps the item it introduces
# rather than standing alone.
_LIST_MARKER = re.compile(r"^[-*•+]?\s*(?:\d+|[A-Za-z])[.)]$")

RULESETS = ("clinical", "outline")


def _pyrush_nlp():
    """Blank English pipeline carrying medspaCy's PyRuSH sentencizer, built once."""
    global _PYRUSH_NLP
    if _PYRUSH_NLP is None:
        from loguru import logger as _loguru_logger
        from PyRuSH import PyRuSHSentencizer  # noqa: F401: registers the pipeline factory
        from spacy.lang.en import English

        _loguru_logger.disable("PyRuSH")
        _PYRUSH_NLP = English()
        _PYRUSH_NLP.add_pipe("medspacy_pyrush")
    return _PYRUSH_NLP


def _outline(text: str) -> list[str]:
    """The clinical split, with each line break also a sentence boundary.

    gemma, glm, qwen3.6 and ministral write markdown outlines, not the clinical prose PyRuSH is
    tuned for. A bullet rarely ends in a period, so PyRuSH runs consecutive list items together
    into one unit of thousands of characters -- no sentence, and one that hides the repetition
    `taxonomy.degenerate` counts once a looping trace collapses into it. So the outline ruleset
    takes the boundaries PyRuSH draws and adds three: one at every line break, one after a
    terminator glued to markup that PyRuSH runs through (`_TERM_MARKUP`), and one after a
    terminator followed by an opening bracket or emphasis char that starts a parenthetical
    PyRuSH runs into the sentence before (`_TERM_OPEN`). It then undoes two PyRuSH draws in the
    wrong place: a split inside a genus abbreviation ("E. coli", "S. aureus") and a split after
    an ellipsis. A "vs." is left split, as PyRuSH and the clinical rules do it.

    A fragment carrying no word -- a stray "**" left by emphasis, a "```" fence on its own line
    -- joins the sentence beside it rather than becoming a row of pure punctuation; a genus or
    ellipsis fragment rejoins the sentence before it (`_GENUS`, `_ELLIPSIS_END`); and a bare list
    marker ("1.", "* 2.") keeps the item it introduces (`_LIST_MARKER`). Every cut only slices,
    so each sentence stays a verbatim substring of the trace and the sentences still tile it,
    which `spans.locate_sentences` relies on.
    """
    doc = _pyrush_nlp()(text)
    edges = {0, len(text)}
    edges.update(m.end() for m in _LINE_BREAK.finditer(text))
    edges.update(sent.start_char for sent in doc.sents if sent.start_char)
    edges.update(
        m.end()
        for m in _TERM_MARKUP.finditer(text)
        if _TERM_CHAR.search(m.group()) and _MARKUP_CHAR.search(m.group()) and ".." not in m.group()
    )
    edges.update(m.end() for m in _TERM_OPEN.finditer(text))
    pieces: list[tuple[int, int]] = []
    for start, end in zip(sorted(edges), sorted(edges)[1:]):
        if not text[start:end].strip():
            continue
        if pieces:
            previous = text[pieces[-1][0]:pieces[-1][1]].rstrip()
            current = text[start:end]
            if (
                not _WORD.search(current)
                or not _WORD.search(previous)
                or _ELLIPSIS_END.search(previous)
                or (_GENUS.search(previous) and current.lstrip()[:1].islower())
                or _LIST_MARKER.match(previous)
            ):
                pieces[-1] = (pieces[-1][0], end)
                continue
        pieces.append((start, end))
    return [text[start:end].strip() for start, end in pieces]


def split_into_sentences(text: str, rules: str = "clinical") -> list[str]:
    """Split reasoning text into sentences, dropping empty ones.

    `rules` names the ruleset, as a model's config does through `sentence_rules`. "clinical" is
    medspaCy's PyRuSH, which is what every model collected before the 2026 run was split under
    and what the six clinical models are still split under, untouched. "outline" is that same
    split with line breaks added as boundaries, for the four models that write markdown outlines;
    see `_outline`. Every returned sentence is a verbatim substring of `text`, so
    `spans.locate_sentences` can find it.
    """
    if rules == "outline":
        return _outline(text)
    if rules == "clinical":
        return [s.text.strip() for s in _pyrush_nlp()(text).sents if s.text.strip()]
    raise ValueError(f"unknown ruleset {rules!r}; known: {list(RULESETS)}")
