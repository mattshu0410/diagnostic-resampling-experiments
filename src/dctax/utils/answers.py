"""Reading a model's final diagnosis out of its response, for grading.

The prompt asks for the diagnosis inside <answer> tags and the last such block is taken. Models
that skip or misuse the tags need a fallback, so each way of recovering an answer is registered
here by name and each model's config names the ones that apply to it:

    answer_fallbacks: [sandwiched]     # in configs/models/qwq-32b.yaml

    from dctax.utils import answers
    answers.extract(response, cfg.answer_fallbacks)

Adding a model means writing its YAML. It only means touching this file when the model needs a
way of recovering an answer that no existing model needed.

A fallback marked `override` replaces an answer that was already read, rather than only filling
in a blank one.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SANDWICH = re.compile(r"</answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SENTENCE = re.compile(r"(.+?[.!?])(?:\s|$)", re.DOTALL)
BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
OPENER = re.compile(r"<answer>")
# glm-4.7-flash sometimes emits <tool_call>answer> where the prompt asked for <answer>, and
# closes it normally. The tag name after <tool_call> varies and is sometimes missing entirely.
TOOL_CALL_OPENER = re.compile(r"<tool_call>\s*(?:answer|diagnosis)?>?", re.IGNORECASE)
TOOL_CALL_CLOSER = re.compile(r"</(?:answer|diagnosis)>", re.IGNORECASE)

PLACEHOLDER = "...the name of the disease/entity..."
PLACEHOLDER_HINT = "the name of the disease/entity"
# Either slot of the prompt's output template, as echoed back, with the spacing models vary.
PLACEHOLDER_SLOT = re.compile(
    r"\.\.\.\s*(?:the name of the disease/entity|your internal reasoning for the diagnosis)"
    r"\s*\.\.\.",
    re.IGNORECASE,
)
# Markup a diagnosis arrives dressed in, never part of one. Emphasis sits tight against the
# words, so it closes up; a tag can stand between them, so it leaves a space behind.
EMPHASIS = re.compile(r"\*\*|`")
# An ellipsis where the span begins or ends is the template's, not the model's punctuation.
ELLIPSIS_EDGE = re.compile(r"\A\s*(?:\.{2,}|…)\s*|\s*(?:\.{2,}|…)\s*\Z")
# The prompt's own vocabulary for its output format. A span carrying any of it is the model
# talking about where to put the diagnosis, not naming one; no disease is called any of these.
FORMAT_TALK = re.compile(
    r"\b(?:answers?|tags?|template|output requirement|internal reasoning|disease/entity)\b",
    re.IGNORECASE,
)
TAG = re.compile(r"</?[^<>\s]*>?")
WORD = re.compile(r"[A-Za-z0-9]")

# A diagnosis named in prose, and the trailing clause that explains rather than names it.
STATED = re.compile(
    r"(?:most likely|final|primary|definitive|working|overall)\s+diagnosis(?:\s+is|:)"
    r"\s*(?:probably\s+|likely\s+)?"
    r"([^\n.;]{3,120})",
    re.IGNORECASE,
)
STATED_CLAUSE = re.compile(
    r"\s+(?:based on|given|due to|secondary to)\b.*\Z|,\s*(?:which|especially|presenting|and)\b.*\Z",
    re.IGNORECASE,
)
# The same thing said the other way round, and as a bare heading.
STATED_REVERSED = re.compile(
    r"([^\n.;:*]{3,90}?)\s+(?:is|remains)\s+the\s+(?:most likely|final|primary|best)\s+diagnosis",
    re.IGNORECASE,
)
STATED_BARE = re.compile(
    r"(?m)^[\s*#]*(?:final\s+|most likely\s+)?diagnosis\s*[:\-]\s*([^\n.;]{3,120})",
    re.IGNORECASE,
)
# A connective carried into the capture, never part of the name.
LEAD_IN = re.compile(r"\A\s*(?:therefore|thus|so|hence|overall|in summary|given that)\b[,:]?\s*",
                     re.IGNORECASE)
# A capture opening on any of these is the sentence explaining the diagnosis, not naming it.
NOT_A_NAME = re.compile(
    r"\A(?:often|usually|typically|made|based|confirmed|established|suggested|supported|clinical"
    r"|concise|straightforward|clear|not|less|more|a combination|the combination|the case|this case)"
    r"\b",
    re.IGNORECASE,
)
MIN_STATED = 6

# More than this many bold spans after </think> is an enumeration, not an answer.
MAX_BOLD = 3
MAX_BOLD_CHARS = 160
# How far past an opening tag to look for the prompt's echoed placeholder.
HINT_WINDOW = 60


@dataclass(frozen=True)
class Fallback:
    """One way of recovering an answer, and when it applies.

    read      returns the answer it can find, or an empty string
    override  replace an answer already read, rather than only filling a blank one
    """

    read: Callable[[str], str]
    override: bool = False


def _after_think(response: str) -> str:
    index = response.rfind("</think>")
    return response[index + len("</think>"):].strip() if index >= 0 else ""


def _sandwiched(response: str) -> str:
    """The diagnosis left between two closing tags when the opener is missing."""
    found = SANDWICH.findall(response)
    return found[-1].strip() if found else ""


def _prose_after_think(response: str) -> str:
    """The first sentence of the prose a model writes after </think> instead of tagging it."""
    after = _after_think(response)
    if not after:
        return ""
    sentence = SENTENCE.match(after)
    return sentence.group(1).strip() if sentence else after


def _bold_after_think(response: str) -> str:
    """The diagnosis a model emphasises as **bold** after </think>."""
    index = response.rfind("</think>")
    if index < 0:
        return ""
    bolds = BOLD.findall(response[index:])
    if not bolds or len(bolds) > MAX_BOLD:
        return ""
    candidate = bolds[0].strip()
    return candidate if candidate and len(candidate) <= MAX_BOLD_CHARS else ""


def _tool_call_answer(response: str) -> str:
    """The diagnosis behind a <tool_call>answer> opener, which glm-4.7-flash emits for <answer>.

    The closer is the ordinary one, so only the opening tag is malformed and the diagnosis
    inside it is well formed. Read from the last such opener, since the model also opens
    <tool_call>reasoning> blocks ahead of the answer, and only when what it wraps is short
    enough to be a diagnosis rather than a block it forgot to close.
    """
    opens = [m.end() for m in TOOL_CALL_OPENER.finditer(response)]
    if not opens:
        return ""
    rest = response[opens[-1]:]
    close = TOOL_CALL_CLOSER.search(rest)
    return (rest[: close.start()] if close else rest).strip()


def _stated_diagnosis(response: str) -> str:
    """The diagnosis a model names in prose, for responses that open no answer tag at all.

    A model that reasons its way to "the most likely diagnosis is X" and then stops has given
    an answer, just not where it was asked to. The last such phrase is the one it settled on;
    the clause that trails it justifies the choice rather than naming it, so it is cut.
    """
    # Tried in order of how squarely each names the diagnosis rather than merely sitting near
    # it, so the looser patterns only ever add coverage the stricter one did not have.
    found = None
    for pattern in (STATED, STATED_BARE, STATED_REVERSED):
        matches = pattern.findall(response)
        if matches:
            found = matches[-1]
            break
    if found is None:
        return ""
    # "Final diagnosis: based on the above, the most likely diagnosis is X" states it twice.
    # The inner phrase is the one that names it.
    candidate = LEAD_IN.sub("", found)
    inner = STATED.findall(candidate)
    if inner:
        candidate = inner[-1]
    candidate = STATED_CLAUSE.sub("", candidate).strip(" *:-")
    if len(candidate) < MIN_STATED or NOT_A_NAME.match(candidate):
        return ""
    return candidate


def _last_opener(response: str) -> str:
    """Text after the last real opening tag, for responses whose closers are mis-nested."""
    openers = [
        m.end() for m in OPENER.finditer(response)
        if PLACEHOLDER_HINT not in response[m.end():m.end() + HINT_WINDOW]
    ]
    if not openers:
        return ""
    rest = response[openers[-1]:]
    close = rest.find("</answer>")
    return (rest[:close] if close >= 0 else rest).strip()


FALLBACKS: dict[str, Fallback] = {
    "sandwiched": Fallback(_sandwiched),
    "prose_after_think": Fallback(_prose_after_think),
    "bold_after_think": Fallback(_bold_after_think),
    "last_opener": Fallback(_last_opener, override=True),
    "tool_call_answer": Fallback(_tool_call_answer, override=True),
    "stated_diagnosis": Fallback(_stated_diagnosis),
}


def _diagnosis(text: str) -> str:
    """The diagnosis inside a span, or nothing when the span holds no words.

    A reader that lands on an echoed template slot, or on a tag with nothing inside it, has
    found no diagnosis. Saying so lets the readers after it try, instead of standing behind a
    span that is discarded anyway.

    A span longer than a diagnosis ever is has the same answer, as does one still carrying the
    prompt's vocabulary for its own output format -- a reader that ran on into the reasoning, or
    landed on the model discussing where the diagnosis goes, found no diagnosis either.

    What the readers return is otherwise right but dressed: models wrap the diagnosis in a
    heading, in bold, or against a tag they opened and never closed, and one that arrives as
    "<h1>\nMercury poisoning\n</h1>" is the same answer as one that arrives bare. Undressing it
    here means every reader gets it, and it cannot touch an answer that was already clean.
    """
    bare = EMPHASIS.sub("", PLACEHOLDER_SLOT.sub("", text).replace(PLACEHOLDER, ""))
    found = ELLIPSIS_EDGE.sub("", " ".join(TAG.sub(" ", bare).split())).strip()
    if not WORD.search(found) or len(found) > MAX_BOLD_CHARS or FORMAT_TALK.search(found):
        return ""
    return found


def tagged(response: str) -> str:
    """The last <answer> block that reads as a diagnosis, with the echoed placeholder removed.

    The last filled block is not always the answer. A model that keeps generating after it can
    open a later block and fill it with meta-text -- often left unclosed at the token cap -- and
    one that degenerates into repeating "</think><answer>" closes a run of empty blocks after
    the answer it did give. So skip from the end past blocks that hold no diagnosis (empty, the
    echoed template, or a span too long or too format-laden to be a name) and return the first
    that does.
    """
    for block in reversed(ANSWER.findall(response)):
        found = block.strip().replace(PLACEHOLDER, "").strip()
        if found and _diagnosis(found):
            return found
    return ""


def extract(response: str, fallbacks: Sequence[str] = ()) -> str:
    """The model's final diagnosis, or an empty string when none can be read."""
    if not response:
        return ""
    unknown = sorted(set(fallbacks) - set(FALLBACKS))
    if unknown:
        raise ValueError(f"unknown answer fallbacks {unknown}; known: {sorted(FALLBACKS)}")

    # An ellipsis-only span is the prompt's placeholder echoed back. Clearing it as each reader
    # returns, rather than on the way out, lets the fallbacks see the answer as missing, which
    # it is: an override fallback that finds only the placeholder would otherwise stand in front
    # of the ones after it.
    result = _diagnosis(tagged(response))
    for name in fallbacks:
        fallback = FALLBACKS[name]
        if result and not fallback.override:
            continue
        found = _diagnosis(fallback.read(response))
        if found:
            result = found

    return result
