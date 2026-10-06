"""One forward pass per record: the state, then each question with its options, tokenized to fixed positions.

    <state>

    Question: <instructions>
    Options:
    - <option 1>
    - <option 2>
    Answer:

The pointer head scores each option by its span (from `-` to the newline that closes it) against the question's final
token (the colon of `Answer:`). The prompt is plain text, tokenized once like any other string, and those positions
are found from the tokenizer's character offsets, so a server only has to send `text(record)` through the tokenizer.
This layout is this repo's own: Kev's serving template is not in the pinned files, so a Kev head will not drop in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Literal

from tokenizers import Tokenizer

from .api import MAX_OPTIONS, Question, Record, option_text

MAX_STATE_TOKENS = 7552  # Kev's training limit


@dataclass(frozen=True, slots=True)
class Example:
    ids: tuple[int, ...]
    finals: tuple[int, ...]  # per question: index of its final token
    closes: tuple[tuple[int, ...], ...]  # per question: index of each option's closing token
    starts: tuple[tuple[int, ...], ...]  # per question: index of each option's first token
    labels: tuple[int, ...]
    targets: tuple[tuple[float, ...] | None, ...]
    ordered: tuple[bool, ...] = ()  # per question: a score question, whose options are ordered levels
    kinds: tuple[str, ...] = ()  # per question: choice | noul | score
    sources: tuple[str, ...] = ()  # per question: its `src` tag (dataset or skill family), for breakdowns


@dataclass(frozen=True, slots=True)
class Layout:
    """The prompt text and the character positions the pointer head reads."""

    text: str
    state_end: int
    finals: tuple[int, ...]  # per question: the character of its `Answer:` colon
    starts: tuple[tuple[int, ...], ...]  # per question: each option's `-`
    closes: tuple[tuple[int, ...], ...]  # per question: each option's closing newline


type Style = Literal["dash", "letters"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def layout(record: Record, style: Style = "dash") -> Layout:
    """`dash` (the pointer head's layout): `- option` lines. `letters`: `A) option` lines, for the letter scorer, which
    reads Qwen's own next-token distribution over " A", " B", ... after `Answer:` (at most 26 options)."""
    parts, finals, starts, closes = [record.state], [], [], []
    size = len(record.state)

    def add(piece: str) -> int:
        nonlocal size
        parts.append(piece)
        size += len(piece)
        return size

    for q in record.questions:
        add(f"\n\nQuestion: {q.instructions}\nOptions:\n")
        first, last = [], []
        for i, option in enumerate(q.options):
            first.append(size)
            last.append(add(f"{LETTERS[i]}) {option}\n" if style == "letters" else f"- {option}\n") - 1)
        finals.append(add("Answer:") - 1)
        starts.append(tuple(first))
        closes.append(tuple(last))
    return Layout("".join(parts), len(record.state), tuple(finals), tuple(starts), tuple(closes))


def text(record: Record, style: Style = "dash") -> str:
    """The exact string the model reads for this record."""
    return layout(record, style).text


def encode(
    record: Record, tokenizer: Tokenizer, max_state: int = MAX_STATE_TOKENS, style: Style = "dash"
) -> Example | None:
    """None when the state is longer than `max_state` tokens, or (letters) a question has more than 26 options."""
    if style == "letters" and any(len(q.options) > len(LETTERS) for q in record.questions):
        return None
    plan = layout(record, style)
    encoding = tokenizer.encode(plan.text, add_special_tokens=False)
    first = [-1] * len(plan.text)  # character -> first token covering it
    last = [-1] * len(plan.text)  # character -> last token covering it (a character can span byte-level tokens)
    for i, (a, b) in enumerate(encoding.offsets):
        for c in range(a, b):
            if first[c] < 0:
                first[c] = i
            last[c] = i
    if last[plan.state_end - 1] + 1 > max_state:
        return None
    return Example(
        tuple(encoding.ids),
        tuple(last[c] for c in plan.finals),
        tuple(tuple(last[c] for c in q) for q in plan.closes),
        tuple(tuple(first[c] for c in q) for q in plan.starts),
        tuple(q.label for q in record.questions),
        tuple(q.target for q in record.questions),
        tuple(q.type == "score" for q in record.questions),
        tuple(q.type for q in record.questions),
        tuple(q.source for q in record.questions),
    )


def split(record: Record) -> list[Record]:
    """One record per question, each with the full state: Kev's "one row per question". A question is scored on the
    state alone, never on the other questions asked with it, so an answer doesn't depend on how a request is split."""
    return [replace(record, questions=(q,)) for q in record.questions]


def shuffle_options(record: Record, rng: random.Random) -> Record:
    """The record with each choice question's options in a random order, label and target remapped.

    Some sources list options in a fixed order that tracks the label (CFPB issues in `documents` put the answer first
    in 92% of 3-option questions, in dev and test too), so a pointer can score well by position alone. Score levels
    are ordered and noul is always (no, yes), so only `choice` is shuffled."""
    return replace(record, questions=tuple(_shuffled(q, rng) if q.type == "choice" else q for q in record.questions))


# Kev's augmentation (kev/data.py: NONE_OPTIONS, DISTRACTORS, augment, none_pair), on parsed records.
NONE_OPTIONS = (
    ("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
    ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
    ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
    ("other", "An answer not covered by the other options"), ("no_match", "No option matches"),
)  # fmt: skip
DISTRACTORS = (
    ("weather", "Bad weather caused it"), ("purple", "The colour purple"), ("pancakes", "A recipe for pancakes"),
    ("taxes", "Unrelated: quarterly tax filing"),
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Augment:
    """Per choice question, exclusive chances: replace the answer with a "none" option (it becomes the answer), add a
    "none" option as a wrong alternative, or add an irrelevant distractor. Kev's defaults. Order is always shuffled."""

    none: float = 0.10
    none_distract: float = 0.12
    distract: float = 0.15


def _with_options(q: Question, keys: list[str], options: list[str], label: int) -> Question:
    return replace(q, keys=tuple(keys), options=tuple(options), label=label)


def _shuffled(q: Question, rng: random.Random) -> Question:
    order = list(range(len(q.options)))
    rng.shuffle(order)
    return replace(
        q,
        keys=tuple(q.keys[i] for i in order),
        options=tuple(q.options[i] for i in order),
        label=order.index(q.label),
        target=None if q.target is None else tuple(q.target[i] for i in order),
    )


def _free(pool: tuple[tuple[str, str | None], ...], q: Question) -> list[tuple[str, str | None]]:
    """Pool entries whose key and rendered text are both new to the question."""
    return [(k, d) for k, d in pool if k not in q.keys and option_text(k, d) not in q.options]


KEV_AUGMENT = Augment()


def augment(record: Record, rng: random.Random, p: Augment = KEV_AUGMENT) -> Record:
    """Kev's choice-question augmentation. Soft-target questions are only shuffled: adding or removing an option
    would change what the target means. Score and noul questions are left alone."""

    def one(q: Question) -> Question:
        if q.type != "choice":
            return q
        if q.target is not None or len(q.options) < 2:
            return _shuffled(q, rng)
        keys, options, label = list(q.keys), list(q.options), q.label
        r = rng.random()
        nones, distractors = _free(NONE_OPTIONS, q), _free(DISTRACTORS, q)
        if len(keys) > 2 and r < p.none and nones:
            k, d = rng.choice(nones)
            del keys[label], options[label]
            keys.append(k)
            options.append(option_text(k, d))
            label = len(keys) - 1
        elif p.none <= r < p.none + p.none_distract and len(keys) < MAX_OPTIONS and nones:
            k, d = rng.choice(nones)
            keys.append(k)
            options.append(option_text(k, d))
        elif p.none + p.none_distract <= r < p.none + p.none_distract + p.distract and len(keys) < MAX_OPTIONS:
            if distractors:
                k, d = rng.choice(distractors)
                keys.append(k)
                options.append(option_text(k, d))
        return _shuffled(_with_options(q, keys, options, label), rng)

    return replace(record, questions=tuple(one(q) for q in record.questions))


type Perturbation = Literal["none-replace", "none-add", "distract"]


def perturb(record: Record, rng: random.Random, mode: Perturbation) -> Record:
    """One of Kev's augmentations applied to every eligible choice question, for evaluation: `none-replace` swaps the
    true option for a "none" option, which becomes the answer (needs 3+ options); `none-add` adds a "none" option as
    a wrong answer; `distract` adds an irrelevant option. Order is shuffled; other questions are left alone."""
    p = {"none-replace": Augment(1, 0, 0), "none-add": Augment(0, 1, 0), "distract": Augment(0, 0, 1)}[mode]
    return augment(record, rng, p)


def none_pair(record: Record, rng: random.Random) -> tuple[Record, Record] | None:
    """Kev's minimal pair: one choice question, twice, in the same order with the same added "none" option: once with
    the true option present (none is wrong), once with it removed (none is right). The only difference the model can
    use is whether the evidence matches an option. None when no choice question has >= 3 options and a hard label."""
    eligible = [q for q in record.questions if q.type == "choice" and len(q.options) >= 3 and q.target is None]
    if not eligible:
        return None
    q = rng.choice(eligible)
    k, d = rng.choice(_free(NONE_OPTIONS, q) or [("none_of_these", None)])
    present = _shuffled(_with_options(q, [*q.keys, k], [*q.options, option_text(k, d)], q.label), rng)
    gone = present.label
    keys = [x for i, x in enumerate(present.keys) if i != gone]
    options = [x for i, x in enumerate(present.options) if i != gone]
    absent = _with_options(present, keys, options, keys.index(k))
    return replace(record, questions=(present,)), replace(record, questions=(absent,))
