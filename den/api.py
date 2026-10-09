"""Labelled System One requests: the training and evaluation record format.

A record is a `/v1/systemone` request plus a label on every question:

    {"state": ..., "questions": {"team": {"type": "choice", "criteria": {...}, "label": "billing"}}}

Labels are the option name (choice), a bool (noul) or a level index (score). An optional `target` gives a soft
distribution over the same keys. By default a target marks an unknowable item (no right answer, left out of accuracy);
with `"target_from": "teacher"` it is a teacher's distribution whose argmax is the label, so the question is trained
on the distribution and still scored on the label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type Json = str | int | float | bool | list[Json] | dict[str, Json] | None
type QuestionType = Literal["choice", "score", "noul"]

MAX_OPTIONS = 255


class RecordError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    type: QuestionType
    instructions: str
    keys: tuple[str, ...]
    options: tuple[str, ...]
    label: int
    target: tuple[float, ...] | None
    source: str
    teacher: bool = False  # the target is a teacher's distribution, not an unknowable item's

    @property
    def unknowable(self) -> bool:
        return self.target is not None and not self.teacher


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    state: str
    questions: tuple[Question, ...]
    provenance: str | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.state)


def render(value: Json, indent: int = 0) -> str:
    """Must match Kev's `api.render` byte for byte."""
    pad = "  " * indent
    match value:
        case None:
            return ""
        case str() | int() | float() | bool():
            return str(value)
        case list():
            return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in value)
        case dict():
            return "\n".join(
                f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, dict | list) else f"{pad}{k}: {render(x)}"
                for k, x in value.items()
            )


def option_text(name: str, description: Json) -> str:
    return name if description is None or description == "" else f"{name}: {render(description)}"


def squash(text: str) -> str:
    """Casefolded, with every run of whitespace collapsed to one space."""
    return " ".join(text.casefold().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(squash(text).encode()).hexdigest()


def question_key(state: str, question: Question) -> str:
    """A question's identity: its state, wording and options."""
    return fingerprint("\x1f".join((state, question.instructions, *question.options)))


def _question(qid: str, raw: Json) -> Question:
    if not isinstance(raw, dict):
        raise RecordError(f"question {qid!r} is not an object")
    qtype, criteria, label = raw.get("type"), raw.get("criteria"), raw.get("label")
    source = raw.get("src")
    match qtype:
        case "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
                raise RecordError(f"{qid}: choice needs 1..{MAX_OPTIONS} named options")
            if any(not k.strip() for k in criteria):
                raise RecordError(f"{qid}: empty option name")
            keys = tuple(criteria)
            options = tuple(option_text(k, v) for k, v in criteria.items())
            if not isinstance(label, str) or label not in criteria:
                raise RecordError(f"{qid}: label {label!r} is not an option")
            index = keys.index(label)
        case "score":
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_OPTIONS:
                raise RecordError(f"{qid}: score needs 1..{MAX_OPTIONS} levels")
            keys = tuple(str(i) for i in range(len(criteria)))
            options = tuple(render(x) for x in criteria)
            if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(criteria):
                raise RecordError(f"{qid}: label {label!r} is not a level index")
            index = label
        case "noul":
            described = criteria or {}
            if not isinstance(described, dict) or set(described) - {"true", "false"}:
                raise RecordError(f"{qid}: noul criteria may only describe 'true' and 'false'")
            keys = ("false", "true")
            options = (option_text("no", described.get("false")), option_text("yes", described.get("true")))
            if not isinstance(label, bool):
                raise RecordError(f"{qid}: noul label must be a bool, got {label!r}")
            index = int(label)
        case _:
            raise RecordError(f"{qid}: unknown question type {qtype!r}")
    if len(set(options)) != len(options):
        raise RecordError(f"{qid}: two options render to the same text")
    instructions = render(raw.get("instructions"))
    if not instructions.strip():
        raise RecordError(f"{qid}: empty instructions")
    target = _target(qid, raw.get("target"), keys)
    teacher = raw.get("target_from") == "teacher"
    if teacher and (target is None or max(target) > target[index] + 1e-9):
        raise RecordError(f"{qid}: a teacher target needs its label among the most likely options")
    return Question(qid, qtype, instructions, keys, options, index, target, str(source), teacher)


def _target(qid: str, raw: Json, keys: tuple[str, ...]) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - set(keys):
        raise RecordError(f"{qid}: target must map option keys to weights")
    weights = [raw.get(k, 0.0) for k in keys]
    if not all(isinstance(w, int | float) and not isinstance(w, bool) and w >= 0 for w in weights):
        raise RecordError(f"{qid}: target weights must be non-negative numbers")
    total = float(sum(w for w in weights if isinstance(w, int | float)))
    if total <= 0:
        raise RecordError(f"{qid}: target puts no mass on any option")
    return tuple(float(w) / total for w in weights if isinstance(w, int | float))


def parse(raw: Json, record_id: str, provenance: str | None = None) -> Record:
    if not isinstance(raw, dict):
        raise RecordError("record is not an object")
    if "state" not in raw:
        raise RecordError("record has no state")
    questions = raw.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RecordError("record needs a non-empty questions object")
    state = render(raw["state"])
    if not state.strip():
        raise RecordError("empty state")
    return Record(record_id, state, tuple(_question(qid, q) for qid, q in questions.items()), provenance)


def read(path: Path, lines: Collection[int] | None = None) -> Iterator[Record]:
    """Every record in a JSONL file, or only those on the given 1-based `lines` (the rest are never parsed)."""
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or (lines is not None and n not in lines):
                continue
            try:
                raw: Json = json.loads(line)
                meta = raw.get("_meta") if isinstance(raw, dict) else None
                meta = meta if isinstance(meta, dict) else {}
                origin = meta.get("text_sha256")
                yield parse(raw, str(meta.get("id") or f"{path.stem}:{n}"), origin if isinstance(origin, str) else None)
            except (RecordError, json.JSONDecodeError) as e:
                raise RecordError(f"{path}:{n}: {e}") from e
