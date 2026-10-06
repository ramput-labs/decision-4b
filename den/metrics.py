"""Every evaluation number, computed in one place from per-question probabilities.

`den evaluate`, the calibration report written during training, and `den overfit` all call `summarize`, so the same
question always gets the same score. Soft-target (unknowable) questions have no right answer: they are counted, not
scored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

BINS = 10  # expected calibration error: equal-width confidence bins
RISK = 0.05  # coverage: the share answerable at <= 5% error, most confident first


@dataclass(frozen=True, slots=True)
class Answer:
    probs: Sequence[float]  # one per option, summing to 1
    label: int
    kind: str = "choice"  # choice | noul | score
    source: str = ""  # the question's `src` tag: dataset or skill family
    target: Sequence[float] | None = None  # soft target: an unknowable item


def ece(rows: Sequence[tuple[float, bool]]) -> float:
    """Expected calibration error over (confidence, correct) pairs: the bin-weighted gap between mean confidence
    and accuracy, bins (0, .1], (.1, .2], ... (.9, 1]."""
    total = 0.0
    for b in range(BINS):
        members = [(c, r) for c, r in rows if b / BINS < c <= (b + 1) / BINS or (b == 0 and c == 0)]
        if members:
            gap = sum(c for c, _ in members) / len(members) - sum(r for _, r in members) / len(members)
            total += abs(gap) * len(members) / max(len(rows), 1)
    return total


def coverage(rows: Sequence[tuple[float, bool]], risk: float = RISK) -> float:
    """The largest share of questions, taken most confident first, whose error rate is at most `risk`."""
    covered = wrong = 0
    for i, (_, right) in enumerate(sorted(rows, key=lambda x: -x[0]), 1):
        wrong += not right
        if wrong <= risk * i:
            covered = i
    return covered / max(len(rows), 1)


def summarize(answers: Sequence[Answer]) -> dict[str, Any]:
    """Accuracy, NLL, Brier, ECE and coverage at 5% error over hard-labelled questions; accuracy and ECE per question
    type; accuracy per source; and, for score questions, the expected level's MAE and the ranked probability score."""
    hard = [a for a in answers if a.target is None]
    graded: list[tuple[float, bool]] = []
    nll = brier = level_error = rps = 0.0
    by_kind: dict[str, list[tuple[float, bool]]] = {}
    by_source: dict[str, list[bool]] = {}
    scored = 0
    for a in hard:
        p = list(a.probs)
        top = max(range(len(p)), key=p.__getitem__)
        right = top == a.label
        graded.append((p[top], right))
        nll -= math.log(max(p[a.label], 1e-12))
        brier += sum((x - (i == a.label)) ** 2 for i, x in enumerate(p))
        by_kind.setdefault(a.kind, []).append((p[top], right))
        by_source.setdefault(a.source, []).append(right)
        if a.kind == "score" and len(p) > 1:
            scored += 1
            level_error += abs(sum(i * x for i, x in enumerate(p)) - a.label)
            cdf = [sum(p[: i + 1]) for i in range(len(p) - 1)]
            rps += sum((c - (i >= a.label)) ** 2 for i, c in enumerate(cdf)) / (len(p) - 1)
    n = len(graded)
    result: dict[str, Any] = {
        "questions": n,
        "soft_skipped": len(answers) - n,
        "accuracy": sum(r for _, r in graded) / max(n, 1),
        "nll": nll / max(n, 1),
        "brier": brier / max(n, 1),
        "ece": ece(graded),
        f"coverage_at_{RISK:.0%}_error": coverage(graded),
        "accuracy_by_type": {k: sum(r for _, r in v) / len(v) for k, v in sorted(by_kind.items())},
        "ece_by_type": {k: ece(v) for k, v in sorted(by_kind.items())},
        "accuracy_by_source": {k: sum(v) / len(v) for k, v in sorted(by_source.items()) if k},
    }
    if scored:
        result["score_level_mae"] = level_error / scored
        result["score_rps"] = rps / scored
    return result
