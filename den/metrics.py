"""Every evaluation number, computed in one place from per-question probabilities.

`den evaluate`, the calibration report written during training, and `den overfit` all call `summarize`, so the same
question always gets the same score. Unknowable questions (a soft target with no right answer) are counted, not
scored. Teacher questions (a soft target whose argmax is the label) are scored on the label, and their distance from
the teacher's distribution is reported as `kl_to_target`.
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
    target: Sequence[float] | None = None  # soft target: an unknowable item's, or a teacher's
    teacher: bool = False  # the target is a teacher's distribution: the label is still graded

    @property
    def unknowable(self) -> bool:
        return self.target is not None and not self.teacher


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
    hard = [a for a in answers if not a.unknowable]
    graded: list[tuple[float, bool]] = []
    nll = brier = level_error = rps = 0.0
    by_kind: dict[str, list[tuple[float, bool]]] = {}
    by_source: dict[str, list[bool]] = {}
    scored = 0
    for a in hard:
        p = list(a.probs)
        top = argmax(p)
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
    if taught := [a for a in hard if a.target is not None]:
        result["teacher_questions"] = len(taught)
        result["kl_to_target"] = sum(kl(a.target, a.probs) for a in taught if a.target is not None) / len(taught)
    return result


def kl(target: Sequence[float], probs: Sequence[float]) -> float:
    """KL(target || probs): how far the predicted distribution is from the teacher's."""
    return sum(t * math.log(t / max(p, 1e-12)) for t, p in zip(target, probs, strict=True) if t > 0)


@dataclass(frozen=True, slots=True)
class Row:
    """One scored question with its suite metadata (`_meta`): what Kev's robustness checks pair rows by."""

    answer: Answer
    record: str  # _meta.id
    question: str
    keys: Sequence[str]
    variant: str = "clean"  # clean | permuted | none_present | none_absent | ...
    parent: str | None = None  # the clean record a variant was made from
    pair: str | None = None  # contrastive pair id; `sibling` is "a" or "b"
    sibling: str | None = None
    control: str | None = None  # an unknowable record's intact control
    origin: str = ""  # _meta.source: "unknowable" and "unknowable_control" are scored on confidence


def argmax(p: Sequence[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def robustness(rows: Sequence[Row]) -> dict[str, Any]:
    """Kev's benchmark checks (`kev.benchmark.summarize`), over whatever structure the file carries:
    - `clean`: the headline numbers on clean, knowable rows only (Kev's `clean`; ours above also count the variants);
    - `permutation`: each permuted variant against its clean parent, options aligned by key: the mean largest
      probability change and how often the answer changes (Kev's option-order changes);
    - `paired_flip`: contrastive pairs whose answers differ must flip, pairs whose answers agree must not;
    - `unknowable`: confidence where the deciding evidence was removed, against the intact controls;
    - `variants`: accuracy per variant."""
    clean = [r for r in rows if r.variant == "clean"]
    out: dict[str, Any] = {}
    if len(clean) != len(rows):
        out["clean"] = summarize([r.answer for r in clean if r.origin != "unknowable"])
        by_variant: dict[str, list[bool]] = {}
        for r in rows:
            if not r.answer.unknowable:
                by_variant.setdefault(r.variant, []).append(argmax(r.answer.probs) == r.answer.label)
        out["variants"] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_variant.items())}
    lookup = {(r.record, r.question): r for r in clean}
    deltas, flips = [], []
    for r in rows:
        if r.variant == "permuted" and r.answer.kind == "choice" and (o := lookup.get((r.parent or "", r.question))):
            aligned = [r.answer.probs[list(r.keys).index(k)] for k in o.keys]
            deltas.append(max(abs(a - b) for a, b in zip(aligned, o.answer.probs, strict=True)))
            flips.append(argmax(aligned) != argmax(o.answer.probs))
    if deltas:
        out["permutation"] = {
            "n": len(deltas),
            "mean_max_delta": sum(deltas) / len(deltas),
            "flip_rate": sum(flips) / len(flips),
        }
    if (pairs := paired_flip(clean)) is not None:
        out["paired_flip"] = pairs
    if (unknown := unknowable(clean)) is not None:
        out["unknowable"] = unknown
    return out


def paired_flip(rows: Sequence[Row]) -> dict[str, Any] | None:
    """Kev's `paired_flip`: among contrastive pairs (two states differing in what decides), those whose true answers
    differ should get different answers (`flip_rate`), those whose answers agree the same one (`invariance_rate`). A
    model that ignores the state can't flip. Pairs missing a sibling (a sampled file) are counted, not scored."""
    by_pair: dict[tuple[str, str], dict[str, Row]] = {}
    for r in rows:
        if r.pair and r.sibling:
            by_pair.setdefault((r.pair, r.question), {})[r.sibling] = r
    if not by_pair:
        return None
    whole = [p for p in by_pair.values() if set(p) == {"a", "b"}]
    truth = [(p, p["a"].keys[p["a"].answer.label], p["b"].keys[p["b"].answer.label]) for p in whole]

    def said(r: Row) -> str:
        return r.keys[argmax(r.answer.probs)]

    def both(ps: list[dict[str, Row]]) -> float | None:
        return sum(all(said(r) == r.keys[r.answer.label] for r in p.values()) for p in ps) / len(ps) if ps else None

    relevant = [p for p, a, b in truth if a != b]
    invariant = [p for p, a, b in truth if a == b]
    result: dict[str, Any] = {
        "pairs": len(relevant),
        "flip_rate": sum(said(p["a"]) != said(p["b"]) for p in relevant) / len(relevant) if relevant else None,
        "both_correct_rate": both(relevant),
        "incomplete_pairs": len(by_pair) - len(whole),
    }
    if invariant:
        result |= {
            "invariant_pairs": len(invariant),
            "invariant_both_correct_rate": both(invariant),
            "invariance_rate": sum(said(p["a"]) == said(p["b"]) for p in invariant) / len(invariant),
        }
    return result


def unknowable(rows: Sequence[Row]) -> dict[str, Any] | None:
    """Kev's `unknowable_report`: on records whose deciding evidence was removed, accuracy means nothing; what counts is
    whether the model knows it can't know (mean top probability, share answered at >= 0.9), against intact controls."""
    unk = [r for r in rows if r.origin == "unknowable"]
    if not unk:
        return None
    ctl = [r for r in rows if r.origin == "unknowable_control"]

    def conf(rs: Sequence[Row]) -> list[float]:
        return [max(r.answer.probs) for r in rs]

    def mean(xs: Sequence[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    by_id = {r.record: r for r in ctl}
    paired = [(max(r.answer.probs), max(by_id[r.control].answer.probs)) for r in unk if r.control in by_id]
    return {
        "n": len(unk),
        "mean_max_p": mean(conf(unk)),
        "share_at_0_9": mean([c >= 0.9 for c in conf(unk)]),
        "control_mean_max_p": mean(conf(ctl)),
        "control_share_at_0_9": mean([c >= 0.9 for c in conf(ctl)]),
        "control_acc": mean([argmax(r.answer.probs) == r.answer.label for r in ctl]),
        "paired_confidence_drop": mean([c - u for u, c in paired]),
        "share_less_confident_than_control": mean([u < c for u, c in paired]),
    }


BOOTSTRAP = 2000  # resamples for paired confidence intervals (Kev's kev.compare uses 2000)


def paired_bootstrap(
    groups: Sequence[str], a: Sequence[float], b: Sequence[float], resamples: int = BOOTSTRAP, seed: int = 0
) -> dict[str, float]:
    """Mean of b - a over paired questions, with a 95% percentile interval from resampling whole groups (records):
    questions sharing a state are not independent, so they are drawn together. Seeded, so the interval reproduces."""
    import random

    sums: dict[str, list[float]] = {}
    for g, x, y in zip(groups, a, b, strict=True):
        s = sums.setdefault(g, [0.0, 0.0])
        s[0] += y - x
        s[1] += 1
    cells = list(sums.values())
    n = sum(c[1] for c in cells)
    diff = sum(c[0] for c in cells) / max(n, 1)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples if cells else 0):
        picked = [cells[rng.randrange(len(cells))] for _ in cells]
        draws.append(sum(c[0] for c in picked) / max(sum(c[1] for c in picked), 1))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))] if draws else diff
    high = draws[int(0.975 * (len(draws) - 1))] if draws else diff
    return {"diff": diff, "low": low, "high": high, "questions": n, "groups": len(cells)}
