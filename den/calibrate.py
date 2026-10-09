"""Temperature scaling, fitted on the calibration split only (never dev or test).

T is bounded to [1/20, 20] (an unbounded fit on a small or badly wrong set runs to infinity) and found by a grid over
log T then golden-section refinement: deterministic, with no line search to fail where the loss goes flat."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional

LOG_T_BOUND = math.log(20)
GRID = 121  # coarse points over [-bound, bound]: steps of 0.05 in log T
REFINE = 40  # golden-section iterations within the best grid cell: width 0.1 * 0.618^40, far below any use


def calibration_nll(logits: torch.Tensor, goal: torch.Tensor, log_t: float) -> float:
    """Mean NLL of softmax(logits / T) against `goal` (one-hot or soft) rows; -inf logits mark padded options."""
    logp = functional.log_softmax(logits / math.exp(log_t), dim=-1).nan_to_num(neginf=0.0)
    return float(-(goal * logp).sum(-1).mean())


def fit_temperature(
    scores: Sequence[torch.Tensor], labels: Sequence[int], targets: Sequence[Sequence[float] | None]
) -> float:
    """The T in [1/20, 20] that minimises calibration NLL of softmax(scores / T). Fit on the calibration split only."""
    if not scores:
        return 1.0
    width = max(len(s) for s in scores)
    logits = torch.full((len(scores), width), -torch.inf)
    goal = torch.zeros(len(scores), width)
    for i, (s, label, target) in enumerate(zip(scores, labels, targets, strict=True)):
        logits[i, : len(s)] = s.detach().float().cpu()
        if target is None:
            goal[i, label] = 1.0
        else:
            goal[i, : len(s)] = torch.tensor(target)

    def loss(log_t: float) -> float:
        return calibration_nll(logits, goal, log_t)

    grid = [-LOG_T_BOUND + 2 * LOG_T_BOUND * i / (GRID - 1) for i in range(GRID)]
    values = [loss(x) for x in grid]
    best = min(range(GRID), key=values.__getitem__)
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, GRID - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = loss(a), loss(b)
    for _ in range(REFINE):
        if fa <= fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = loss(a)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = loss(b)
    candidates = [(values[best], grid[best]), (fa, a), (fb, b)]
    return math.exp(min(candidates)[1])
