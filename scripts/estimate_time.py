"""Runbook phase 3: how long the real rounds take, from the timing run's measured `tokens_per_second`.

    uv run python -m scripts.estimate_time                         # runs/timing/{1-base,...} and runs/timing/round2
    uv run python -m scripts.estimate_time --budget-hours 5 --spent-minutes 45

The estimate runs high: 20 steps include kernel compilation and each stage's longest batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROUND1 = {"1-base": 6.4e6, "2-dates": 0.6e6, "3-documents": 6.3e6, "4-skills": 9.9e6}  # tokens per stage
ROUND2 = 10.2e6
PER_STAGE = 2  # minutes to load, validate and save
OTHER_PHASES = 80  # minutes for phases 5, 7, 8 and 9


def minutes(run: Path, tokens: float) -> float:
    speed = float(json.loads((run / "run.json").read_text(encoding="utf-8"))["tokens_per_second"])
    return tokens / speed / 60 + PER_STAGE


def plan(round1: float, round2: float | None, budget: float | None) -> str:
    if budget is None:
        return ""
    if round2 is not None and round1 + round2 + OTHER_PHASES <= budget:
        return "decision: run both rounds"
    if round1 + OTHER_PHASES <= budget:
        return "decision: run round 1 only, skip phase 6, and say so in the report"
    return "decision: stop and report the estimate"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.estimate_time")
    p.add_argument("--runs", type=Path, default=Path("runs/timing"), help="the timing run's stage directories")
    p.add_argument("--round2", type=Path, help="the timing run of round 2 (default: RUNS/round2)")
    p.add_argument("--budget-hours", type=float, help="BUDGET_HOURS, to print the phase 3 decision")
    p.add_argument("--spent-minutes", type=float, default=0, help="minutes already spent (phases 0-3)")
    args = p.parse_args(argv)
    round1 = sum(minutes(args.runs / stage, tokens) for stage, tokens in ROUND1.items())
    second = args.round2 or args.runs / "round2"
    round2 = minutes(second, ROUND2) if (second / "run.json").is_file() else None
    budget = None if args.budget_hours is None else 60 * args.budget_hours - args.spent_minutes
    print(
        f"round 1 {round1:.0f} min" + (f", round 2 {round2:.0f} min" if round2 is not None else ", no round 2 timing")
    )
    if decision := plan(round1, round2, budget):
        print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
