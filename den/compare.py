"""Choosing between runs, on dev only: `den compare` (which round ships), `den compare --paired` (question by question,
with 95% intervals) and `den baselines` (modes A/B/C/D side by side, for the model card).

    den compare runs/kev-recipe/4-skills runs/round2
    den compare --paired runs/kev-recipe/4-skills runs/round2
    den baselines D=runs/round2 A=runs/base-a C=runs/base-c
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import evidence
from .catalog import role
from .paths import read_json

MAX_DROP = 0.01  # the largest dev accuracy drop on any file a winner (or a new release) may have
BASELINE_METRICS = ("questions", "accuracy", "nll", "brier", "ece", "coverage_at_5%_error", "ms_per_question")


def compare_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den compare")
    p.add_argument("runs", nargs="+", help="run directories, each evaluated on the same dev files")
    p.add_argument("--files", nargs="+", help="dev files to judge on (default: those every run has)")
    p.add_argument(
        "--max-drop", type=float, default=MAX_DROP, help="largest accuracy drop on any file a winner may have"
    )
    p.add_argument("--paired", action="store_true", help="two runs (dirs, hf:, release:), question by question")
    args = p.parse_args(argv)
    if args.paired:
        return paired_main(args.runs, args.files)
    runs = [Path(r) for r in args.runs]
    reports = {run: read_json(run / "eval.json") for run in runs}
    files = args.files or sorted(set.intersection(*(set(r) for r in reports.values())))
    if not files or any(role(f) == "test" for f in files):
        raise SystemExit("compare on dev files that every run was evaluated on; never on test")
    print(f"{'file':<40}" + "".join(f"{run.name:>16}" for run in runs))
    for f in files:
        print(f"{f:<40}" + "".join(f"{reports[run][f]['accuracy']:>16.4f}" for run in runs))
    mean = {run: sum(reports[run][f]["accuracy"] for f in files) / len(files) for run in runs}
    print(f"{'mean accuracy':<40}" + "".join(f"{mean[run]:>16.4f}" for run in runs))
    best = runs[0]
    for run in runs[1:]:
        drop = max(reports[runs[0]][f]["accuracy"] - reports[run][f]["accuracy"] for f in files)
        if mean[run] > mean[best] and drop <= args.max_drop:
            best = run
    print(f"ship: {best}")
    return 0


def paired_main(runs: list[str], files: list[str] | None) -> int:
    """B against A on the questions both answered (their committed evidence rows), saved beside B's evidence."""
    if len(runs) != 2:
        raise SystemExit("--paired compares exactly two runs: A B (B against A)")
    a, b = runs
    shared = sorted(set(evidence.keys(a)) & set(evidence.keys(b)))
    keys = files or [k for k in shared if role(k) != "test"]  # test only when named
    if not keys:
        raise SystemExit(f"no evaluated file in common: evaluate both on the same files ({shared or 'none'})")
    table = {k: got for k in keys if (got := evidence.paired(a, b, k)) is not None}
    print(f"{'file':<36}{'n':>7}{'acc A':>9}{'acc B':>9}  {'B - A [95% CI]':<29}{'verdict':<21}{'nll B - A':>11}")
    for k, t in table.items():
        acc, nll = t["accuracy"], t["nll"]
        ci = f"{acc['diff']:+.4f} [{acc['ci95'][0]:+.4f}, {acc['ci95'][1]:+.4f}]"
        line = f"{k:<36}{t['questions']:>7}{acc['a']:>9.4f}{acc['b']:>9.4f}  {ci:<29}{acc['b_vs_a']:<21}"
        print(f"{line}{nll['diff']:>+11.4f}")
    target = evidence.EVIDENCE / evidence.name(b) / f"comparison-vs-{evidence.name(a).replace('/', '__')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"a": a, "b": b, "files": table}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {target}")
    return 0


def baselines(runs: dict[str, Path], files: Sequence[str]) -> dict[str, Any]:
    """Every run's numbers on the same files, with its mode read from head.json: A zero-shot Qwen (letters, no LoRA),
    B LoRA + letters, C frozen Qwen + head, D LoRA + head."""
    table: dict[str, Any] = {"files": list(files), "runs": {}}
    for label, run in runs.items():
        evaluated = read_json(run / "eval.json")
        if missing := [f for f in files if f not in evaluated]:
            raise SystemExit(f"{run} was not evaluated on {missing}: den evaluate --run {run} ... first")
        head = read_json(run / "head.json")
        letters, lora = head.get("kind") == "letters", bool(head.get("lora"))
        table["runs"][label] = {
            "path": str(run),
            "mode": ("B" if lora else "A") if letters else ("D" if lora else "C"),
            "head": head.get("kind"),
            "lora": head.get("lora"),
            "temperature": head.get("temperature"),
            "results": {f: {m: evaluated[f].get(m) for m in BASELINE_METRICS} for f in files},
        }
    return table


def markdown(table: dict[str, Any]) -> str:
    """The baselines table, one row per file and run."""
    rows = ["| file | run | mode | questions | accuracy | nll | brier | ece | cov@5% |", "|---" * 9 + "|"]
    nan = float("nan")
    for f in table["files"]:
        for label, r in table["runs"].items():
            m = {k: nan if v is None else v for k, v in r["results"][f].items()}
            rows.append(
                f"| `{f}` | {label} | {r['mode']} | {m['questions']:.0f} | {m['accuracy']:.4f} | {m['nll']:.4f} "
                f"| {m['brier']:.4f} | {m['ece']:.4f} | {m['coverage_at_5%_error']:.3f} |"
            )
    return "\n".join(rows)


def baselines_main(argv: list[str] | None = None) -> int:
    """The mode comparison, written into the first run's baselines.json (which `den publish` puts on the card)."""
    p = argparse.ArgumentParser(prog="den baselines")
    p.add_argument("runs", nargs="+", help="LABEL=RUN pairs, the shipped run first, e.g. D=runs/round2 A=runs/base-a")
    p.add_argument("--files", nargs="+", help="files every run was evaluated on (default: those the first run has)")
    args = p.parse_args(argv)
    if bad := [r for r in args.runs if "=" not in r]:
        raise SystemExit(f"give runs as LABEL=RUN: {bad}")
    runs = {label: Path(path) for label, _, path in (r.partition("=") for r in args.runs)}
    first = next(iter(runs.values()))
    table = baselines(runs, args.files or sorted(read_json(first / "eval.json")))
    (first / "baselines.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(markdown(table))
    print(f"wrote {first / 'baselines.json'}")
    return 0
