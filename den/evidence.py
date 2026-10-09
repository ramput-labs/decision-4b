"""Every evaluation, kept in git: what was measured, on what, and every question's answer.

`den evaluate` writes, for each run and file, `reports/runs/<run>/<file>/report.json` (every number `metrics`
computes) and `rows.json` (each question's calibrated probabilities, label and the links Kev's checks pair rows by),
plus one `reports/runs/<run>/provenance.json` (the run as given, its base, LoRA, head and temperature, the commit it
was trained and evaluated at, and the sha256 of every file scored). No weights: a run is a few MB, so it is committed
and outlives the machine that trained it, like Kev's runs/.

Rows make comparisons question by question: `den compare --paired A B` matches two runs' rows and reports the
accuracy and NLL difference with a 95% interval from resampling whole records (`metrics.paired_bootstrap`), so "v2 is
better" can be told apart from noise. `den release` records that comparison against the parent version.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import Answer, Row, paired_bootstrap
from .paths import REPORTS

EVIDENCE = REPORTS / "runs"


def name(run: str) -> str:
    """The evidence folder of a run: its path under runs/ (runs/kev-recipe/4-skills -> kev-recipe/4-skills), the
    release (release:v1 -> release-v1), the Hub repo and revision (hf:org/x@abc -> hf-org-x-abc), or a folder named
    directly (evidence:kev-recipe/4-skills)."""
    if run.startswith("evidence:"):  # a folder already: what a release records
        return run.removeprefix("evidence:")
    if run.startswith("release:"):
        return "release-" + run.removeprefix("release:")
    if run.startswith("hf:"):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", run.replace(":", "-").replace("/", "-").replace("@", "-"))
    path = Path(run)
    parts = path.parts
    return "/".join(parts[parts.index("runs") + 1 :]) if "runs" in parts[:-1] else path.name


def folder(run: str, key: str, root: Path | None = None) -> Path:
    """reports/runs/<run>/<file without .jsonl>[+augmentation]/"""
    return (root or EVIDENCE) / name(run) / key.replace(".jsonl", "")


def encode(r: Row) -> dict[str, Any]:
    """A row as JSON: probabilities to 6 places, empty links left out."""
    a = r.answer
    out: dict[str, Any] = {
        "id": r.record,
        "question": r.question,
        "source": a.source,
        "type": a.kind,
        "variant": r.variant,
        "keys": list(r.keys),
        "label": a.label,
        "p": [round(float(x), 6) for x in a.probs],
    }
    if a.target is not None:
        out["target"] = [round(float(x), 6) for x in a.target]
    links = {"parent": r.parent, "pair": r.pair, "sibling": r.sibling, "control": r.control, "origin": r.origin}
    return out | {k: v for k, v in links.items() if v}


def decode(d: dict[str, Any]) -> Row:
    answer = Answer(d["p"], d["label"], d["type"], d["source"], d.get("target"))
    return Row(
        answer,
        d["id"],
        d["question"],
        d["keys"],
        d["variant"],
        d.get("parent"),
        d.get("pair"),
        d.get("sibling"),
        d.get("control"),
        d.get("origin", ""),
    )


def write(run: str, key: str, result: dict[str, Any], rows: Sequence[Row], root: Path | None = None) -> Path:
    where = folder(run, key, root)
    where.mkdir(parents=True, exist_ok=True)
    (where / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    body = ",\n".join(json.dumps(encode(r), separators=(",", ":")) for r in rows)
    (where / "rows.json").write_text(f"[\n{body}\n]\n", encoding="utf-8")
    return where


def provenance(run: str, path: Path, scored: dict[str, dict[str, Any]], commit: str, root: Path | None = None) -> None:
    """reports/runs/<run>/provenance.json: what was evaluated, merged with what earlier evaluations recorded."""
    target = (root or EVIDENCE) / name(run) / "provenance.json"
    known: dict[str, Any] = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    head = _json(path / "head.json")
    trained = _json(path / "run.json")
    entry = {
        "run": run,
        "base": head.get("base"),
        "lora": head.get("lora"),
        "head": {k: head.get(k) for k in ("kind", "dim", "layers", "heads", "rep", "proj")},
        "temperature": head.get("temperature"),
        "trained_commit": trained.get("commit"),
        "evaluated_commit": commit,
        "files": {**known.get("files", {}), **scored},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return loaded


def rows(run: str, key: str, root: Path | None = None) -> list[Row]:
    path = folder(run, key, root) / "rows.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: den evaluate --run {run} {key.split('+')[0]}")
    return [decode(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def keys(run: str, root: Path | None = None) -> list[str]:
    """Every file key a run has evidence for."""
    base = (root or EVIDENCE) / name(run)
    return sorted(str(p.parent.relative_to(base)) + ".jsonl" for p in base.rglob("rows.json"))


def paired(a: str, b: str, key: str, root: Path | None = None) -> dict[str, Any] | None:
    """b against a on the questions both answered (same record, question and variant): accuracy and NLL differences
    with 95% intervals. Soft-target (unknowable) questions have no right answer and are left out."""
    left = {(r.record, r.question, r.variant): r for r in rows(a, key, root) if r.answer.target is None}
    right = {(r.record, r.question, r.variant): r for r in rows(b, key, root) if r.answer.target is None}
    shared = sorted(set(left) & set(right))
    if not shared:
        return None

    def correct(r: Row) -> float:
        p = list(r.answer.probs)
        return float(max(range(len(p)), key=p.__getitem__) == r.answer.label)

    def nll(r: Row) -> float:
        return -math.log(max(float(r.answer.probs[r.answer.label]), 1e-12))

    groups = [k[0] for k in shared]
    acc = paired_bootstrap(groups, [correct(left[k]) for k in shared], [correct(right[k]) for k in shared])
    loss = paired_bootstrap(groups, [nll(left[k]) for k in shared], [nll(right[k]) for k in shared])
    return {
        "questions": int(acc["questions"]),
        "records": int(acc["groups"]),
        "accuracy": {
            "a": sum(correct(left[k]) for k in shared) / len(shared),
            "b": sum(correct(right[k]) for k in shared) / len(shared),
            **_ci(acc),
        },
        "nll": {
            "a": sum(nll(left[k]) for k in shared) / len(shared),
            "b": sum(nll(right[k]) for k in shared) / len(shared),
            **_ci(loss, higher_is_better=False),
        },
        "only_in_a": len(set(left) - set(right)),
        "only_in_b": len(set(right) - set(left)),
    }


def _ci(stats: dict[str, float], higher_is_better: bool = True) -> dict[str, Any]:
    up, down = ("better", "worse") if higher_is_better else ("worse", "better")
    verdict = up if stats["low"] > 0 else down if stats["high"] < 0 else "no clear difference"
    return {"diff": stats["diff"], "ci95": [stats["low"], stats["high"]], "b_vs_a": verdict}
