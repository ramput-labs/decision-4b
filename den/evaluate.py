"""`den evaluate`: a trained run's accuracy, NLL and calibration on files under data/clean.

    den evaluate --run runs/round2 dev/core.jsonl                 # test/ needs --final, once per run
    den evaluate --run runs/round2 --augment pairs dev/core.jsonl # one of Kev's augmentations

Results go into the run's `eval.json` and, per file, into committed evidence (`evidence.py`).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import evidence
from .api import Json, Question, Record, read
from .catalog import record_path, role
from .data import sample
from .fetch import digest
from .metrics import Answer, Row, argmax, robustness, summarize
from .paths import CLEAN, locate
from .prompt import LETTERS, none_pair, perturb, rotate_options, split
from .runtime import Model

type Scored = tuple[str, Question, Answer, Answer]  # record id, question, calibrated, uncalibrated (T = 1)
type Scoring = tuple[dict[str, Any], list[Row]]  # the report, and every question's row


def scored(model: Model, records: Sequence[Record]) -> list[Scored]:
    """Every question's probabilities, calibrated and not. Questions the letter scorer can't read (more than 26
    options) are left out and counted as `unscored_questions`."""
    out = []
    for record in (one for r in records for one in split(r)):
        if model.style == "letters" and len(record.questions[0].options) > len(LETTERS):
            continue
        logits, _ = model.logits(record)
        for q, s, c in zip(record.questions, logits, model.calibrate(record, logits), strict=True):
            calibrated = Answer(c.softmax(-1).tolist(), q.label, q.type, q.source, q.target, q.teacher)
            raw = Answer(s.softmax(-1).tolist(), q.label, q.type, q.source, q.target, q.teacher)
            out.append((record.id, q, calibrated, raw))
    return out


def report(items: Sequence[Scored]) -> dict[str, Any]:
    """`metrics.summarize` at the run's temperature, with the headline numbers at T = 1 beside it."""
    result = summarize([c for _, _, c, _ in items])
    before = summarize([r for _, _, _, r in items])
    result["uncalibrated"] = {k: before[k] for k in ("accuracy", "nll", "brier", "ece")}
    return result


def metas(path: Path) -> dict[str, dict[str, Json]]:
    """Each record's `_meta` by record id: the links Kev's robustness checks pair rows by."""
    out: dict[str, dict[str, Json]] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip() and isinstance(meta := json.loads(line).get("_meta"), dict):
                out[str(meta.get("id") or f"{path.stem}:{n}")] = meta
    return out


def row(item: Scored, meta: dict[str, Json] | None = None) -> Row:
    rid, q, calibrated, _ = item
    m = meta or {}

    def text(key: str) -> str | None:
        return str(value) if (value := m.get(key)) else None

    return Row(calibrated, rid, q.id, q.keys, text("variant") or "clean", text("parent_id"), text("pair_id"),
               text("sibling"), text("control_id"), text("source") or "")  # fmt: skip


def metrics(model: Model, records: Sequence[Record], meta: dict[str, dict[str, Json]] | None = None) -> Scoring:
    """The report, plus Kev's robustness checks on whatever structure the file's `_meta` carries."""
    items = scored(model, records)
    result = report(items)
    result["unscored_questions"] = sum(len(r.questions) for r in records) - result["questions"] - result["soft_skipped"]
    rows = [row(i, (meta or {}).get(i[0])) for i in items]
    if meta is not None and (checks := robustness(rows)):
        result["robustness"] = checks
    return result, rows


def permutation_metrics(model: Model, records: Sequence[Record]) -> Scoring:
    """Every choice question again with its options rotated, scored as usual, plus how much the probabilities move
    and how often the answer changes against the original order."""
    rows = [one for r in records for one in split(r) if one.questions[0].type == "choice"]
    rotated = [rotate_options(r, random.Random(f"rotate:{i}")) for i, r in enumerate(rows)]
    original, moved = scored(model, rows), scored(model, rotated)
    result = report(moved)
    clean = [row(i) for i in original]
    shifted = [row(i, {"variant": "permuted", "parent_id": i[0]}) for i in moved]
    result["robustness"] = {"permutation": robustness(clean + shifted)["permutation"]} if moved else {}
    return result, clean + shifted


def pair_metrics(model: Model, records: Sequence[Record]) -> Scoring:
    """Kev's minimal pairs: each eligible choice question with the true option present (a "none" option is wrong) and
    removed (the same "none" option is right). `pair_accuracy` counts pairs with both halves right."""
    halves: list[Record] = []
    for i, r in enumerate(one for rec in records for one in split(rec)):
        if (pair := none_pair(r, random.Random(f"pair:{i}"))) is not None:
            halves += pair
    items = scored(model, halves)
    result = report(items)
    right = [argmax(c.probs) == c.label for _, _, c, _ in items]
    pairs = list(zip(right[::2], right[1::2], strict=True))
    total = max(len(pairs), 1)
    result |= {
        "pairs": len(pairs),
        "pair_accuracy": sum(a and b for a, b in pairs) / total,
        "accuracy_present": sum(a for a, _ in pairs) / total,
        "accuracy_absent": sum(b for _, b in pairs) / total,
    }
    rows = [row(i, {"variant": "pair_present" if n % 2 == 0 else "pair_absent", "pair_id": f"{i[0]}#{n // 2}",
                    "sibling": "ab"[n % 2]}) for n, i in enumerate(items)]  # fmt: skip
    return result, rows


def perturbed(records: Sequence[Record], mode: str | None) -> list[Record]:
    """The records, one question each, with `mode` applied (seeded per record)."""
    rows = [one for r in records for one in split(r)]
    if mode is None:
        return rows
    return [perturb(r, random.Random(f"perturb:{i}"), mode) for i, r in enumerate(rows)]  # type: ignore[arg-type]


def robust_line(checks: dict[str, Any]) -> str:
    """The robustness numbers worth a glance in the log line."""
    parts = []
    if p := checks.get("permutation"):
        parts.append(f"order-flip {p['flip_rate']:.3f}")
    if (f := checks.get("paired_flip")) and f.get("flip_rate") is not None:
        parts.append(f"pair-flip {f['flip_rate']:.3f}")
    if (u := checks.get("unknowable")) and u.get("share_at_0_9") is not None:
        parts.append(f"unknowable@0.9 {u['share_at_0_9']:.3f}")
    if (kl := checks.get("kl_to_target")) is not None:
        parts.append(f"kl {kl:.3f}")
    return "  " + "  ".join(parts) if parts else ""


def evaluate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den evaluate")
    p.add_argument("--run", required=True, help="a run directory, hf:<org>/<name>[@<revision>] or release:<version>")
    p.add_argument("files", nargs="+", help="files under data/clean, e.g. dev/core.jsonl")
    p.add_argument("--final", action="store_true", help="allow test/ files: once per release candidate")
    p.add_argument("--limit", type=int, default=0, help="score a seeded sample of this many records per file (0: all)")
    p.add_argument("--max-options", type=int, default=0, help="score only questions with at most this many options")
    p.add_argument(
        "--augment",
        choices=("none-replace", "none-add", "distract", "pairs", "permute"),
        help="score with one of Kev's augmentations; `pairs` minimal pairs, `permute` rotated options",
    )
    p.add_argument("--backend")
    p.add_argument("--no-evidence", action="store_true", help="don't write reports/runs/<run>/ (a throwaway check)")
    args = p.parse_args(argv)
    args.files = [record_path(f) for f in args.files]  # `dev/../test/x` is test
    if not args.final and any(role(f) == "test" for f in args.files):
        raise SystemExit("test partitions are read once per release candidate: pass --final for that one read")
    remote = args.run.startswith(("hf:", "release:"))
    report_path = None if remote else Path(args.run) / "eval.json"  # never into the Hub cache
    results = json.loads(report_path.read_text(encoding="utf-8")) if report_path and report_path.is_file() else {}
    suffix = (f"+{args.augment}" if args.augment else "") + (f"+max{args.max_options}" if args.max_options else "")
    read_before = set(results) | set(evidence.keys(args.run))
    if again := [f + suffix for f in args.files if role(f) == "test" and f + suffix in read_before]:
        raise SystemExit(f"{args.run} has already read {again}: the locked test set is read once per run")
    path = locate(args.run)
    model = Model(path, args.backend)
    scored_files: dict[str, dict[str, Any]] = {}
    for f in args.files:
        records = sample(CLEAN / f, args.limit, seed=0) if args.limit else list(read(CLEAN / f))
        if args.max_options:
            records = [r for r in perturbed(records, None) if len(r.questions[0].options) <= args.max_options]
        started = time.perf_counter()
        if args.augment == "pairs":
            result, answered = pair_metrics(model, records)
        elif args.augment == "permute":
            result, answered = permutation_metrics(model, records)
        else:  # a file's own variants, pairs and controls only mean something unperturbed
            meta = None if args.augment else metas(CLEAN / f)
            result, answered = metrics(model, perturbed(records, args.augment), meta)
        seconds = time.perf_counter() - started
        result["ms_per_question"] = round(1000 * seconds / max(result["questions"] + result["soft_skipped"], 1), 1)
        if args.limit:
            result["sampled_records"] = len(records)
        result["read_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["locked_test"] = role(f) == "test"
        results[f + suffix] = result
        if report_path:  # after every file: a test read is recorded even if a later file fails
            report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if not args.no_evidence:
            evidence.write(args.run, f + suffix, result, answered)
            scored_files[f + suffix] = {
                "sha256": digest(CLEAN / f),
                "read_at": result["read_at"],
                "questions": len(answered),
                "sampled_records": result.get("sampled_records"),
            }
        coverage = result["coverage_at_5%_error"]
        checks = result.get("robustness", {}) | {"kl_to_target": result.get("kl_to_target")}
        print(f"{f + suffix:<40} n {result['questions']:>6.0f}  acc {result['accuracy']:.4f}  nll {result['nll']:.4f}  "
              f"brier {result['brier']:.4f}  ece {result['ece']:.4f}  cov@5% {coverage:.3f}" + robust_line(checks),
              flush=True)  # fmt: skip
    if scored_files:
        from .doctor import commit

        evidence.provenance(args.run, path, scored_files, commit())
        print(f"evidence: {evidence.EVIDENCE / evidence.name(args.run)} (commit it)")
    return 0
