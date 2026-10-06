"""A trained run, usable: load it, answer /v1/systemone requests, and measure it on labelled files.

A run directory (runs/<name>/, or the same files downloaded from the Hub) holds `merged/`, the LoRA folded into bf16
weights by `den train --merge`, and `head.safetensors` + `head.json`, the pointer head with its config and
temperature. The backbone runs on whichever backend this machine has (MLX, CUDA, CPU) through `device.load`, like the
base model; the head runs in PyTorch on the CPU.

    den predict  --run runs/round2 requests.jsonl                # one /v1/systemone response per line
    den predict  --run hf:<org>/<name> requests.jsonl           # the same, straight from the Hub
    den evaluate --run runs/round2 dev/core.jsonl                # files under data/clean; test needs --final
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from .api import Json, Record, RecordError, parse, read
from .device import load
from .metrics import Answer, summarize
from .model import load_head
from .prompt import LETTERS, Style, encode, none_pair, perturb, split
from .train import sample

CLEAN = Path("data/clean")


def locate(run: str) -> Path:
    """A run directory: a local path, or `hf:<org>/<name>[@<revision>]`, downloaded once into the Hub cache."""
    if not run.startswith("hf:"):
        return Path(run)
    from huggingface_hub import snapshot_download

    repo, _, revision = run.removeprefix("hf:").partition("@")
    return Path(snapshot_download(repo, revision=revision or None))


def base_weights(key: str) -> Path:
    """A base backbone by its pinned key: `models/<key>` when downloaded, else the pinned Hub revision (cached), so a
    head-only run loads on a clean machine too."""
    local = Path("models") / key
    if (local / "config.json").is_file():
        return local
    from huggingface_hub import snapshot_download

    from .pins import MODELS

    if key not in MODELS:
        raise SystemExit(f"unknown base {key!r}: `den models` lists them")
    return Path(snapshot_download(MODELS[key].source.repo, revision=MODELS[key].source.revision))


class Model:
    """A trained run, ready to answer: the backbone (merged weights, or the frozen base for a head-only run) on this
    machine's backend, and the pointer head with its temperature. `den.load(run)` builds one."""

    def __init__(self, run: Path, backend: str | None = None) -> None:
        meta = json.loads((run / "head.json").read_text(encoding="utf-8")) if (run / "head.json").is_file() else {}
        if not meta:
            raise SystemExit(f"{run} has no head.json: not a den run")
        weights = run / "merged"
        if not (weights / "config.json").is_file():
            if meta.get("lora"):
                raise SystemExit(f"{weights} is missing: train with --merge, or download the run's merged/ folder")
            weights = base_weights(str(meta["base"]))  # a head-only run reads the frozen base
        self.backbone = load(weights, backend)  # type: ignore[arg-type]
        self.head, self.meta = load_head(run, self.backbone.hidden_size)  # refuses a head built for another size
        self.head.eval()
        self.temperature = float(self.meta["temperature"])
        self.temperatures: dict[str, float] = self.meta.get("temperatures") or {}  # per question type, if fitted
        self.style: Style = "letters" if self.meta["kind"] == "letters" else "dash"
        self.tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))

    def predict(self, raw: dict[str, Json], name: str = "den-latest") -> dict[str, Any]:
        """The /v1/systemone response for one request (Kev's shape), as `den serve` returns it."""
        from .serve import respond

        return respond(self, raw, name)

    def logits(self, record: Record) -> tuple[list[torch.Tensor], int]:
        """Uncalibrated logits for each question's options, and the tokens read. Each question is read with the state
        alone, as in training (`prompt.split`)."""
        out, tokens = [], 0
        for one in split(record):
            example = encode(one, self.tokenizer, max_state=1 << 30, style=self.style)
            if example is None:
                raise RecordError(f"{one.questions[0].id}: the letter scorer reads at most 26 options")
            tokens += len(example.ids)
            hidden = torch.from_numpy(self.backbone.hidden_states(example.ids))[None]
            with torch.no_grad():
                out.append(self.head(hidden, [example])[0][0].float())
        return out, tokens

    def calibrate(self, record: Record, logits: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """logits / T, with the question type's own T when one was fitted (`--calibrate-by-type`)."""
        return [
            s / self.temperatures.get(q.type, self.temperature) for q, s in zip(record.questions, logits, strict=True)
        ]

    def read(self, record: Record) -> tuple[list[torch.Tensor], int]:
        """Calibrated logits for each question's options, and the tokens read."""
        raw, tokens = self.logits(record)
        return self.calibrate(record, raw), tokens

    def scores(self, record: Record) -> list[torch.Tensor]:
        return self.read(record)[0]


def request(raw: dict[str, Json]) -> Record:
    """An unlabelled request as a Record: each question gets a placeholder label, which nothing reads."""
    questions = raw.get("questions")
    if not isinstance(questions, dict):
        raise SystemExit("request needs a questions object")
    filled: dict[str, Json] = {}
    for qid, q in questions.items():
        if isinstance(q, dict) and "label" not in q:
            criteria = q.get("criteria")
            placeholder: Json
            match q.get("type"):
                case "choice":
                    placeholder = next(iter(criteria), None) if isinstance(criteria, dict) else None
                case "score":
                    placeholder = 0
                case _:
                    placeholder = False
            q = {**q, "label": placeholder}
        filled[qid] = q
    return parse({**raw, "questions": filled}, "request")


def answers(model: Model, records: Sequence[Record]) -> tuple[list[Answer], list[Answer]]:
    """Every question's probabilities, calibrated and uncalibrated (T = 1), for `metrics.summarize`. A question the
    run can't score (the letter scorer past 26 options) is left out; `metrics` reports how many."""
    calibrated, raw = [], []
    for record in (one for r in records for one in split(r)):
        if model.style == "letters" and len(record.questions[0].options) > len(LETTERS):
            continue
        logits, _ = model.logits(record)
        for q, s, c in zip(record.questions, logits, model.calibrate(record, logits), strict=True):
            calibrated.append(Answer(c.softmax(-1).tolist(), q.label, q.type, q.source, q.target))
            raw.append(Answer(s.softmax(-1).tolist(), q.label, q.type, q.source, q.target))
    return calibrated, raw


def metrics(model: Model, records: Sequence[Record]) -> dict[str, Any]:
    """`metrics.summarize` at the run's temperature, with the same headline numbers uncalibrated beside it."""
    result = report(*answers(model, records))
    result["unscored_questions"] = sum(len(r.questions) for r in records) - result["questions"] - result["soft_skipped"]
    return result


def report(calibrated: Sequence[Answer], raw: Sequence[Answer]) -> dict[str, Any]:
    result = summarize(calibrated)
    before = summarize(raw)
    result["uncalibrated"] = {k: before[k] for k in ("accuracy", "nll", "brier", "ece")}
    return result


def perturbed(records: Sequence[Record], mode: str | None) -> list[Record]:
    """The records, one question each, with `mode` applied (seeded per record, so every run sees the same)."""
    rows = [one for r in records for one in split(r)]
    if mode is None:
        return rows
    return [perturb(r, random.Random(f"perturb:{i}"), mode) for i, r in enumerate(rows)]  # type: ignore[arg-type]


def pair_metrics(model: Model, records: Sequence[Record]) -> dict[str, Any]:
    """Kev's minimal pairs: each eligible choice question twice, with the true option present (a "none" option is
    wrong) and removed (the same "none" option is right). `pair_accuracy` counts pairs with both halves right: the
    model must read whether the evidence matches an option, not just pick a familiar one."""
    halves: list[Record] = []
    for i, r in enumerate(one for rec in records for one in split(rec)):
        if (pair := none_pair(r, random.Random(f"pair:{i}"))) is not None:
            halves += pair
    calibrated, raw = answers(model, halves)
    result = report(calibrated, raw)
    right = [max(range(len(a.probs)), key=a.probs.__getitem__) == a.label for a in calibrated]
    pairs = list(zip(right[::2], right[1::2], strict=True))
    result["pairs"] = len(pairs)
    result["pair_accuracy"] = sum(a and b for a, b in pairs) / max(len(pairs), 1)
    result["accuracy_present"] = sum(a for a, _ in pairs) / max(len(pairs), 1)
    result["accuracy_absent"] = sum(b for _, b in pairs) / max(len(pairs), 1)
    return result


def evaluate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den evaluate")
    p.add_argument("--run", required=True, help="a run directory, or hf:<org>/<name>[@<revision>]")
    p.add_argument("files", nargs="+", help="files under data/clean, e.g. dev/core.jsonl")
    p.add_argument("--final", action="store_true", help="allow test/ files: once per release candidate")
    p.add_argument("--limit", type=int, default=0, help="score a seeded sample of this many records per file (0: all)")
    p.add_argument(
        "--max-options",
        type=int,
        default=0,
        help="score only questions with at most this many options (e.g. 26, to compare with the letter scorer)",
    )
    p.add_argument(
        "--augment",
        choices=("none-replace", "none-add", "distract", "pairs"),
        help="score the files with one of Kev's augmentations applied; `pairs` scores minimal pairs",
    )
    p.add_argument("--backend")
    args = p.parse_args(argv)
    if not args.final and any(Path(f).parts[0] == "test" for f in args.files):
        raise SystemExit("test partitions are read once per release candidate: pass --final for that one read")
    run = locate(args.run)
    model = Model(run, args.backend)
    report_path = Path(args.run) / "eval.json" if not args.run.startswith("hf:") else None  # never into the Hub cache
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path and report_path.is_file() else {}
    suffix = (f"+{args.augment}" if args.augment else "") + (f"+max{args.max_options}" if args.max_options else "")
    for f in args.files:
        records = sample(CLEAN / f, args.limit, seed=0) if args.limit else list(read(CLEAN / f))
        if args.max_options:
            rows = [one for r in records for one in split(r)]
            records = [r for r in rows if len(r.questions[0].options) <= args.max_options]
        started = time.perf_counter()
        result = (
            pair_metrics(model, records)
            if args.augment == "pairs"
            else metrics(model, perturbed(records, args.augment))
        )
        result["ms_per_question"] = round(
            1000 * (time.perf_counter() - started) / max(result["questions"] + result["soft_skipped"], 1), 1
        )
        if args.limit:
            result["sampled_records"] = len(records)
        report[f + suffix] = result
        coverage = result["coverage_at_5%_error"]
        print(f"{f + suffix:<40} n {result['questions']:>6.0f}  acc {result['accuracy']:.4f}  nll {result['nll']:.4f}  "
              f"brier {result['brier']:.4f}  ece {result['ece']:.4f}  cov@5% {coverage:.3f}", flush=True)  # fmt: skip
    if report_path:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    """The /v1/systemone response for each request line, exactly as `den serve` returns it."""
    from .serve import respond

    p = argparse.ArgumentParser(prog="den predict")
    p.add_argument("--run", required=True, help="a run directory, or hf:<org>/<name>[@<revision>]")
    p.add_argument("requests", nargs="?", default="-", help="JSONL of /v1/systemone requests; - reads stdin")
    p.add_argument("--backend")
    args = p.parse_args(argv)
    model = Model(locate(args.run), args.backend)
    text = sys.stdin.read() if args.requests == "-" else Path(args.requests).read_text(encoding="utf-8")
    for line in filter(str.strip, text.splitlines()):
        print(json.dumps(respond(model, json.loads(line))), flush=True)
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    """Pick the run to ship from dev results (`eval.json` of each run), never from test."""
    p = argparse.ArgumentParser(prog="den compare")
    p.add_argument("runs", nargs="+", type=Path, help="run directories, each evaluated on the same dev files")
    p.add_argument("--files", nargs="+", help="dev files to judge on (default: those every run has)")
    p.add_argument("--max-drop", type=float, default=0.01, help="largest accuracy drop on any file a winner may have")
    args = p.parse_args(argv)
    reports = {run: json.loads((run / "eval.json").read_text(encoding="utf-8")) for run in args.runs}
    files = args.files or sorted(set.intersection(*(set(r) for r in reports.values())))
    if not files or any(Path(f).parts[0] == "test" for f in files):
        raise SystemExit("compare on dev files that every run was evaluated on; never on test")
    print(f"{'file':<40}" + "".join(f"{run.name:>16}" for run in args.runs))
    for f in files:
        print(f"{f:<40}" + "".join(f"{reports[run][f]['accuracy']:>16.4f}" for run in args.runs))
    mean = {run: sum(reports[run][f]["accuracy"] for f in files) / len(files) for run in args.runs}
    print(f"{'mean accuracy':<40}" + "".join(f"{mean[run]:>16.4f}" for run in args.runs))
    first = args.runs[0]
    best = first
    for run in args.runs[1:]:
        drop = max(reports[first][f]["accuracy"] - reports[run][f]["accuracy"] for f in files)
        if mean[run] > mean[best] and drop <= args.max_drop:
            best = run
    print(f"ship: {best}")
    return 0
