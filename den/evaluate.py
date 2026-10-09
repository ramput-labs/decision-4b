"""A trained run, usable: load it, answer /v1/systemone requests, and measure it on labelled files.

A run directory (runs/<name>/, or the same files downloaded from the Hub) holds `merged/`, the LoRA folded into bf16
weights by `den train --merge`, and `head.safetensors` + `head.json`, the pointer head with its config and
temperature. The backbone runs on whichever backend this machine has (MLX, CUDA, CPU) through `device.load`, like the
base model; the head runs in PyTorch on the CPU.

    den predict  --run runs/round2 requests.jsonl                # one /v1/systemone response per line
    den predict  --run hf:<org>/<name> requests.jsonl           # the same, straight from the Hub
    den evaluate --run runs/round2 dev/core.jsonl                # files under data/clean; test needs --final
    den baselines D=runs/round2 A=runs/base-a C=runs/base-c test/core.jsonl   # one table, into D's baselines.json
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

from . import evidence
from .api import Json, Question, Record, RecordError, parse, read
from .catalog import record_path, role
from .device import load
from .fetch import digest
from .metrics import Answer, Row, robustness, summarize
from .model import load_head
from .paths import CLEAN, model_dir
from .prompt import LETTERS, Style, encode, none_pair, perturb, rotate_options, split
from .release import locate
from .train import sample


def base_weights(key: str) -> Path:
    """A base backbone by its pinned key: `models/<key>` when downloaded, else the pinned Hub revision (cached), so a
    head-only run loads on a clean machine too."""
    local = model_dir(key)
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


type Scored = tuple[str, Question, Answer, Answer]  # record id, question, calibrated, uncalibrated (T = 1)


def scored(model: Model, records: Sequence[Record]) -> list[Scored]:
    """Every question's probabilities, calibrated and uncalibrated. A question the run can't score (the letter scorer
    past 26 options) is left out; `metrics` reports how many."""
    out = []
    for record in (one for r in records for one in split(r)):
        if model.style == "letters" and len(record.questions[0].options) > len(LETTERS):
            continue
        logits, _ = model.logits(record)
        for q, s, c in zip(record.questions, logits, model.calibrate(record, logits), strict=True):
            calibrated = Answer(c.softmax(-1).tolist(), q.label, q.type, q.source, q.target)
            out.append((record.id, q, calibrated, Answer(s.softmax(-1).tolist(), q.label, q.type, q.source, q.target)))
    return out


def answers(model: Model, records: Sequence[Record]) -> tuple[list[Answer], list[Answer]]:
    """Calibrated and uncalibrated answers, for `metrics.summarize`."""
    rows = scored(model, records)
    return [c for _, _, c, _ in rows], [r for _, _, _, r in rows]


def metas(path: Path) -> dict[str, dict[str, Json]]:
    """Each record's `_meta`, by record id: the variant, parent, pair and control links Kev's checks pair rows by."""
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
        value = m.get(key)
        return str(value) if value else None

    return Row(calibrated, rid, q.id, q.keys, text("variant") or "clean", text("parent_id"), text("pair_id"),
               text("sibling"), text("control_id"), text("source") or "")  # fmt: skip


type Scoring = tuple[dict[str, Any], list[Row]]  # the report, and every question's row (`evidence`)


def metrics(model: Model, records: Sequence[Record], meta: dict[str, dict[str, Json]] | None = None) -> Scoring:
    """`metrics.summarize` at the run's temperature, with the same headline numbers uncalibrated beside it, and, given
    the file's `_meta`, Kev's robustness checks (`metrics.robustness`) on whatever structure it has."""
    items = scored(model, records)
    result = report([c for _, _, c, _ in items], [r for _, _, _, r in items])
    result["unscored_questions"] = sum(len(r.questions) for r in records) - result["questions"] - result["soft_skipped"]
    rows = [row(i, (meta or {}).get(i[0])) for i in items]
    if meta is not None and (checks := robustness(rows)):
        result["robustness"] = checks
    return result, rows


def permutation_metrics(model: Model, records: Sequence[Record]) -> Scoring:
    """Option order: every choice question read again with its options rotated (`prompt.rotate_options`), scored as
    usual, plus `robustness.permutation` against the original order: how much the probabilities move and how often the
    answer changes. Isolation needs no such check: each question is always read with the state alone (`split`)."""
    rows = [one for r in records for one in split(r) if one.questions[0].type == "choice"]
    rotated = [rotate_options(r, random.Random(f"rotate:{i}")) for i, r in enumerate(rows)]
    original, moved = scored(model, rows), scored(model, rotated)
    result = report([c for _, _, c, _ in moved], [r for _, _, _, r in moved])
    clean = [row(i) for i in original]
    shifted = [row(i, {"variant": "permuted", "parent_id": i[0]}) for i in moved]
    result["robustness"] = {"permutation": robustness(clean + shifted)["permutation"]} if moved else {}
    return result, clean + shifted


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


def pair_metrics(model: Model, records: Sequence[Record]) -> Scoring:
    """Kev's minimal pairs: each eligible choice question twice, with the true option present (a "none" option is
    wrong) and removed (the same "none" option is right). `pair_accuracy` counts pairs with both halves right: the
    model must read whether the evidence matches an option, not just pick a familiar one."""
    halves: list[Record] = []
    for i, r in enumerate(one for rec in records for one in split(rec)):
        if (pair := none_pair(r, random.Random(f"pair:{i}"))) is not None:
            halves += pair
    items = scored(model, halves)
    calibrated, raw = [c for _, _, c, _ in items], [r for _, _, _, r in items]
    result = report(calibrated, raw)
    right = [max(range(len(a.probs)), key=a.probs.__getitem__) == a.label for a in calibrated]
    pairs = list(zip(right[::2], right[1::2], strict=True))
    result["pairs"] = len(pairs)
    result["pair_accuracy"] = sum(a and b for a, b in pairs) / max(len(pairs), 1)
    result["accuracy_present"] = sum(a for a, _ in pairs) / max(len(pairs), 1)
    result["accuracy_absent"] = sum(b for _, b in pairs) / max(len(pairs), 1)
    halves_rows = [row(i, {"variant": "pair_present" if n % 2 == 0 else "pair_absent", "pair_id": f"{i[0]}#{n // 2}",
                           "sibling": "ab"[n % 2]}) for n, i in enumerate(items)]  # fmt: skip
    return result, halves_rows


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
        choices=("none-replace", "none-add", "distract", "pairs", "permute"),
        help="score the files with one of Kev's augmentations applied; `pairs` scores minimal pairs, `permute` the "
        "change when every choice question's options are rotated",
    )
    p.add_argument("--backend")
    p.add_argument("--no-evidence", action="store_true", help="don't write reports/runs/<run>/ (a throwaway check)")
    args = p.parse_args(argv)
    args.files = [record_path(f) for f in args.files]  # `dev/../test/x` is test, and must match the record of reads
    if not args.final and any(role(f) == "test" for f in args.files):
        raise SystemExit("test partitions are read once per release candidate: pass --final for that one read")
    remote = args.run.startswith(("hf:", "release:"))
    report_path = None if remote else Path(args.run) / "eval.json"  # never into the Hub cache
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path and report_path.is_file() else {}
    suffix = (f"+{args.augment}" if args.augment else "") + (f"+max{args.max_options}" if args.max_options else "")
    read_before = set(report) | set(evidence.keys(args.run))  # local eval.json, or the committed evidence of any run
    if again := [f + suffix for f in args.files if role(f) == "test" and f + suffix in read_before]:
        raise SystemExit(f"{args.run} has already read {again}: the locked test set is read once per run")
    path = locate(args.run)
    model = Model(path, args.backend)
    scored_files: dict[str, dict[str, Any]] = {}
    for f in args.files:
        records = sample(CLEAN / f, args.limit, seed=0) if args.limit else list(read(CLEAN / f))
        if args.max_options:
            rows = [one for r in records for one in split(r)]
            records = [r for r in rows if len(r.questions[0].options) <= args.max_options]
        started = time.perf_counter()
        if args.augment == "pairs":
            result, answered = pair_metrics(model, records)
        elif args.augment == "permute":
            result, answered = permutation_metrics(model, records)
        else:  # the file's own variants, pairs and controls are only meaningful unperturbed
            result, answered = metrics(
                model, perturbed(records, args.augment), None if args.augment else metas(CLEAN / f)
            )
        result["ms_per_question"] = round(
            1000 * (time.perf_counter() - started) / max(result["questions"] + result["soft_skipped"], 1), 1
        )
        if args.limit:
            result["sampled_records"] = len(records)
        result["read_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["locked_test"] = Path(f).parts[0] == "test"
        report[f + suffix] = result
        if report_path:  # after every file: a test file read is recorded even if a later one fails
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not args.no_evidence:
            evidence.write(args.run, f + suffix, result, answered)
            scored_files[f + suffix] = {
                "sha256": digest(CLEAN / f),
                "read_at": result["read_at"],
                "questions": len(answered),
                "sampled_records": result.get("sampled_records"),
            }
        coverage = result["coverage_at_5%_error"]
        print(f"{f + suffix:<40} n {result['questions']:>6.0f}  acc {result['accuracy']:.4f}  nll {result['nll']:.4f}  "
              f"brier {result['brier']:.4f}  ece {result['ece']:.4f}  cov@5% {coverage:.3f}"
              + robust_line(result.get("robustness", {})), flush=True)  # fmt: skip
    if scored_files:
        from .train import commit

        evidence.provenance(args.run, path, scored_files, commit())
        print(f"evidence: {evidence.EVIDENCE / evidence.name(args.run)} (commit it)")
    return 0


def robust_line(checks: dict[str, Any]) -> str:
    """The robustness numbers worth a glance, for the evaluate log line."""
    parts = []
    if p := checks.get("permutation"):
        parts.append(f"order-flip {p['flip_rate']:.3f}")
    if (f := checks.get("paired_flip")) and f.get("flip_rate") is not None:
        parts.append(f"pair-flip {f['flip_rate']:.3f}")
    if (u := checks.get("unknowable")) and u.get("share_at_0_9") is not None:
        parts.append(f"unknowable@0.9 {u['share_at_0_9']:.3f}")
    return "  " + "  ".join(parts) if parts else ""


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
    p.add_argument("runs", nargs="+", help="run directories, each evaluated on the same dev files")
    p.add_argument("--files", nargs="+", help="dev files to judge on (default: those every run has)")
    p.add_argument("--max-drop", type=float, default=0.01, help="largest accuracy drop on any file a winner may have")
    p.add_argument(
        "--paired",
        action="store_true",
        help="two runs (dirs, hf:, release:), question by question, with 95%% intervals",
    )
    args = p.parse_args(argv)
    if args.paired:
        return paired_main(args.runs, args.files)
    args.runs = [Path(r) for r in args.runs]
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


def paired_main(runs: list[str], files: list[str] | None) -> int:
    """b against a on the questions both answered (their committed rows, `evidence`): the accuracy and NLL change with
    a 95% interval from resampling records, so a gain can be told apart from noise. Saved beside b's evidence."""
    if len(runs) != 2:
        raise SystemExit("--paired compares exactly two runs: A B (B against A)")
    a, b = runs
    shared = sorted(set(evidence.keys(a)) & set(evidence.keys(b)))
    keys = files or [k for k in shared if Path(k).parts[0] != "test"]  # test only when named: it was read already
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


BASELINE_METRICS = ("questions", "accuracy", "nll", "brier", "ece", "coverage_at_5%_error", "ms_per_question")


def baselines(runs: dict[str, Path], files: Sequence[str]) -> dict[str, Any]:
    """Every run's numbers on the same files, from each run's eval.json, with its mode read from head.json:
    A zero-shot Qwen (letters, no LoRA, no training), B LoRA + letters, C frozen Qwen + head, D LoRA + head."""
    table: dict[str, Any] = {"files": list(files), "runs": {}}
    for label, run in runs.items():
        evaluated = json.loads((run / "eval.json").read_text(encoding="utf-8"))
        if missing := [f for f in files if f not in evaluated]:
            raise SystemExit(f"{run} was not evaluated on {missing}: den evaluate --run {run} ... first")
        head = json.loads((run / "head.json").read_text(encoding="utf-8"))
        letters, lora = head.get("kind") == "letters", bool(head.get("lora"))
        mode = ("B" if lora else "A") if letters else ("D" if lora else "C")  # letters + no LoRA never trains
        table["runs"][label] = {
            "path": str(run),
            "mode": mode,
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
    """The baseline comparison (Qwen zero-shot A, frozen head C, LoRA + letters B, ours D) on the same files, written
    into the first run's baselines.json, which `den publish` puts on the model card."""
    p = argparse.ArgumentParser(prog="den baselines")
    p.add_argument("runs", nargs="+", help="LABEL=RUN pairs, the shipped run first, e.g. D=runs/round2 A=runs/base-a")
    p.add_argument("--files", nargs="+", help="files every run was evaluated on (default: those the first run has)")
    args = p.parse_args(argv)
    if bad := [r for r in args.runs if "=" not in r]:
        raise SystemExit(f"give runs as LABEL=RUN: {bad}")
    runs = {label: Path(path) for label, _, path in (r.partition("=") for r in args.runs)}
    first = next(iter(runs.values()))
    files = args.files or sorted(json.loads((first / "eval.json").read_text(encoding="utf-8")))
    table = baselines(runs, files)
    (first / "baselines.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(markdown(table))
    print(f"wrote {first / 'baselines.json'}")
    return 0
