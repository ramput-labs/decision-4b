"""`den probe`: modes A and C on any machine, from a frozen backbone's cached hidden states.

    den probe --head-kind letters --dev dev/core.jsonl --max-options 26     # mode A: zero-shot answer letters
    den probe --train train/core.jsonl --dev dev/core.jsonl --max-options 26  # mode C: frozen Qwen + head

Both fit T on the calibration rows and are scored on the same dev rows by `metrics.summarize`, like `den train`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .api import Record
from .catalog import record_path, role
from .data import UNCAPPED, load
from .head import HeadConfig, LetterHead, make_head, question_loss
from .metrics import Answer, summarize
from .paths import CLEAN, base_weights
from .prompt import LETTERS, Example, Style, encode, split

type Cached = list[tuple[torch.Tensor, Example]]


class Features:
    """A frozen backbone (`device.load`: MLX, CUDA or CPU) that reads rows into hidden states."""

    def __init__(self, model: str, backend: str | None, style: Style = "dash") -> None:
        from tokenizers import Tokenizer

        from .device import load as load_backbone

        self.weights = base_weights(model)
        self.backbone = load_backbone(self.weights, backend)  # type: ignore[arg-type]
        self.tokenizer = Tokenizer.from_file(str(self.weights / "tokenizer.json"))
        self.style = style
        self.seconds = 0.0  # time spent in the backbone

    def __call__(self, record: Record) -> tuple[torch.Tensor, Example]:
        example = encode(record, self.tokenizer, max_state=UNCAPPED, style=self.style)
        assert example is not None
        started = time.perf_counter()
        hidden = torch.from_numpy(self.backbone.hidden_states(example.ids))[None]
        self.seconds += time.perf_counter() - started
        return hidden, example


def fit_head(head: nn.Module, cached: Cached, steps: int, lr: float, every: int) -> list[dict[str, float]]:
    """Full-batch AdamW on cached hidden states; refuses a missing or non-finite gradient. Returns the curve."""
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    curve: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        losses, right = [], 0
        for hidden, example in cached:
            (scores,) = head(hidden, [example])[0]
            losses.append(question_loss(scores, example.labels[0], example.targets[0]))
            right += int(scores.argmax()) == example.labels[0]
        loss = torch.stack(losses).mean()
        loss.backward()  # type: ignore[no-untyped-call]
        if silent := [n for n, p in head.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]:
            raise SystemExit(f"step {step}: head parameters without a finite gradient: {silent}")
        optimizer.step()
        if step == 1 or step % every == 0 or step == steps:
            curve.append({"step": step, "loss": round(loss.item(), 4), "accuracy": right / len(cached)})
            print(f"step {step:>4}  train loss {loss.item():.4f}  train acc {right / len(cached):.3f}", flush=True)
    head.eval()
    return curve


def tied_letters(head: LetterHead, weights: Path, tokenizer: Any) -> None:  # noqa: ANN401 (a tokenizers.Tokenizer)
    """The checkpoint's tied embedding rows of " A" ... " Z", read straight from its shards."""
    from safetensors import safe_open

    ids = [tokenizer.encode(f" {c}", add_special_tokens=False).ids[0] for c in LETTERS]
    for shard in sorted(weights.glob("*.safetensors")):
        with safe_open(shard, "pt") as handle:
            names: list[str] = list(handle.keys())
            name = next((k for k in names if k.endswith("embed_tokens.weight")), None)
            if name and not name.startswith("mtp"):
                head.set_letters(handle.get_tensor(name)[ids])
    assert float(head.letters.abs().sum()) > 0, "no embedding rows found"  # type: ignore[operator]


def probe_main(argv: list[str] | None = None) -> int:
    from .calibrate import fit_temperature

    p = argparse.ArgumentParser(prog="den probe")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--head-kind", choices=("set", "pointer", "letters"), default="set")
    p.add_argument("--train", nargs="*", default=[], help="files under data/clean (not for letters)")
    p.add_argument("--dev", nargs="+", required=True, help="files under data/clean; never test")
    p.add_argument("--calibration", nargs="+", default=["calibration/core.jsonl"])
    p.add_argument("--max-options", type=int, default=0, help="score only questions with at most this many options")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    args.train, args.dev, args.calibration = (
        [record_path(f) for f in fs] for fs in (args.train, args.dev, args.calibration)
    )
    if any(role(f) == "test" for f in [*args.train, *args.dev, *args.calibration]):
        raise SystemExit("probe never reads test files")
    if (args.head_kind == "letters") == bool(args.train):
        raise SystemExit("letters is zero-shot (no --train); set/pointer need --train")
    torch.manual_seed(args.seed)
    style: Style = "letters" if args.head_kind == "letters" else "dash"
    limit = min(args.max_options or 255, len(LETTERS) if style == "letters" else 255)

    def rows(files: list[str]) -> list[Record]:
        records = load([CLEAN / f for f in files])
        return [one for r in records for one in split(r) if len(one.questions[0].options) <= limit]

    read = Features(args.model, args.backend, style)
    train_rows, calib_rows, dev_rows = rows(args.train), rows(args.calibration), rows(args.dev)
    started = time.perf_counter()
    train, calib = [read(r) for r in train_rows], [read(r) for r in calib_rows]
    read.seconds = 0.0
    dev = [read(r) for r in dev_rows]
    print(
        f"read {len(train)} train, {len(calib)} calibration, {len(dev)} dev rows on {read.backbone.device}", flush=True
    )

    head = make_head(read.backbone.hidden_size, HeadConfig(args.head_kind))
    curve: list[dict[str, float]] = []
    if isinstance(head, LetterHead):
        tied_letters(head, read.weights, read.tokenizer)
    else:
        curve = fit_head(head, train, args.steps, args.lr, every=50)
    head.eval()
    with torch.no_grad():
        calib_logits = [head(h, [e])[0][0].float() for h, e in calib]
        dev_logits = [head(h, [e])[0][0].float() for h, e in dev]
    questions = [r.questions[0] for r in calib_rows]
    temperature = fit_temperature(calib_logits, [q.label for q in questions], [q.target for q in questions])

    def answers(scale: float) -> list[Answer]:
        return [
            Answer((s / scale).softmax(-1).tolist(), q.label, q.type, q.source, q.target, q.teacher)
            for s, q in zip(dev_logits, (r.questions[0] for r in dev_rows), strict=True)
        ]

    result, before = summarize(answers(temperature)), summarize(answers(1.0))
    report = {
        "mode": "A (zero-shot letters)" if style == "letters" else f"C (frozen backbone + {args.head_kind} head)",
        "device": str(read.backbone.device),
        "rows": {"train": len(train), "calibration": len(calib), "dev": len(dev)},
        "max_options": limit,
        "trainable_params": sum(p.numel() for p in head.parameters()),
        "curve": curve,
        "temperature": temperature,
        "dev": result,
        "dev_uncalibrated": {k: before[k] for k in ("accuracy", "nll", "brier", "ece")},
        "backbone_ms_per_question": round(1000 * read.seconds / max(len(dev), 1), 1),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    keys = ("accuracy", "nll", "brier", "ece", "coverage_at_5%_error", "accuracy_by_type", "ece_by_type")
    shown = {**{k: v for k, v in report.items() if k not in ("dev", "curve")}, "dev": {k: result[k] for k in keys}}
    print(json.dumps(shown, indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0
