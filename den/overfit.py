"""The overfit gate: can the pointer head fit 100 real examples on the backbone's real hidden states?

    den overfit                                   # 100 rows of train/core, the default head, on this machine's backend
    den overfit --head-kind pointer --steps 400   # the same gate for another head

The backbone reads each row once (frozen, through `device.load`: MLX, CUDA or CPU); the head then trains on those
hidden states until it fits. A head that can't reach ~100% here has a bug in its positions, labels, masking, loss or
gradients, and no amount of LoRA will fix that. Once it fits, the same trained head is put through behavioral checks:

- determinism: the same input gives the same scores
- permutation: with the options in reverse order (re-read by the backbone), does the prediction follow the option's
  content or its position? (`content_agreement`; a position-only scorer would score near chance)
- replacement: with the predicted option's text replaced by an unrelated one, does its probability fall?

On GPU, `den train --overfit 100` runs the same gate with LoRA through the real training loop.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .api import Question, Record
from .catalog import record_path, role
from .evaluate import base_weights
from .metrics import Answer, summarize
from .model import HeadConfig, make_head, question_loss
from .prompt import Example, encode, split
from .train import CLEAN, load

UNRELATED = "A recipe for pancakes with blueberries"


def reversed_options(q: Question) -> Question:
    n = len(q.options)
    target = None if q.target is None else tuple(reversed(q.target))
    return replace(q, keys=q.keys[::-1], options=q.options[::-1], label=n - 1 - q.label, target=target)


def replaced_option(q: Question, i: int) -> Question:
    options = list(q.options)
    options[i] = f"{q.keys[i]}: {UNRELATED}" if ":" in options[i] else UNRELATED
    return replace(q, options=tuple(options))


type Scorer = Callable[[Record], torch.Tensor]  # a row -> its option probabilities


def _frozen(
    args: argparse.Namespace, variants: list[Record], config: HeadConfig
) -> tuple[Scorer, list[dict[str, float]], dict[str, Any]]:
    """Head only, on cached features: the backbone (`device.load`: MLX, CUDA or CPU) reads every row once."""
    from tokenizers import Tokenizer

    from .device import load as load_features

    weights = base_weights(args.model)
    backbone = load_features(weights, args.backend)
    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))

    def read(record: Record) -> tuple[torch.Tensor, Example]:
        example = encode(record, tokenizer, max_state=1 << 30)
        assert example is not None
        return torch.from_numpy(backbone.hidden_states(example.ids))[None], example

    cached = [read(r) for r in variants]
    print(f"read {len(variants)} rows on {backbone.device}", flush=True)
    head = make_head(backbone.hidden_size, config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    curve: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        losses, right = [], 0
        for hidden, example in cached:
            (scores,) = head(hidden, [example])[0]
            losses.append(question_loss(scores, example.labels[0], None))
            right += int(scores.argmax()) == example.labels[0]
        loss = torch.stack(losses).mean()
        loss.backward()  # type: ignore[no-untyped-call]
        silent = [n for n, p in head.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        if silent:
            raise SystemExit(f"step {step}: head parameters without a finite gradient: {silent}")
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            value = loss.item()
            curve.append({"step": step, "loss": round(value, 4), "accuracy": right / len(cached)})
            print(f"step {step:>4}  loss {value:.4f}  train acc {right / len(cached):.3f}", flush=True)
    head.eval()

    def probs(record: Record) -> torch.Tensor:
        hidden, example = read(record)
        with torch.no_grad():
            scores: torch.Tensor = head(hidden, [example])[0][0]
        return scores.float().softmax(-1)

    info = {
        "path": "frozen",
        "device": str(backbone.device),
        "trainable_params": sum(p.numel() for p in head.parameters()),
    }
    return probs, curve, info


def _lora(
    args: argparse.Namespace, variants: list[Record], config: HeadConfig
) -> tuple[Scorer, list[dict[str, float]], dict[str, Any]]:
    """The training path itself: Qwen + LoRA (Unsloth on CUDA) + head, BF16 autocast, backward through the backbone
    every step, minibatches of --batch rows. Step 1 checks the wiring: a finite loss, nonzero LoRA-B and head
    gradients, no gradient on the frozen base, and an optimizer holding exactly the trainable parameters."""
    from tokenizers import Tokenizer

    from .device import hidden_size
    from .model import Engine, LoraConfig, SystemOne, adapted, collate, load_backbone, param_groups

    weights = base_weights(args.model)
    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    examples = [encode(r, tokenizer, max_state=1 << 30) for r in variants]
    assert all(e is not None for e in examples)
    rows: list[Example] = [e for e in examples if e is not None]
    engine: Engine = args.engine or ("unsloth" if torch.cuda.is_available() else "peft")
    backbone = load_backbone(weights, LoraConfig(), max(len(e.ids) for e in rows) + 8, args.seed, engine)
    model = SystemOne(backbone, make_head(hidden_size(weights), config))
    device = next(backbone.parameters()).device
    model.to(device)
    modules = adapted(backbone)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if any(".visual." in n for n in modules) or not modules:
        raise SystemExit(f"LoRA is on the wrong modules: {len(modules)} adapted, e.g. {modules[:2]}")
    if {str(p.dtype) for p in trainable.values()} != {"torch.float32"}:
        raise SystemExit(f"trainable parameters must be fp32, got {sorted({str(p.dtype) for p in trainable.values()})}")
    frozen = [n for n, p in backbone.named_parameters() if "lora_" not in n and p.requires_grad]
    if frozen:
        raise SystemExit(f"base weights are trainable: {frozen[:3]}")
    groups = param_groups(model, args.lora_lr, args.lr, 0.0, set())  # refuses any mismatch with `trainable`
    optimizer = torch.optim.AdamW(groups)
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if not cuda or torch.cuda.is_bf16_supported() else torch.float16
    count = sum(p.numel() for p in trainable.values())
    base_dtype = next(iter(backbone.parameters())).dtype
    print(
        f"lora   engine {engine}  device {device}  autocast {dtype}  modules {len(modules)}  trainable {count}  "
        f"base dtype {base_dtype}",
        flush=True,
    )

    def scores(batch: list[Example]) -> list[list[torch.Tensor]]:
        b = collate(batch)
        out: list[list[torch.Tensor]] = model.scores(b["input_ids"].to(device), b["attention_mask"].to(device), batch)
        return out

    def accuracy() -> float:
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            right = sum(
                int(s.argmax()) == e.labels[0]
                for i in range(0, len(rows), args.batch)
                for e, (s,) in zip(rows[i : i + args.batch], scores(rows[i : i + args.batch]), strict=True)
            )
        model.train()
        return right / len(rows)

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    curve: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        start = (step - 1) * args.batch % len(order)
        batch = [rows[order[(start + j) % len(order)]] for j in range(min(args.batch, len(order)))]
        optimizer.zero_grad(set_to_none=True)
        b = collate(batch)
        with torch.autocast(device.type, dtype=dtype):
            loss = model(b["input_ids"].to(device), b["attention_mask"].to(device), batch)["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"step {step}: loss is {loss.item()}")
        loss.backward()
        if step == 1:
            bad = [n for n, p in trainable.items() if p.grad is None or not torch.isfinite(p.grad).all()]
            lora_b = [p.grad for n, p in trainable.items() if "lora_B" in n and p.grad is not None]
            head = [p.grad for n, p in trainable.items() if n.startswith("head.") and p.grad is not None]
            leaked = [n for n, p in backbone.named_parameters() if "lora_" not in n and p.grad is not None]
            moving_lora = any(float(g.abs().sum()) > 0 for g in lora_b)
            moving_head = any(float(g.abs().sum()) > 0 for g in head)
            if bad or leaked or not moving_lora or not moving_head:
                raise SystemExit(
                    f"step 1 gradients: missing/non-finite {bad[:3]}, frozen with grads {leaked[:3]}, "
                    f"lora_B nonzero {moving_lora}, head nonzero {moving_head}"
                )
            print("step 1 gradients: finite, LoRA-B and head nonzero, frozen base untouched", flush=True)
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            acc = accuracy()
            curve.append({"step": step, "loss": round(loss.item(), 4), "accuracy": acc})
            print(f"step {step:>4}  batch loss {loss.item():.4f}  train acc {acc:.3f}", flush=True)
    model.eval()

    def probs(record: Record) -> torch.Tensor:
        example = encode(record, tokenizer, max_state=1 << 30)
        assert example is not None
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            (s,) = scores([example])[0]
        return s.float().softmax(-1).cpu()

    info: dict[str, Any] = {
        "path": "lora",
        "engine": engine,
        "device": str(device),
        "lora_modules": len(modules),
        "trainable_params": count,
        "base_dtype": str(base_dtype),
        "autocast": str(dtype),
    }
    if cuda:
        info["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    return probs, curve, info


def overfit_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den overfit")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--file", default="train/core.jsonl", help="a file under data/clean")
    p.add_argument("--n", type=int, default=100, help="rows (one question each)")
    p.add_argument("--steps", type=int, default=300, help="optimizer steps")
    p.add_argument(
        "--path",
        choices=("auto", "frozen", "lora"),
        default="auto",
        help="lora: Qwen + LoRA + head, backward through Qwen (the CUDA training path); frozen: the head on cached "
        "features (MLX/CPU friendly); auto: lora on CUDA, frozen elsewhere",
    )
    p.add_argument("--engine", choices=("unsloth", "peft"), help="lora path: default unsloth on CUDA, peft elsewhere")
    p.add_argument("--batch", type=int, default=16, help="lora path: rows per step")
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-kind", choices=("set", "pointer"), default="set")
    p.add_argument("--option-rep", choices=("end", "marker", "mean", "attn"))
    p.add_argument("--head-proj", choices=("linear", "mlp"), default="linear")
    p.add_argument("--pass-at", type=float, default=0.95, help="train accuracy the gate requires")
    p.add_argument(
        "--orders",
        type=int,
        default=1,
        help="train each row in this many option orders (random, never the reversed one the check uses)",
    )
    p.add_argument("--backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the report here as JSON")
    args = p.parse_args(argv)
    args.file = record_path(args.file)
    if role(args.file) == "test":
        raise SystemExit("overfit trains on --file: never a test file")
    path = args.path if args.path != "auto" else ("lora" if torch.cuda.is_available() else "frozen")

    torch.manual_seed(args.seed)
    rows = [one for r in load([CLEAN / args.file]) for one in split(r)]
    rows = [r for r in random.Random(args.seed).sample(rows, len(rows)) if r.questions[0].target is None][: args.n]

    def orders(record: Record, i: int) -> list[Record]:
        """The row as given, plus args.orders - 1 random option orders, never the fully reversed one."""
        q = record.questions[0]
        out, seen = [record], {tuple(range(len(q.options))), tuple(reversed(range(len(q.options))))}
        rng = random.Random(f"{args.seed}:{i}")
        for _ in range(20 * args.orders):
            if len(out) >= args.orders or q.type != "choice" or len(q.options) < 3:
                break
            order = list(range(len(q.options)))
            rng.shuffle(order)
            if tuple(order) not in seen:
                seen.add(tuple(order))
                moved = replace(q, keys=tuple(q.keys[j] for j in order), options=tuple(q.options[j] for j in order),
                                label=order.index(q.label))  # fmt: skip
                out.append(replace(record, questions=(moved,)))
        return out

    variants = [v for i, r in enumerate(rows) for v in orders(r, i)]
    config = HeadConfig(args.head_kind, rep=args.option_rep, proj=args.head_proj)
    started = time.perf_counter()
    probs, curve, info = (_lora if path == "lora" else _frozen)(args, variants, config)

    final = [probs(r) for r in rows]
    fitted = summarize(
        [Answer(pr.tolist(), r.questions[0].label, r.questions[0].type) for pr, r in zip(final, rows, strict=True)]
    )
    again = [probs(r) for r in rows[:10]]
    deterministic = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(again, final[:10], strict=True))
    agree = moved = checked = 0
    chance = position_only = 0.0
    for record, pr in zip(rows, final, strict=True):
        q = record.questions[0]
        if q.type != "choice" or len(q.options) < 3:
            continue
        checked += 1
        pick = int(pr.argmax())
        chance += 1 / len(q.options)  # a uniform guess on the reversed order
        position_only += pick == len(q.options) - 1 - pick  # a scorer of positions agrees only at the middle
        flipped = probs(replace(record, questions=(reversed_options(q),)))
        agree += q.keys[pick] == q.keys[::-1][int(flipped.argmax())]  # the same option, found at its new place
        swapped = probs(replace(record, questions=(replaced_option(q, pick),)))
        moved += float(swapped[pick]) < float(pr[pick])
    first, last = curve[0], curve[-1]
    report = {
        **info,
        "rows": len(rows),
        "head": {"kind": config.kind, "rep": config.option_rep, "proj": config.proj},
        "train_accuracy": fitted["accuracy"],
        "train_nll": fitted["nll"],
        "loss_drop": round(first["loss"] / max(last["loss"], 1e-9), 1),
        "curve": curve,
        "deterministic": deterministic,
        "choice_rows_checked": checked,
        "orders": args.orders,
        "content_agreement": agree / max(checked, 1),
        "chance_agreement": chance / max(checked, 1),
        "position_only_agreement": position_only / max(checked, 1),
        "replacement_lowers_probability": moved / max(checked, 1),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    report["passed"] = fitted["accuracy"] >= args.pass_at and deterministic and last["loss"] < 0.5 * first["loss"]
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS" if report["passed"] else f"FAIL: train accuracy {fitted['accuracy']:.3f} (needs {args.pass_at}), "
          f"loss {first['loss']} -> {last['loss']}, deterministic {deterministic}")  # fmt: skip
    return 0 if report["passed"] else 1


def probe_main(argv: list[str] | None = None) -> int:
    """Modes A and C on any machine: a frozen backbone (`device.load`: MLX, CUDA or CPU) reads train, calibration
    and dev rows once; then either the letter scorer answers zero-shot (mode A: Qwen's own answer-letter logits,
    h . E[letter] with E the checkpoint's tied embedding rows), or a pointer head trains on the train rows (mode C).
    Both fit T on the calibration rows and are scored on the same dev rows by `metrics.summarize`.

        den probe --head-kind letters --dev dev/core.jsonl --max-options 26     # mode A
        den probe --train train/core.jsonl --dev dev/core.jsonl --max-options 26  # mode C
    """
    import time

    from safetensors import safe_open
    from tokenizers import Tokenizer

    from .calibrate import fit_temperature
    from .device import load as load_backbone
    from .model import LetterHead
    from .prompt import LETTERS, Style

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
        return [
            one for r in load([CLEAN / f for f in files]) for one in split(r) if len(one.questions[0].options) <= limit
        ]

    weights = base_weights(args.model)
    backbone = load_backbone(weights, args.backend)
    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    reading = 0.0

    def read(record: Record) -> tuple[torch.Tensor, Example]:
        nonlocal reading
        example = encode(record, tokenizer, max_state=1 << 30, style=style)
        assert example is not None
        started = time.perf_counter()
        hidden = torch.from_numpy(backbone.hidden_states(example.ids))[None]
        reading += time.perf_counter() - started
        return hidden, example

    train_rows, calib_rows, dev_rows = rows(args.train), rows(args.calibration), rows(args.dev)
    started = time.perf_counter()
    train, calib = [read(r) for r in train_rows], [read(r) for r in calib_rows]
    reading = 0.0
    dev = [read(r) for r in dev_rows]
    dev_ms = 1000 * reading / max(len(dev), 1)
    print(f"read {len(train)} train, {len(calib)} calibration, {len(dev)} dev rows on {backbone.device}", flush=True)

    config = HeadConfig(args.head_kind)
    head = make_head(backbone.hidden_size, config)
    curve: list[dict[str, float]] = []
    if isinstance(head, LetterHead):
        ids = [tokenizer.encode(f" {c}", add_special_tokens=False).ids[0] for c in LETTERS]
        for shard in sorted(weights.glob("*.safetensors")):
            with safe_open(shard, "pt") as handle:
                names: list[str] = list(handle.keys())
                name = next((k for k in names if k.endswith("embed_tokens.weight")), None)
                if name and not name.startswith("mtp"):
                    head.set_letters(handle.get_tensor(name)[ids])
        assert float(head.letters.abs().sum()) > 0, "no embedding rows found"  # type: ignore[operator]
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            losses, right = [], 0
            for hidden, example in train:
                (scores,) = head(hidden, [example])[0]
                losses.append(question_loss(scores, example.labels[0], example.targets[0]))
                right += int(scores.argmax()) == example.labels[0]
            loss = torch.stack(losses).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                raise SystemExit(f"step {step}: a head gradient is missing or not finite")
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                curve.append({"step": step, "loss": round(loss.item(), 4), "accuracy": right / len(train)})
                print(f"step {step:>4}  train loss {loss.item():.4f}  train acc {right / len(train):.3f}", flush=True)
    head.eval()
    with torch.no_grad():

        def logits(cached: list[tuple[torch.Tensor, Example]]) -> list[torch.Tensor]:
            return [head(h, [e])[0][0].float() for h, e in cached]

        calib_logits, dev_logits = logits(calib), logits(dev)
    labelled = [(s, r.questions[0]) for s, r in zip(calib_logits, calib_rows, strict=True)]
    temperature = fit_temperature(
        [s for s, _ in labelled], [q.label for _, q in labelled], [q.target for _, q in labelled]
    )

    def answers(scale: float) -> list[Answer]:
        return [
            Answer(
                (s / scale).softmax(-1).tolist(),
                r.questions[0].label,
                r.questions[0].type,
                r.questions[0].source,
                r.questions[0].target,
            )
            for s, r in zip(dev_logits, dev_rows, strict=True)
        ]

    result = summarize(answers(temperature))
    before = summarize(answers(1.0))
    report = {
        "mode": "A (zero-shot letters)"
        if args.head_kind == "letters"
        else f"C (frozen backbone + {args.head_kind} head)",
        "device": str(backbone.device),
        "rows": {"train": len(train), "calibration": len(calib), "dev": len(dev)},
        "max_options": limit,
        "trainable_params": sum(p.numel() for p in head.parameters()),
        "curve": curve,
        "temperature": temperature,
        "dev": result,
        "dev_uncalibrated": {k: before[k] for k in ("accuracy", "nll", "brier", "ece")},
        "backbone_ms_per_question": round(dev_ms, 1),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    keys = ("accuracy", "nll", "brier", "ece", "coverage_at_5%_error", "accuracy_by_type", "ece_by_type")
    print(json.dumps({**{k: v for k, v in report.items() if k not in ("dev", "curve")},
                      "dev": {k: result[k] for k in keys}}, indent=2), flush=True)  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0
