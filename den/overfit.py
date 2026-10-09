"""`den overfit`, the gate before any long run: can the head fit 100 real rows?

    den overfit                                   # 100 rows of train/core; LoRA on CUDA, head-only elsewhere
    den overfit --head-kind pointer --steps 400   # the same gate for another head

A head that can't reach ~100% here has a bug in its positions, labels, masking, loss or gradients, and no amount of
LoRA will fix that. Once it fits, the same head is checked for determinism, for following an option's content rather
than its position (options reversed), and for lowering an option's probability when its text is replaced.
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
from .data import UNCAPPED, load
from .head import HeadConfig, make_head
from .metrics import Answer, summarize
from .paths import CLEAN, base_weights
from .probe import Features, fit_head
from .prompt import Example, encode, reorder, split

UNRELATED = "A recipe for pancakes with blueberries"

type Scorer = Callable[[Record], torch.Tensor]  # a row -> its option probabilities
type Fitted = tuple[Scorer, list[dict[str, float]], dict[str, Any]]


def reversed_options(q: Question) -> Question:
    return reorder(q, range(len(q.options) - 1, -1, -1))


def replaced_option(q: Question, i: int) -> Question:
    options = list(q.options)
    options[i] = f"{q.keys[i]}: {UNRELATED}" if ":" in options[i] else UNRELATED
    return replace(q, options=tuple(options))


def _frozen(args: argparse.Namespace, variants: list[Record], config: HeadConfig) -> Fitted:
    """The head alone, on hidden states the frozen backbone reads once."""
    read = Features(args.model, args.backend)
    cached = [read(r) for r in variants]
    print(f"read {len(variants)} rows on {read.backbone.device}", flush=True)
    head = make_head(read.backbone.hidden_size, config)
    curve = fit_head(head, cached, args.steps, args.lr, every=25)

    def probs(record: Record) -> torch.Tensor:
        hidden, example = read(record)
        with torch.no_grad():
            scores: torch.Tensor = head(hidden, [example])[0][0]
        return scores.float().softmax(-1)

    info = {"path": "frozen", "device": str(read.backbone.device),
            "trainable_params": sum(p.numel() for p in head.parameters())}  # fmt: skip
    return probs, curve, info


def _lora(args: argparse.Namespace, variants: list[Record], config: HeadConfig) -> Fitted:
    """The training path itself: Qwen + LoRA + head under autocast, backward through the backbone. Step 1 checks the
    wiring: a finite loss, nonzero LoRA-B and head gradients, and none on the frozen base."""
    from tokenizers import Tokenizer

    from .device import hidden_size
    from .lora import Engine, LoraConfig, adapted, load_backbone
    from .model import SystemOne, collate
    from .trainer import param_groups

    weights = base_weights(args.model)
    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    rows: list[Example] = [e for r in variants if (e := encode(r, tokenizer, max_state=UNCAPPED)) is not None]
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
    if frozen := [n for n, p in backbone.named_parameters() if "lora_" not in n and p.requires_grad]:
        raise SystemExit(f"base weights are trainable: {frozen[:3]}")
    optimizer = torch.optim.AdamW(param_groups(model, args.lora_lr, args.lr, 0.0, set()))
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if not cuda or torch.cuda.is_bf16_supported() else torch.float16
    count = sum(p.numel() for p in trainable.values())
    base_dtype = next(iter(backbone.parameters())).dtype
    print(f"lora   engine {engine}  device {device}  autocast {dtype}  modules {len(modules)}  trainable {count}  "
          f"base dtype {base_dtype}", flush=True)  # fmt: skip

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
            leaked = [n for n, p in backbone.named_parameters() if "lora_" not in n and p.grad is not None]
            moving_lora = any(
                float(p.grad.abs().sum()) > 0 for n, p in trainable.items() if "lora_B" in n and p.grad is not None
            )
            moving_head = any(
                float(p.grad.abs().sum()) > 0
                for n, p in trainable.items()
                if n.startswith("head.") and p.grad is not None
            )
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
        example = encode(record, tokenizer, max_state=UNCAPPED)
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


def orders(record: Record, n: int, rng: random.Random) -> list[Record]:
    """The row as given, plus up to n - 1 random option orders, never the fully reversed one the check uses."""
    q = record.questions[0]
    out, seen = [record], {tuple(range(len(q.options))), tuple(reversed(range(len(q.options))))}
    for _ in range(20 * n):
        if len(out) >= n or q.type != "choice" or len(q.options) < 3:
            break
        order = list(range(len(q.options)))
        rng.shuffle(order)
        if tuple(order) not in seen:
            seen.add(tuple(order))
            out.append(replace(record, questions=(reorder(q, order),)))
    return out


def behaviour(rows: list[Record], final: list[torch.Tensor], probs: Scorer) -> dict[str, float]:
    """On choice rows with 3+ options: does the prediction follow the option's content when the options are reversed,
    and does its probability fall when its text is replaced by an unrelated one?"""
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
        agree += q.keys[pick] == q.keys[::-1][int(flipped.argmax())]
        swapped = probs(replace(record, questions=(replaced_option(q, pick),)))
        moved += float(swapped[pick]) < float(pr[pick])
    n = max(checked, 1)
    return {"choice_rows_checked": checked, "content_agreement": agree / n, "chance_agreement": chance / n,
            "position_only_agreement": position_only / n, "replacement_lowers_probability": moved / n}  # fmt: skip


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
        help="lora: LoRA + head through Qwen; frozen: the head on cached features; auto: lora on CUDA",
    )
    p.add_argument("--engine", choices=("unsloth", "peft"), help="lora path: default unsloth on CUDA, peft elsewhere")
    p.add_argument("--batch", type=int, default=16, help="lora path: rows per step")
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-kind", choices=("set", "pointer"), default="set")
    p.add_argument("--option-rep", choices=("end", "marker", "mean", "attn"))
    p.add_argument("--head-proj", choices=("linear", "mlp"), default="linear")
    p.add_argument("--pass-at", type=float, default=0.95, help="train accuracy the gate requires")
    p.add_argument("--orders", type=int, default=1, help="train each row in this many random option orders")
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
    variants = [v for i, r in enumerate(rows) for v in orders(r, args.orders, random.Random(f"{args.seed}:{i}"))]
    config = HeadConfig(args.head_kind, rep=args.option_rep, proj=args.head_proj)
    started = time.perf_counter()
    probs, curve, info = (_lora if path == "lora" else _frozen)(args, variants, config)

    final = [probs(r) for r in rows]
    fitted = summarize(
        [Answer(pr.tolist(), r.questions[0].label, r.questions[0].type) for pr, r in zip(final, rows, strict=True)]
    )
    deterministic = all(torch.allclose(probs(r), b, atol=1e-5) for r, b in zip(rows[:10], final[:10], strict=True))
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
        "orders": args.orders,
        **behaviour(rows, final, probs),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    report["passed"] = fitted["accuracy"] >= args.pass_at and deterministic and last["loss"] < 0.5 * first["loss"]
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS" if report["passed"] else f"FAIL: train accuracy {fitted['accuracy']:.3f} (needs {args.pass_at}), "
          f"loss {first['loss']} -> {last['loss']}, deterministic {deterministic}")  # fmt: skip
    return 0 if report["passed"] else 1
