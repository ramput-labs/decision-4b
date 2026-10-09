"""`den train`: LoRA + head training on data/clean, driven by a Hugging Face `Trainer` (which Unsloth patches).

The Trainer owns the loop: bf16, gradient accumulation, clipping, the cosine schedule with warmup and length-grouped
batches. This module adds what is ours: dev validation while training, the lowest-dev-NLL checkpoint in `best/`, the
temperature fit on the calibration split, and the saved run (adapter, head, `run.json`; with `--merge` the merged
weights and `integrity.json`). `--dry-run` only tokenizes the data, so it runs on any machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .catalog import record_path, role
from .data import KEV_TRAIN, UNCAPPED, Manifest, Shuffled, describe, examples, gather, licence_counts, load
from .device import hidden_size
from .licences import RANK
from .paths import CLEAN, LOCKS, locate, model_dir, read_json
from .prompt import LETTERS, MAX_STATE_TOKENS, Example, Style

CONTINUE_FILES = ["adapter_model.safetensors", "adapter_config.json", "head.safetensors", "head.json", "run.json"]
REPORT_METRICS = ("questions", "accuracy", "nll", "brier", "ece", "ece_by_type", "accuracy_by_type", "kl_to_target")
MIN_TYPE_QUESTIONS = 50  # --calibrate-by-type: fewer calibration questions of a type share the global T

type Scores = tuple[list[Any], list[int], list[tuple[float, ...] | None]]  # per question: logits, label, soft target


def pinned(model: str) -> dict[str, str | None]:
    """The base model's pinned Hub repo and revision (None for a model outside pins.py)."""
    from .pins import MODELS

    found = MODELS.get(model)
    return {"key": model, "repo": found.source.repo if found else None,
            "revision": found.source.revision if found else None}  # fmt: skip


def base_files(model: str) -> dict[str, str]:
    """The base checkpoint's files and sha256 as `den model` locked them (empty if it isn't locked)."""
    files = read_json(LOCKS / "models.json").get("entries", {}).get(model, {}).get("files", {})
    return {path: f["sha256"] for path, f in sorted(files.items())}


def dataset_hash(used: Manifest) -> str:
    """One sha256 over every training file's sha256 and the lines taken from it: equal hashes, equal data."""
    return hashlib.sha256(json.dumps(sorted(used, key=lambda u: str(u["path"])), sort_keys=True).encode()).hexdigest()


def train_history(log: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Trainer's log as plain JSON."""
    return [
        {k: v if isinstance(v, int | str) else round(float(v), 6) for k, v in row.items() if v is not None}
        for row in log
    ]


class Scorer:
    """Option scores and metrics for the weights the model holds now."""

    def __init__(self, model: Any, batch: int, bf16: bool) -> None:  # noqa: ANN401 (SystemOne, typed lazily)
        self.model, self.batch, self.bf16 = model, batch, bf16

    def predict(self, rows: list[Example]) -> Scores:
        import torch

        from .model import collate

        self.model.eval()
        scores, labels, targets = [], [], []
        device = next(self.model.parameters()).device  # where the Trainer put it, not where it was loaded
        dtype = torch.bfloat16 if self.bf16 else torch.float16
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, len(rows), self.batch):
                b = collate(rows[i : i + self.batch])
                per = self.model.scores(b["input_ids"].to(device), b["attention_mask"].to(device), b["examples"])
                for example, s in zip(b["examples"], per, strict=True):
                    scores += [q.float().cpu() for q in s]
                    labels += example.labels
                    targets += example.targets
        self.model.train()
        return scores, labels, targets

    @staticmethod
    def summary(rows: list[Example], scored: Scores, temperature: float, by_type: dict[str, float]) -> dict[str, Any]:
        """`metrics.summarize` at T (the question type's own T where one was fitted)."""
        from .metrics import Answer, summarize

        types = [k for e in rows for k in (e.kinds or ("choice",) * len(e.labels))]
        sources = [s for e in rows for s in (e.sources or ("",) * len(e.labels))]
        teachers = [t for e in rows for t in (e.teachers or (False,) * len(e.labels))]
        got = summarize([
            Answer((s / by_type.get(k, temperature)).softmax(-1).tolist(), y, k, src, t, teacher)
            for s, y, t, k, src, teacher in zip(*scored, types, sources, teachers, strict=True)
        ])  # fmt: skip
        return {m: got[m] for m in REPORT_METRICS if m in got}

    def nll_accuracy(self, rows: list[Example]) -> tuple[float, float]:
        got = self.summary(rows, self.predict(rows), 1.0, {})
        return got["nll"], got["accuracy"]

    def calibrate(
        self, calibration: list[Example], dev: list[Example], args: argparse.Namespace, label: str = ""
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """T fitted on the calibration split (and per question type with --calibrate-by-type), with the calibration
        split and dev scored before and after it."""
        from .calibrate import fit_temperature

        temperature, by_type, report = 1.0, dict[str, float](), dict[str, Any]()
        if calibration:
            scored = self.predict(calibration)
            temperature = fit_temperature(*scored)
            if args.calibrate_by_type:
                types = [k for e in calibration for k in (e.kinds or ("choice",) * len(e.labels))]
                for kind in sorted(set(types)):
                    pick = [i for i, k in enumerate(types) if k == kind]
                    if len(pick) >= MIN_TYPE_QUESTIONS:
                        logits, labels, targets = scored
                        by_type[kind] = fit_temperature(
                            [logits[i] for i in pick], [labels[i] for i in pick], [targets[i] for i in pick]
                        )
            report["calibration_split"] = {
                "files": args.calibration,
                "before": self.summary(calibration, scored, 1.0, {}),
                "after": self.summary(calibration, scored, temperature, by_type),
            }
            print(f"{label}calibration  T {temperature:.3f}" + "".join(f"  {k} {t:.3f}" for k, t in by_type.items()))
        report |= {"temperature": temperature, "temperatures": by_type}
        if dev:
            scored = self.predict(dev)
            report["before"], report["after"] = (
                self.summary(dev, scored, 1.0, {}),
                self.summary(dev, scored, temperature, by_type),
            )
            before, after = report["before"], report["after"]
            print(
                f"{label}dev at T  nll {after['nll']:.4f}  acc {after['accuracy']:.4f}  brier {after['brier']:.4f}  "
                f"ece {after['ece']:.4f}   (T=1: nll {before['nll']:.4f}  ece {before['ece']:.4f})",
                flush=True,
            )
        return temperature, by_type, report


class Best:
    """The lowest dev NLL (T = 1) seen so far, its weights mirrored to `best/` so `--resume` keeps it."""

    def __init__(self, directory: Path, resume: bool) -> None:
        self.dir = directory
        self.selection: dict[str, Any] = {}
        self.considered: list[int] = []
        if resume and (directory / "selection.json").is_file():
            self.selection = read_json(directory / "selection.json")
        elif directory.exists():
            shutil.rmtree(directory)  # an earlier run's, in the same --out

    def consider(self, model: Any, step: int, nll: float, accuracy: float) -> None:  # noqa: ANN401
        from .trainer import save_trainable

        self.considered.append(step)
        if not self.selection or nll < self.selection["nll"]:
            self.selection.update(step=step, nll=round(nll, 4), accuracy=round(accuracy, 4))
            save_trainable(model, self.dir)
            (self.dir / "selection.json").write_text(json.dumps(self.selection) + "\n", encoding="utf-8")


class Monitor:
    """Validation while training (a fixed dev sample every --eval-steps steps, all of dev at each epoch's end) and
    the --max-minutes deadline."""

    def __init__(
        self, scorer: Scorer, best: Best, probe: list[Example], dev: list[Example], every: int, minutes: float
    ) -> None:
        self.scorer, self.best, self.probe, self.dev, self.every, self.minutes = (
            scorer,
            best,
            probe,
            dev,
            every,
            minutes,
        )
        self.history: list[dict[str, float]] = []
        self.stopped: int | None = None
        self.started = time.time()

    def validate(self, step: int, rows: list[Example], select: bool, **mark: int) -> tuple[float, float]:
        nll, acc = self.scorer.nll_accuracy(rows)
        self.history.append({"step": step, "nll": round(nll, 4), "accuracy": round(acc, 4), **mark})
        if select:
            self.best.consider(self.scorer.model, step, nll, acc)
        return nll, acc

    def callback(self, data: Shuffled) -> Any:  # noqa: ANN401 (a TrainerCallback, imported lazily)
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        monitor = self

        class Hooks(TrainerCallback):
            def on_epoch_begin(
                self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: object
            ) -> None:
                data.set_epoch(int(state.epoch or 0))  # restored from the checkpoint on --resume

            def on_step_end(
                self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: object
            ) -> None:
                step = state.global_step
                if monitor.probe and step % monitor.every == 0:
                    nll, acc = monitor.validate(step, monitor.probe, select=True)
                    print(f"\nstep {step} dev({len(monitor.probe)})  nll {nll:.4f}  acc {acc:.4f}", flush=True)
                if monitor.minutes and time.time() - monitor.started > 60 * monitor.minutes:
                    monitor.stopped = step
                    print(f"\nstopping at step {step}: --max-minutes {monitor.minutes} reached", flush=True)
                    control.should_training_stop = True

            def on_epoch_end(
                self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: object
            ) -> None:
                if monitor.dev:
                    nll, acc = monitor.validate(state.global_step, monitor.dev, select=not monitor.probe, full=1)
                    print(f"\nepoch dev  nll {nll:.4f}  acc {acc:.4f}", flush=True)

        return Hooks()


def train(
    args: argparse.Namespace,
    model_path: Path,
    data: Shuffled,
    dev: list[Example],
    calibration: list[Example],
    data_used: Manifest,
) -> None:
    from . import lora as lora_module
    from .head import HeadConfig, make_head, save_head
    from .lora import LORA_PRESETS, LoraConfig, adapted, save_merged, warm_start

    lora = LoraConfig(args.lora, args.lora_alpha or 2 * args.lora, targets=LORA_PRESETS[args.lora_targets],
                      rslora=args.rslora)  # fmt: skip
    config = HeadConfig(args.head_kind, args.head_dim, args.head_layers, args.head_heads, rep=args.option_rep,
                        proj=args.head_proj)  # fmt: skip
    longest = max([data.longest, *(len(e.ids) for e in dev + calibration)])
    backbone = lora_module.load_backbone(model_path, lora, longest, args.seed, args.engine)  # before transformers

    import torch
    from safetensors.torch import load_file
    from transformers import TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from .doctor import commit, runtime, versions
    from .model import SystemOne, collate
    from .trainer import TRAINABLE, PointerTrainer, restore, trainable_state

    head = make_head(hidden_size(model_path), config)
    if config.kind == "letters":  # Qwen's own answer-letter logits: the tied embedding rows of " A" ... " Z"
        vocab = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        ids = [vocab.encode(f" {c}", add_special_tokens=False).ids[0] for c in LETTERS]
        head.set_letters(backbone.get_input_embeddings().weight[ids])  # type: ignore[operator]
    model = SystemOne(backbone, head, args.ordinal_weight)
    model.to(next(backbone.parameters()).device)
    if args.init_from:
        previous = warm_start(backbone, model.head, args.init_from, args.model, lora)
        print(f"init   from {args.init_from} (its T {previous:.3f})", flush=True)
    cuda = torch.cuda.is_available()
    bf16 = importlib.import_module("unsloth").is_bfloat16_supported() if args.engine == "unsloth" else True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.epochs and not trainable:
        raise SystemExit("nothing to train (--lora 0 with --head-kind letters): use --epochs 0 for zero-shot Qwen")
    setup = {
        "engine": args.engine,
        "dtype": "bf16" if bf16 else "fp16",
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "lora_modules": len(adapted(backbone)),
        "trainable_params": trainable,
        "head_params": sum(p.numel() for p in model.head.parameters()),
        # fp32: bf16 adapters would round 2e-5 updates away; Unsloth and PEFT both upcast LoRA for training
        "trainable_dtype": ",".join(
            sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters() if p.requires_grad})
        ),
        "records_per_step": args.batch * args.accum,
        "planned_steps": args.max_steps or math.ceil(len(data) / (args.batch * args.accum)) * args.epochs,
    }
    print("setup  " + "  ".join(f"{k} {v}" for k, v in setup.items()), flush=True)

    out = Path(args.out)
    scorer = Scorer(model, args.batch, bf16)
    probe = random.Random(args.seed).sample(dev, min(args.eval_max, len(dev))) if args.eval_steps else []
    chosen = probe or dev  # the rows `best` is chosen on
    best = Best(out / "best", bool(args.resume))
    monitor = Monitor(scorer, best, probe, dev, args.eval_steps, args.max_minutes)
    trainer = None
    if args.epochs:
        trainer = PointerTrainer(
            lengths=data.lengths,
            head_lr=args.head_lr or args.lr,
            model=model,
            args=TrainingArguments(
                output_dir=args.out,
                per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=args.accum,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps or -1,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                lr_scheduler_type="cosine",
                warmup_steps=args.warmup,  # a fraction of all steps
                max_grad_norm=1.0,
                bf16=bf16,
                fp16=not bf16,
                logging_steps=10,
                save_strategy="steps" if args.save_steps else "no",  # the run itself is saved below
                save_steps=args.save_steps or 500,
                save_total_limit=2,
                report_to=args.report_to,
                run_name=out.name,
                remove_unused_columns=False,  # the head needs every example's positions
                dataloader_pin_memory=cuda,
                tf32=cuda,
                use_cpu=not cuda,
                seed=args.seed,
                include_num_input_tokens_seen=True,
            ),
            train_dataset=data,
            data_collator=collate,
            callbacks=[monitor.callback(data)],
        )
    resume = None
    if args.resume and trainer:
        resume = get_last_checkpoint(args.out) if args.resume == "latest" else args.resume  # type: ignore[no-untyped-call]
        if resume is None:
            raise SystemExit(f"--resume latest: no checkpoint-* in {args.out}")
        print(f"resume from {resume}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume) if trainer else None  # --epochs 0: zero-shot
    minutes = (time.time() - monitor.started) / 60
    seen = int(trainer.state.num_input_tokens_seen) if trainer else 0
    final_step = trainer.state.global_step if trainer else 0
    if trainer and chosen and final_step not in best.considered:
        monitor.validate(final_step, chosen, select=True, final=1)

    temperature, temperatures, report = scorer.calibrate(calibration, dev, args)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "base": args.model,
        "lora": lora.rank,
        "lora_alpha": lora.alpha,
        "rslora": lora.rslora,
        "engine": args.engine,
    }
    if lora.rank:
        backbone.save_pretrained(out)
    save_head(out, model.head, **meta, temperature=temperature, temperatures=temperatures)
    (out / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    selection = None
    if b := best.selection:
        selection = {**b, "on": "dev sample" if probe else "dev", "questions": sum(len(e.labels) for e in chosen),
                     "metric": "nll at T=1"}  # fmt: skip
        if b["step"] == final_step:
            shutil.rmtree(best.dir, ignore_errors=True)
            selection["path"] = "."
        else:  # give the best weights their own T, then bring the final weights back for the merge
            final = trainable_state(model)
            restore(model, load_file(best.dir / TRAINABLE), str(best.dir))
            best_t, best_ts, best_report = scorer.calibrate(calibration, dev, args, f"best (step {b['step']}) ")
            if lora.rank:
                backbone.save_pretrained(best.dir)
            save_head(best.dir, model.head, **meta, temperature=best_t, temperatures=best_ts, step=b["step"])
            restore(model, final, "the final weights")
            for name in (TRAINABLE, "selection.json"):
                (best.dir / name).unlink()
            selection |= {"path": "best", "temperature": best_t, "calibration_report": best_report}
        print(f"best   step {b['step']} of {final_step}  dev nll {b['nll']:.4f}  acc {b['accuracy']:.4f}")
    run = {
        **setup,
        "base": args.model,
        "model": pinned(args.model),
        "base_files": base_files(args.model),
        "data": args.data,
        "dev_files": args.dev,
        "calibration": args.calibration,
        "sources_cap": args.sources_cap,
        "replay": args.replay,
        "augment": args.augment,
        "p_none_pair": args.p_none_pair,
        "init_from": str(args.init_from) if args.init_from else None,
        "init_from_spec": args.init_spec,  # release:<version> or hf:<repo>@<rev> when continued from a published one
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "overfit": args.overfit,
        "head_kind": args.head_kind,
        "ordinal_weight": args.ordinal_weight,
        "max_licence": args.max_licence,
        "lora": {
            "rank": lora.rank,
            "alpha": lora.alpha,
            "rslora": lora.rslora,
            "dropout": lora.dropout,
            "targets": list(lora.targets),
            "modules": len(adapted(backbone)),
        },
        "head": {
            "kind": config.kind,
            "dim": config.dim,
            "layers": config.layers,
            "heads": config.heads,
            "rep": config.option_rep,
            "proj": config.proj,
            "dropout": config.dropout,
        },
        "train_minutes": round(minutes, 1),
        "tokens_seen": seen,
        "tokens_per_second": round(seen / max(60 * minutes, 1e-9)),
        "train_loss": result.training_loss if result else None,
        "steps": result.global_step if result else 0,
        "stopped_by_max_minutes": monitor.stopped is not None,
        "temperature": temperature,
        "calibration_report": report,
        "dev_history": monitor.history,
        "train_history": train_history(trainer.state.log_history if trainer else []),
        "best": selection,
        "data_used": data_used,
        "dataset_sha256": dataset_hash(data_used),
        "licences": licence_counts(data_used),
        "runtime": runtime(),
        "versions": versions(),
        "commit": commit(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}", flush=True)
    if lora.rank and args.merge:  # last: the adapter and head are already safe if this fails
        print(f"saved {out / 'merged'}  ({save_merged(backbone, model_path, out / 'merged')} merged weights)")
        from .integrity import write

        if not write(out, model_path)["ok"]:
            raise SystemExit(f"{out}: the merged run failed its integrity checks (integrity.json)")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="den train")
    add = p.add_argument
    add("--model", default="qwen3.5-4b")
    add("--data", nargs="*", default=[f"train/{s}.jsonl" for s in KEV_TRAIN], help="files under data/clean")
    add("--sources-cap", type=int, default=0, help="also up to this many records of each normalized source (0: none)")
    add("--dev", nargs="*", default=["dev/core.jsonl"], help="files under data/clean; never test")
    add("--calibration", nargs="*", default=["calibration/core.jsonl"], help="files under data/clean to fit T on")
    add("--out", default="runs/qwen3.5-4b-lora")
    add("--epochs", type=int, default=1)
    add("--batch", type=int, default=2)
    add("--accum", type=int, default=4)
    add("--lr", type=float, default=5e-5, help="peak learning rate (Kev-4B: 5e-5 from the base, 2e-5 after)")
    add("--head-lr", type=float, default=0, help="the head's peak learning rate; 0: same as --lr")
    add("--warmup", type=float, default=0.1, help="share of steps spent warming up")
    add("--weight-decay", type=float, default=0.01)
    add("--lora", type=int, default=16, help="LoRA rank; 0 trains the head alone")
    add("--lora-alpha", type=int, default=0, help="LoRA alpha; 0: twice the rank")
    add("--rslora", action="store_true", help="scale by alpha / sqrt(rank)")
    add("--lora-targets", choices=("all", "attention-mlp", "attention"), default="all")
    add("--engine", choices=("unsloth", "peft"), default="unsloth", help="peft runs the same adapters anywhere")
    add("--head-kind", choices=("set", "pointer", "letters"), default="set")
    add("--head-dim", type=int, default=256)
    add("--head-layers", type=int, default=1, help="option-mixing blocks")
    add("--head-heads", type=int, default=4)
    add("--head-proj", choices=("linear", "mlp"), default="linear")
    add("--option-rep", choices=("end", "marker", "mean", "attn"), help="option key (default: the kind's own)")
    add("--ordinal-weight", type=float, default=0.0, help="ranked probability score weight on score questions")
    add("--calibrate-by-type", action="store_true", help="also fit one T per question type")
    add("--augment", choices=("kev", "shuffle", "none"), default="kev", help="choice-option augmentation")
    add("--p-none-pair", type=float, default=0.0, help="share of records that also train a none minimal pair")
    add("--replay", type=int, default=0, help="also this many records sampled from each --replay-from file")
    add("--replay-from", nargs="*", default=["train/core.jsonl"], help="files under data/clean to replay from")
    add("--max-licence", choices=tuple(RANK), help="train only on records licensed at most this class")
    add("--eval-steps", type=int, default=0, help="validate on a dev sample every N steps (0: epoch ends only)")
    add("--eval-max", type=int, default=600, help="questions in that dev sample")
    add("--init-from", help="continue from a run: a directory, hf:<org>/<name>@<rev> or release:<version>")
    add("--overfit", type=int, default=0, help="gate: train on N rows and score on the same N rows")
    add("--limit", type=int, default=0, help="at most this many records from any one file: a small rehearsal")
    add("--max-steps", type=int, default=0, help="stop after this many optimizer steps")
    add("--max-minutes", type=float, default=0, help="end training after this many minutes, then calibrate and save")
    add("--save-steps", type=int, default=0, help="checkpoint every N steps (adapter, head, optimizer)")
    add("--resume", help="continue from a checkpoint directory, or `latest` in --out")
    add("--merge", action="store_true", help="also save the LoRA merged into bf16 weights (merged/)")
    add("--report-to", nargs="+", default=["none"], help="Trainer trackers, e.g. trackio, wandb, tensorboard")
    add("--max-state", type=int, default=MAX_STATE_TOKENS)
    add("--seed", type=int, default=1)
    add("--dry-run", action="store_true", help="tokenize and report; no model, no GPU")
    p.set_defaults(init_spec=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("data", "dev", "calibration", "replay_from"):  # `..` resolved, so the guards see the real role
        setattr(args, name, [record_path(d) for d in getattr(args, name)])
    if any(role(d) == "test" for d in [*args.data, *args.dev, *args.calibration]):
        raise SystemExit("test partitions are read once per release candidate, never while training")
    if any(role(d) != "calibration" for d in args.calibration):
        raise SystemExit("T is fitted on calibration files only, never on train or dev")
    if args.replay and (twice := set(args.replay_from) & set(args.data)):
        raise SystemExit(f"--replay samples {sorted(twice)}, which --data already trains on in full")
    if any(role(f) != "train" for f in args.replay_from):
        raise SystemExit("--replay-from takes train/ files only")
    args.init_spec = args.init_from
    if args.init_from and not args.dry_run:
        args.init_from = locate(args.init_from, CONTINUE_FILES)
        if not (args.init_from / "head.json").is_file():
            raise SystemExit(f"--init-from {args.init_spec}: no head.json there")

    path = model_dir(args.model)
    if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
        raise SystemExit(f"{path} is not downloaded: make model MODEL={args.model}")
    every = [*args.data, *args.dev, *args.calibration, *(args.replay_from if args.replay else [])]
    if missing := [d for d in every if not (CLEAN / d).is_file()]:
        raise SystemExit(f"missing under {CLEAN}: {missing} (make data-download)")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    data_used: Manifest = []
    records = gather(args, data_used)
    style: Style = "letters" if args.head_kind == "letters" else "dash"
    if args.overfit:  # a fixed slice, unaugmented, scored on itself
        records = random.Random(args.seed).sample(records, min(args.overfit, len(records)))
        args.augment, args.p_none_pair = "none", 0.0
    data = Shuffled(records, tokenizer, args.max_state, args.seed, args.augment, args.p_none_pair, style)
    # dev and calibration are read as `den evaluate` and `den serve` read them: --max-state only limits training
    dev, dev_skipped = examples(data.records if args.overfit else load([CLEAN / d for d in args.dev]), tokenizer,
                                UNCAPPED, style=style)  # fmt: skip
    calibration, cal_skipped = examples(load([CLEAN / d for d in args.calibration]), tokenizer, UNCAPPED, style=style)
    print(
        describe("train", data.lengths, sum(len(r.questions) for r in data.records), len(records) - len(data.records))
    )
    for name, rows, skipped in (("dev", dev, dev_skipped), ("calib", calibration, cal_skipped)):
        print(describe(name, [len(e.ids) for e in rows], sum(len(e.labels) for e in rows), skipped))
    if not args.dry_run:
        train(args, path, data, dev, calibration, data_used)
    return 0
