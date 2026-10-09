"""LoRA + pointer-head training on data/clean, driven by a Hugging Face `Trainer` (which Unsloth patches).

The Trainer owns the loop: bf16 autocast, gradient accumulation, clipping, the cosine schedule with warmup, and
length-grouped batches, so a 150-token record is not padded to 7k beside a long document. This module supplies only
what is ours: the data, the pointer-head loss (`model.SystemOne`), a learning rate for the head that differs from the
adapters', dev evaluation each epoch, the temperature fit, and the saved run.

`--dry-run` tokenizes the data and prints its shape without importing torch or unsloth, so it runs on any machine.
Training uses Unsloth, so it needs Linux and an NVIDIA GPU; `--engine peft` runs the same LoRA on any device.
`--lora 0` freezes the backbone and trains the head alone: the baseline that shows what LoRA adds. `--merge` also
writes the LoRA folded into bf16 weights, which the serving backbones load like the base model.

Choice options get Kev's augmentation on every visit (`--augment kev`: none-of-the-above and distractor options, a
fresh order), so position carries no signal; `--p-none-pair` adds Kev's minimal pairs. `--eval-steps` validates on a
dev sample while training; the curve goes into run.json with the exact data used (`data_used`) and the Trainer's loss
log (`train_history`). The checkpoint with the lowest dev NLL is kept in `best/` (adapter + head, its own T) for
`--init-from` or a look back; the final weights are what ships and merges. After `--merge`, `integrity.json` records
the merged run's checks and every model file's sha256 (`den check-run`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import shutil
import statistics
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .api import Record, option_text, read
from .catalog import record_path, role
from .device import hidden_size
from .fetch import digest
from .licences import RANK, Kind, record_kind
from .paths import CLEAN, LOCKS
from .paths import model_dir as model_dir_of
from .prompt import (
    DISTRACTORS,
    LETTERS,
    MAX_STATE_TOKENS,
    NONE_OPTIONS,
    Example,
    Style,
    augment,
    encode,
    none_pair,
    shuffle_options,
    split,
)

KEV_TRAIN = ("core", "dates-unknowable", "documents", "skills", "devtools")  # Kev's stages 1-4


def load(paths: Sequence[Path]) -> list[Record]:
    return [record for path in paths for record in read(path)]


def kinds(path: Path) -> dict[int, Kind]:
    """Each non-empty line's licence class (`licences.record_kind`), by 1-based line number."""
    with path.open(encoding="utf-8") as f:
        return {n: record_kind(json.loads(line)) for n, line in enumerate(f, 1) if line.strip()}


def licensed(path: Path, worst: Kind | None) -> list[int]:
    """The non-empty lines of `path` whose sources are no more restrictive than `worst` (all of them if None)."""
    if worst is None:
        with path.open(encoding="utf-8") as f:
            return [n for n, line in enumerate(f, 1) if line.strip()]
    return [n for n, k in kinds(path).items() if RANK[k] <= RANK[worst]]


def licence_counts(used: Sequence[dict[str, object]]) -> dict[str, int]:
    """How many trained records fall in each licence class, over the run's data manifest: what the card states."""
    counts: dict[str, int] = {}
    for entry in used:
        lines, classes = entry["lines"], kinds(CLEAN / str(entry["path"]))
        for n in lines if isinstance(lines, list) else classes:
            counts[classes[n]] = counts.get(classes[n], 0) + 1
    return {k: counts[k] for k in RANK if k in counts}


def sample(
    path: Path, cap: int, seed: int, used: list[dict[str, object]] | None = None, worst: Kind | None = None
) -> list[Record]:
    """Up to `cap` records of one file, the same ones for the same seed; only the chosen lines are parsed. The file
    and its chosen line numbers are appended to `used`, the run's data manifest. With `worst`, only records whose
    sources are no more restrictive than that licence class are eligible."""
    numbered = licensed(path, worst)
    rng = random.Random(f"{seed}:{path.as_posix()}")
    lines = sorted(rng.sample(numbered, min(cap, len(numbered))))
    if used is not None:
        used.append({"path": path.relative_to(CLEAN).as_posix(), "sha256": digest(path), "lines": lines})
    return list(read(path, set(lines)))


def sources(
    cap: int,
    seed: int,
    known: Sequence[Record] = (),
    used: list[dict[str, object]] | None = None,
    worst: Kind | None = None,
) -> list[Record]:
    """A balanced slice of every normalized trainable source: at most `cap` records each, so yelp's 644k records do
    not drown banking77's 9k. Records whose source text is already in `known` are left out: Kev's core holds 1,000
    of each of its ten public datasets, and a text trained on twice would count double. Eval-only sources never
    reach train/sources (catalog.py enforces it)."""
    seen = {r.provenance for r in known if r.provenance} | {r.fingerprint for r in known}
    return [
        r
        for path in sorted((CLEAN / "train" / "sources").rglob("*.jsonl"))
        for r in sample(path, cap, seed, used, worst)
        if r.provenance not in seen and r.fingerprint not in seen
    ]


def examples(
    records: Sequence[Record],
    tokenizer: Tokenizer,
    max_state: int,
    rng: random.Random | None = None,
    style: Style = "dash",
) -> tuple[list[Example], int]:
    """Encoded records, one per question, with choice options shuffled when `rng` is given, and the count of
    questions skipped (state too long; or, for the letter scorer, more than 26 options)."""
    out, skipped = [], 0
    for record in (one for r in records for one in split(r)):
        if (example := encode(shuffle_options(record, rng) if rng else record, tokenizer, max_state, style)) is None:
            skipped += 1
        else:
            out.append(example)
    return out, skipped


def describe(name: str, lengths: Sequence[int], questions: int, skipped: int) -> str:
    ordered = sorted(lengths)
    p99 = ordered[int(0.99 * (len(ordered) - 1))] if ordered else 0
    median = int(statistics.median(ordered)) if ordered else 0
    longest = ordered[-1] if ordered else 0
    return (
        f"{name:<8}{len(ordered):>8} records{questions:>9} questions  "
        f"tokens median {median} p99 {p99} max {longest} total {sum(ordered) / 1e6:.1f}M  skipped {skipped}"
    )


def pinned(model: str) -> dict[str, str | None]:
    """The base model's pinned Hub repo and revision (None for a model outside pins.py)."""
    from .pins import MODELS

    found = MODELS.get(model)
    return {
        "key": model,
        "repo": found.source.repo if found else None,
        "revision": found.source.revision if found else None,
    }


def base_files(model: str) -> dict[str, str]:
    """The base checkpoint's files and sha256, as `den model` placed and locked them (empty if it isn't locked)."""
    lock = LOCKS / "models.json"
    entries = json.loads(lock.read_text(encoding="utf-8"))["entries"] if lock.is_file() else {}
    files = entries.get(model, {}).get("files", {})
    return {path: f["sha256"] for path, f in sorted(files.items())}


def runtime() -> dict[str, Any]:
    """Where the run ran: platform, CUDA build, cuDNN and driver (None off CUDA)."""
    import platform

    import torch

    from .doctor import driver

    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "cudnn": None,
        "driver": None,
        "gpu_count": 0,
    }
    if torch.cuda.is_available():
        info |= {"cudnn": torch.backends.cudnn.version(), "driver": driver(), "gpu_count": torch.cuda.device_count()}  # type: ignore[no-untyped-call]
        info |= {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),  # this process, over the whole run
        }
    return info


def dataset_hash(used: list[dict[str, object]]) -> str:
    """One sha256 over every training file's sha256 and the lines taken from it: equal hashes, equal data."""
    import hashlib

    return hashlib.sha256(json.dumps(sorted(used, key=lambda u: str(u["path"])), sort_keys=True).encode()).hexdigest()


def versions() -> dict[str, str]:
    """The packages that decide the result, as installed (`den doctor`'s list, and den itself)."""
    from .doctor import PACKAGES, package_version

    return {name: v for name in (*PACKAGES, "den") if (v := package_version(name))}


def commit() -> str:
    """The git commit of this code, with `+dirty` when there are uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty.strip() else "")


def added(tokenizer: Tokenizer, style: Style) -> int:
    """The most tokens Kev's augmentation adds to a question: its longest "none" or distractor option line, plus one
    for a merge across the line boundary."""
    marker = "Z) " if style == "letters" else "- "
    lines = [f"{marker}{option_text(k, d)}\n" for k, d in (*NONE_OPTIONS, *DISTRACTORS)]
    return 1 + max(len(tokenizer.encode(line, add_special_tokens=False).ids) for line in lines)


class Shuffled:
    """Training items encoded on access, a fresh variant each epoch, so each epoch sees new option orders.

    `augment`: "kev" (Kev's augmentation: none-of-the-above and distractor options, shuffled order), "shuffle" (order
    only) or "none". With `p_none_pair`, that share of the records with an eligible choice question also yields Kev's
    minimal pair: two more items, the question with the true option present and removed, sharing one "none" option
    and one order per epoch. Records whose state is too long are dropped; `lengths` feeds the length-grouped sampler.

    The variant is a function of (seed, item, epoch) alone, never of how often an item was read: a resumed run skips
    batches without reading them, and dataloader workers read copies, so a visit count would drift from the
    uninterrupted run. The Trainer sets the epoch (`set_epoch`, through `Epochs`) before each epoch's batches.
    """

    def __init__(
        self,
        records: Sequence[Record],
        tokenizer: Tokenizer,
        max_state: int,
        seed: int,
        augment: str = "kev",
        p_none_pair: float = 0.0,
        style: Style = "dash",
    ) -> None:
        self.records: list[Record] = []
        self.longest = 0  # the longest an item can get once augmented: the backbone's max_seq_length
        self.items: list[tuple[int, int]] = []  # (record, part): part 0 the record, 1 and 2 its pair's halves
        self.lengths: list[int] = []
        self.tokenizer, self.seed, self.augment, self.style = tokenizer, seed, augment, style
        grown = added(tokenizer, style) if augment == "kev" else 0
        for record in records:
            if (example := encode(record, tokenizer, max_state, style)) is None:
                continue
            i = len(self.records)
            self.records.append(record)
            self.items.append((i, 0))
            self.lengths.append(len(example.ids))
            self.longest = max(self.longest, len(example.ids) + grown)
            draw = random.Random(f"{seed}:{i}:pair").random()
            if draw < p_none_pair and (pair := none_pair(record, random.Random(0))) is not None:
                halves = [encode(half, tokenizer, 1 << 30, style) for half in pair]
                if all(halves):  # letters: the added option can take a 26-option question past Z
                    for part, half in enumerate(halves, 1):
                        assert half is not None
                        self.items.append((i, part))
                        self.lengths.append(len(half.ids))
                        self.longest = max(self.longest, len(half.ids))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, n: int) -> Example:
        i, part = self.items[n]
        rng = random.Random(f"{self.seed}:{i}:{part and 'pair'}:{self.epoch}")  # both halves share one stream
        record = self.records[i]
        if part:
            pair = none_pair(record, rng)
            assert pair is not None
            record = pair[part - 1]
        elif self.augment == "kev":
            record = augment(record, rng)
        elif self.augment == "shuffle":
            record = shuffle_options(record, rng)
        example = encode(record, self.tokenizer, 1 << 30, self.style)  # the state's length was checked up front
        if example is None:  # letters: an added option took the question past 26; train it without the addition
            example = encode(self.records[i], self.tokenizer, 1 << 30, self.style)
        assert example is not None
        return example


def train(
    args: argparse.Namespace,
    model_dir: Path,
    data: Shuffled,
    dev: list[Example],
    calibration: list[Example],
    data_used: list[dict[str, object]],
) -> None:
    from .model import LORA_PRESETS, HeadConfig, LoraConfig, load_backbone, make_head

    lora = LoraConfig(
        rank=args.lora,
        alpha=args.lora_alpha or 2 * args.lora,
        targets=LORA_PRESETS[args.lora_targets],
        rslora=args.rslora,
    )
    config = HeadConfig(
        args.head_kind, args.head_dim, args.head_layers, args.head_heads, rep=args.option_rep, proj=args.head_proj
    )
    longest = max([data.longest, *(len(e.ids) for e in dev + calibration)])
    backbone = load_backbone(model_dir, lora, longest, args.seed, args.engine)  # unsloth before transformers

    import torch
    from safetensors.torch import load_file
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    from .calibrate import fit_temperature
    from .metrics import Answer, summarize
    from .model import SystemOne, adapted, collate, save_head, save_merged
    from .trainer import TRAINABLE, PointerTrainer, restore, save_trainable, trainable_state

    head = make_head(hidden_size(model_dir), config)
    if config.kind == "letters":  # Qwen's own answer-letter logits: the tied embedding rows of " A" ... " Z"
        vocab = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        letter_ids = [vocab.encode(f" {c}", add_special_tokens=False).ids[0] for c in LETTERS]
        head.set_letters(backbone.get_input_embeddings().weight[letter_ids])  # type: ignore[operator]
    model = SystemOne(backbone, head, args.ordinal_weight)
    model.to(next(backbone.parameters()).device)
    if args.init_from:
        from .model import warm_start

        previous = warm_start(backbone, model.head, args.init_from, args.model, lora)
        print(f"init   from {args.init_from} (its T {previous:.3f})", flush=True)
    cuda = torch.cuda.is_available()
    bf16 = importlib.import_module("unsloth").is_bfloat16_supported() if args.engine == "unsloth" else True
    pad = 0  # right padding under an attention mask: the id is never read
    per_step = args.batch * args.accum
    planned = args.max_steps or math.ceil(len(data) / per_step) * args.epochs
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
        "records_per_step": per_step,
        "planned_steps": planned,
    }
    print("setup  " + "  ".join(f"{k} {v}" for k, v in setup.items()), flush=True)

    def predict(rows: list[Example]) -> tuple[list[torch.Tensor], list[int], list[tuple[float, ...] | None]]:
        """Option scores for every question, in `rows` order."""
        model.eval()
        scores: list[torch.Tensor] = []
        labels: list[int] = []
        targets: list[tuple[float, ...] | None] = []
        dtype = torch.bfloat16 if bf16 else torch.float16
        device = next(model.parameters()).device  # where the Trainer put it, not where it was loaded
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, len(rows), args.batch):
                batch = collate(rows[i : i + args.batch], pad)
                per = model.scores(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["examples"])
                for example, s in zip(batch["examples"], per, strict=True):
                    scores += [q.float().cpu() for q in s]
                    labels += example.labels
                    targets += example.targets
        model.train()
        return scores, labels, targets

    def evaluate(rows: list[Example], temperature: float = 1.0) -> tuple[float, float]:
        """NLL and accuracy (`metrics.summarize`: hard-labelled questions only, unknowable ones have no answer)."""
        got = summarize(
            [
                Answer((s / temperature).softmax(-1).tolist(), y, target=t)
                for s, y, t in zip(*predict(rows), strict=True)
            ]
        )
        return got["nll"], got["accuracy"]

    every = args.eval_steps
    probe = random.Random(args.seed).sample(dev, min(args.eval_max, len(dev))) if every else []
    history: list[dict[str, float]] = []  # the validation curve, saved in run.json
    # The best checkpoint: lowest dev NLL at T = 1 on the fixed dev sample (all of dev at epoch ends without
    # --eval-steps). Written to best/ as it improves, so it survives --resume; the final weights still ship.
    out = Path(args.out)
    chosen = probe or dev
    best_dir = out / "best"
    best: dict[str, Any] = {}
    if args.resume and (best_dir / "selection.json").is_file():
        best = json.loads((best_dir / "selection.json").read_text(encoding="utf-8"))
    elif best_dir.exists():
        shutil.rmtree(best_dir)  # an earlier run's, in the same --out
    considered: list[int] = []

    def consider(step: int, nll: float, acc: float) -> None:
        considered.append(step)
        if not best or nll < best["nll"]:
            best.update(step=step, nll=round(nll, 4), accuracy=round(acc, 4))
            save_trainable(model, best_dir)
            (best_dir / "selection.json").write_text(json.dumps(best) + "\n", encoding="utf-8")

    class Dev(TrainerCallback):
        """Validation while training: a fixed dev sample every --eval-steps steps, all of dev at each epoch's end."""

        def on_step_end(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
        ) -> None:
            if probe and state.global_step % every == 0:
                nll, acc = evaluate(probe)
                history.append({"step": state.global_step, "nll": round(nll, 4), "accuracy": round(acc, 4)})
                consider(state.global_step, nll, acc)
                print(f"\nstep {state.global_step} dev({len(probe)})  nll {nll:.4f}  acc {acc:.4f}", flush=True)

        def on_epoch_end(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
        ) -> None:
            if dev:
                nll, acc = evaluate(dev)
                history.append({"step": state.global_step, "nll": round(nll, 4), "accuracy": round(acc, 4), "full": 1})
                if not probe:
                    consider(state.global_step, nll, acc)
                print(f"\nepoch dev  nll {nll:.4f}  acc {acc:.4f}", flush=True)

    class Epochs(TrainerCallback):
        """Tells the data which epoch is starting (a resumed one too: `state.epoch` is restored from the checkpoint)."""

        def on_epoch_begin(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
        ) -> None:
            data.set_epoch(int(state.epoch or 0))

    started, limit = time.time(), args.max_minutes
    stopped: list[int] = []  # the step at which --max-minutes ended training, if it did

    class Deadline(TrainerCallback):
        """Ends training cleanly once --max-minutes have passed; calibration, saving and merging still run."""

        def on_step_end(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
        ) -> None:
            if limit and time.time() - started > 60 * limit:
                stopped.append(state.global_step)
                print(f"\nstopping at step {state.global_step}: --max-minutes {limit} reached", flush=True)
                control.should_training_stop = True

    trainer = (
        None
        if not args.epochs
        else PointerTrainer(
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
                # checkpoints (adapter + head + optimizer/scheduler/RNG state) only with --save-steps; the run itself
                # is saved below
                save_strategy="steps" if args.save_steps else "no",
                save_steps=args.save_steps or 500,
                save_total_limit=2,
                report_to=args.report_to,  # the Trainer's own trackers; run.json is written either way
                run_name=Path(args.out).name,
                remove_unused_columns=False,  # the head needs every example's positions
                dataloader_pin_memory=cuda,
                tf32=cuda,  # the head and the softmaxes run in fp32: TF32 tensor cores on H100
                use_cpu=not cuda,
                seed=args.seed,
                include_num_input_tokens_seen=True,  # padded tokens through the model, for tokens_per_second
            ),
            train_dataset=data,
            data_collator=lambda rows: collate(rows, pad),
            callbacks=[Epochs(), Dev(), Deadline()],
        )
    )
    resume = None
    if args.resume and trainer:
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(args.out) if args.resume == "latest" else args.resume  # type: ignore[no-untyped-call]
        if resume is None:
            raise SystemExit(f"--resume latest: no checkpoint-* in {args.out}")
        print(f"resume from {resume}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume) if trainer else None  # --epochs 0: zero-shot
    minutes = (time.time() - started) / 60
    seen = int(trainer.state.num_input_tokens_seen) if trainer else 0
    final_step = trainer.state.global_step if trainer else 0
    if trainer and chosen and final_step not in considered:  # the final weights are a candidate too
        nll, acc = evaluate(chosen)
        history.append({"step": final_step, "nll": round(nll, 4), "accuracy": round(acc, 4), "final": 1})
        consider(final_step, nll, acc)

    def kinds(rows: list[Example]) -> list[str]:
        return [k for e in rows for k in (e.kinds or ("choice",) * len(e.labels))]

    def at(rows: list[Example], fitted: Any, temperature: float, temperatures: dict[str, float]) -> dict[str, Any]:  # noqa: ANN401
        """`metrics.summarize` of `rows` (scores from `predict`) at T, the type's own T where one was fitted."""
        sources = [src for e in rows for src in (e.sources or ("",) * len(e.labels))]
        got = summarize(
            [
                Answer((s / temperatures.get(k, temperature)).softmax(-1).tolist(), y, k, src, t)
                for s, y, t, k, src in zip(*fitted, kinds(rows), sources, strict=True)
            ]
        )
        keep = ("questions", "accuracy", "nll", "brier", "ece", "ece_by_type", "accuracy_by_type")
        return {m: got[m] for m in keep}

    def measure(label: str = "") -> tuple[float, dict[str, float], dict[str, Any]]:
        """T fitted on calibration (and per type with --calibrate-by-type) for the weights the model holds now, with
        the calibration split and dev before and after it."""
        temperature, temperatures = 1.0, {}
        report: dict[str, Any] = {}
        if calibration:
            fitted = predict(calibration)
            temperature = fit_temperature(*fitted)
            if args.calibrate_by_type:  # one T per question type with enough calibration questions; the rest share T
                for kind in sorted(set(kinds(calibration))):  # on the same rows as the shared T, soft targets too
                    pick = [i for i, k in enumerate(kinds(calibration)) if k == kind]
                    if len(pick) >= 50:
                        temperatures[kind] = fit_temperature(
                            [fitted[0][i] for i in pick], [fitted[1][i] for i in pick], [fitted[2][i] for i in pick]
                        )
            report["calibration_split"] = {
                "files": args.calibration,
                "before": at(calibration, fitted, 1.0, {}),
                "after": at(calibration, fitted, temperature, temperatures),
            }
            print(
                f"{label}calibration  T {temperature:.3f}" + "".join(f"  {k} {t:.3f}" for k, t in temperatures.items()),
                flush=True,
            )
        report |= {"temperature": temperature, "temperatures": temperatures}
        if dev:
            scored = predict(dev)
            report["before"], report["after"] = at(dev, scored, 1.0, {}), at(dev, scored, temperature, temperatures)
            before, calibrated = report["before"], report["after"]
            print(
                f"{label}dev at T  nll {calibrated['nll']:.4f}  acc {calibrated['accuracy']:.4f}  "
                f"brier {calibrated['brier']:.4f}  ece {calibrated['ece']:.4f}   "
                f"(T=1: nll {before['nll']:.4f}  ece {before['ece']:.4f})",
                flush=True,
            )
        return temperature, temperatures, report

    temperature, temperatures, report = measure()
    dev_at_t = {"nll": report["after"]["nll"], "accuracy": report["after"]["accuracy"]} if dev else {}

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
    selection: dict[str, Any] | None = None
    if best:
        selection = {**best, "on": "dev sample" if probe else "dev", "questions": sum(len(e.labels) for e in chosen),
                     "metric": "nll at T=1"}  # fmt: skip
        if best["step"] == final_step:  # the final weights are the best: nothing more to keep
            shutil.rmtree(best_dir, ignore_errors=True)
            selection["path"] = "."
        else:  # the best weights get their own T and dev report, then the final weights come back for the merge
            final = trainable_state(model)
            restore(model, load_file(best_dir / TRAINABLE), str(best_dir))
            best_t, best_ts, best_report = measure(f"best (step {best['step']}) ")
            if lora.rank:
                backbone.save_pretrained(best_dir)
            save_head(best_dir, model.head, **meta, temperature=best_t, temperatures=best_ts, step=best["step"])
            restore(model, final, "the final weights")
            for name in (TRAINABLE, "selection.json"):  # the adapter and head hold the weights; run.json the choice
                (best_dir / name).unlink()
            selection |= {"path": "best", "temperature": best_t, "calibration_report": best_report}
        print(f"best   step {best['step']} of {final_step}  dev nll {best['nll']:.4f}  acc {best['accuracy']:.4f}")
    run = {
        **setup,
        "base": args.model,
        "lora_rank": lora.rank,
        "data": args.data,
        "calibration": args.calibration,
        "train_minutes": round(minutes, 1),
        "tokens_seen": seen,
        "tokens_per_second": round(seen / max(60 * minutes, 1e-9)),
        "sources_cap": args.sources_cap,
        "init_from": str(args.init_from) if args.init_from else None,
        "init_from_spec": args.init_spec,  # release:<version> or hf:<repo>@<rev> when continued from a published one
        "replay": args.replay,
        "augment": args.augment,
        "p_none_pair": args.p_none_pair,
        "lr": args.lr,
        "head_kind": args.head_kind,
        "ordinal_weight": args.ordinal_weight,
        "epochs": args.epochs,
        "train_loss": result.training_loss if result else None,
        "steps": result.global_step if result else 0,
        "calibration_report": report,
        "base_files": base_files(args.model),
        "model": pinned(args.model),
        "dataset_sha256": dataset_hash(data_used),
        "runtime": runtime(),
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
        "hyperparameters": json.loads(json.dumps(vars(args), default=str)),
        "lora_targets": args.lora_targets,
        "option_rep": config.option_rep,
        "head_proj": args.head_proj,
        "overfit": args.overfit,
        "seed": args.seed,
        "stopped_by_max_minutes": bool(stopped),
        "temperature": temperature,
        "dev_at_temperature": dev_at_t,
        "dev_history": history,
        "train_history": train_history(trainer.state.log_history if trainer else []),
        "best": selection,
        "dev_files": args.dev,
        "data_used": data_used,
        "max_licence": args.max_licence,
        "licences": licence_counts(data_used),  # trained records per licence class
        "versions": versions(),
        "commit": commit(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}", flush=True)
    if lora.rank and args.merge:  # last: the adapter and head are already safe if this fails
        replaced = save_merged(backbone, model_dir, out / "merged")
        print(f"saved {out / 'merged'}  ({replaced} merged weights)")
        from .integrity import write

        if not write(out, model_dir)["ok"]:
            raise SystemExit(f"{out}: the merged run failed its integrity checks (integrity.json)")


def train_history(log: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Trainer's log (loss, grad_norm, learning_rate every 10 steps; the run's summary last), as plain JSON."""
    return [
        {k: v if isinstance(v, int | str) else round(float(v), 6) for k, v in row.items() if v is not None}
        for row in log
    ]


CONTINUE_FILES = ["adapter_model.safetensors", "adapter_config.json", "head.safetensors", "head.json", "run.json"]


def gather(args: argparse.Namespace, data_used: list[dict[str, object]]) -> list[Record]:
    """The run's training records, one per question: `--data`, a capped slice of every public source, and `--replay`
    from each `--replay-from` suite; each file and its lines go into `data_used`. A source text that is in any of the
    suites this run trains on or replays from is left out of the sources, the whole suite and not only the lines
    sampled now: a run continued from round 1 has already trained on all of core."""
    worst: Kind | None = args.max_licence
    if args.limit:  # a rehearsal (the local GPU): a seeded slice of every file, its lines recorded like replay's
        args.replay, args.sources_cap = min(args.replay, args.limit), min(args.sources_cap, args.limit)
        records = [r for d in args.data for r in sample(CLEAN / d, args.limit, args.seed, data_used, worst)]
    elif worst:  # every record the licence allows, its lines recorded (sampling all of them keeps every one)
        records = [r for d in args.data for r in sample(CLEAN / d, 1 << 62, args.seed, data_used, worst)]
    else:
        records = load([CLEAN / d for d in args.data])
        data_used += [{"path": d, "sha256": digest(CLEAN / d), "lines": "all"} for d in args.data]
    suites = args.replay_from if args.replay else []
    replayed = [r for f in suites for r in sample(CLEAN / f, args.replay, args.seed, data_used, worst)]
    if args.sources_cap:
        if not (CLEAN / "train" / "sources").is_dir():
            raise SystemExit(f"{CLEAN}/train/sources is missing: make data-download")
        known = load([CLEAN / f for f in [*args.data, *suites]])
        records += sources(args.sources_cap, args.seed, known, data_used, worst)
    return [one for r in [*records, *replayed] for one in split(r)]  # one row per question, as served


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="den train")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--data", nargs="*", default=[f"train/{s}.jsonl" for s in KEV_TRAIN], help="files under data/clean")
    p.add_argument(
        "--sources-cap",
        type=int,
        default=0,
        help="also train on up to this many records from each normalized source in data/clean/train/sources (0: none)",
    )
    p.add_argument("--dev", nargs="*", default=["dev/core.jsonl"], help="files under data/clean; never test")
    p.add_argument(
        "--calibration",
        nargs="*",
        default=["calibration/core.jsonl"],
        help="files under data/clean to fit T on; heldout shares unknowable families with train/dates-unknowable",
    )
    p.add_argument("--out", default="runs/qwen3.5-4b-lora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="peak learning rate (Kev-4B: 5e-5 from the base, 2e-5 after)")
    p.add_argument("--head-lr", type=float, default=0, help="the pointer head's peak learning rate; 0: same as --lr")
    p.add_argument("--warmup", type=float, default=0.1, help="share of steps spent warming up (Kev: 0.1)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora", type=int, default=16, help="LoRA rank; 0 trains the head alone")
    p.add_argument("--lora-alpha", type=int, default=0, help="LoRA alpha; 0: twice the rank (Kev-4B)")
    p.add_argument("--rslora", action="store_true", help="rsLoRA: scale by alpha / sqrt(rank), for higher ranks")
    p.add_argument(
        "--max-licence",
        choices=tuple(RANK),
        help="train only on records whose sources are at most this licence class (licences.py), e.g. share-alike "
        "for a model whose training data allows commercial use; default: everything",
    )
    p.add_argument(
        "--report-to",
        nargs="+",
        default=["none"],
        help="Trainer experiment trackers, e.g. trackio, wandb, tensorboard (each needs its package installed)",
    )
    p.add_argument(
        "--engine",
        choices=("unsloth", "peft"),
        default="unsloth",
        help="unsloth (CUDA) trains; peft runs the same adapters anywhere, for checks",
    )
    p.add_argument(
        "--head-kind",
        choices=("set", "pointer", "letters"),
        default="set",
        help="set: ours; pointer: Kev's q.k head; letters: Qwen's own answer-letter logits (text-generation baseline)",
    )
    p.add_argument("--option-rep", choices=("end", "marker", "mean", "attn"), help="option key (default: kind's own)")
    p.add_argument("--head-proj", choices=("linear", "mlp"), default="linear", help="question/option projections")
    p.add_argument(
        "--lora-targets", choices=("all", "attention-mlp", "attention"), default="all", help="LoRA module set"
    )
    p.add_argument("--calibrate-by-type", action="store_true", help="also fit one T per question type")
    p.add_argument(
        "--overfit", type=int, default=0, help="gate: train on N rows, score on the same N rows (no augmentation)"
    )
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument(
        "--ordinal-weight", type=float, default=0.0, help="ranked probability score weight on score questions (Kev: 0)"
    )
    p.add_argument("--head-layers", type=int, default=1, help="option-mixing blocks; 0 for no cross-option attention")
    p.add_argument("--head-heads", type=int, default=4)
    p.add_argument("--augment", choices=("kev", "shuffle", "none"), default="kev", help="choice-option augmentation")
    p.add_argument(
        "--p-none-pair", type=float, default=0.0, help="share of records that also train a none minimal pair"
    )
    p.add_argument(
        "--replay", type=int, default=0, help="also train on this many records sampled from each --replay-from file"
    )
    p.add_argument(
        "--replay-from", nargs="*", default=["train/core.jsonl"], help="files under data/clean to replay from"
    )
    p.add_argument(
        "--eval-steps", type=int, default=0, help="validate on a dev sample every this many steps (0: epoch ends only)"
    )
    p.add_argument("--eval-max", type=int, default=600, help="questions in that dev sample")
    p.add_argument(
        "--init-from",
        help="continue from a run (its adapter and head; same base, rank, head): a run directory, "
        "hf:<org>/<name>@<rev>, or release:<version> (den release)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="at most this many records from any one file (--data, --replay, --sources-cap; seeded): a small rehearsal",
    )
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (a timing run)")
    p.add_argument(
        "--max-minutes", type=float, default=0, help="end training after this many minutes, then calibrate and save"
    )
    p.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (adapter, head, optimizer)")
    p.add_argument("--resume", help="continue from a checkpoint directory, or `latest` in --out")
    p.add_argument("--merge", action="store_true", help="also save the LoRA merged into bf16 weights (runs/.../merged)")
    p.add_argument("--max-state", type=int, default=MAX_STATE_TOKENS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="tokenize and report; no model, no GPU")
    p.set_defaults(init_spec=None)  # --init-from as given (main sets it), recorded beside the resolved run directory
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("data", "dev", "calibration", "replay_from"):  # `..` resolved, so the guards below see the real role
        setattr(args, name, [record_path(d) for d in getattr(args, name)])
    if any(role(d) == "test" for d in [*args.data, *args.dev, *args.calibration]):
        raise SystemExit("test partitions are read once per release candidate, never while training")
    if any(role(d) != "calibration" for d in args.calibration):
        raise SystemExit("T is fitted on calibration files only, never on train or dev")
    if args.replay and (twice := set(args.replay_from) & set(args.data)):
        raise SystemExit(f"--replay samples {sorted(twice)}, which --data already trains on in full")
    if any(role(f) != "train" for f in args.replay_from):
        raise SystemExit("--replay-from takes train/ files only")
    args.init_spec = args.init_from  # as given: a path, hf:<repo>@<rev> or release:<version>; recorded in run.json
    if args.init_from and not args.dry_run:
        from .release import locate

        args.init_from = locate(args.init_from, CONTINUE_FILES)  # a published run: once, into the Hub cache
        if not (args.init_from / "head.json").is_file():
            raise SystemExit(f"--init-from {args.init_spec}: no head.json there")

    model_dir = model_dir_of(args.model)
    if not (model_dir / "config.json").is_file() or not (model_dir / "tokenizer.json").is_file():
        raise SystemExit(f"{model_dir} is not downloaded: make model MODEL={args.model}")
    every = [*args.data, *args.dev, *args.calibration, *(args.replay_from if args.replay else [])]
    if missing := [d for d in every if not (CLEAN / d).is_file()]:
        raise SystemExit(f"missing under {CLEAN}: {missing} (make data-download)")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    data_used: list[dict[str, object]] = []
    records = gather(args, data_used)
    style: Style = "letters" if args.head_kind == "letters" else "dash"
    if args.overfit:  # a fixed slice, unaugmented, scored on itself: a model that can't fit it can't learn
        records = random.Random(args.seed).sample(records, min(args.overfit, len(records)))
        args.augment, args.p_none_pair = "none", 0.0
    data = Shuffled(records, tokenizer, args.max_state, args.seed, args.augment, args.p_none_pair, style)
    dev_records = data.records if args.overfit else load([CLEAN / d for d in args.dev])
    # dev and calibration are scored as `den evaluate` and `den serve` score them, with no cap on the state:
    # --max-state is a training limit, and T and the dev numbers should describe what ships
    dev, dev_skipped = examples(dev_records, tokenizer, 1 << 30, style=style)
    calibration, cal_skipped = examples(load([CLEAN / d for d in args.calibration]), tokenizer, 1 << 30, style=style)
    questions = sum(len(r.questions) for r in data.records)
    print(describe("train", data.lengths, questions, len(records) - len(data.records)))
    for name, rows, skipped in (("dev", dev, dev_skipped), ("calib", calibration, cal_skipped)):
        print(describe(name, [len(e.ids) for e in rows], sum(len(e.labels) for e in rows), skipped))
    if not args.dry_run:
        train(args, model_dir, data, dev, calibration, data_used)
    return 0  # `den train` then exits without interpreter teardown (cli.DELEGATED)
