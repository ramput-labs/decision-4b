"""Publish a trained run to the Hugging Face Hub as a private model that `den predict --run hf:<repo>` can use.

The model card is written from the run's own records (`run.json` from training, `eval.json` from evaluation), so the
numbers on the Hub are the ones measured. The upload is resumable: if it is interrupted, run the same command again.
Afterwards the repo listing is checked for every file `predict` needs.

    den publish --run runs/qwen3.5-4b-lora --repo <org>/<name>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .pins import MODELS

NEEDED = ("head.safetensors", "head.json", "run.json", "merged/config.json", "merged/tokenizer.json")
SKIP = ("checkpoint-*", "*.tmp", ".cache/*", "runs/*")  # Trainer leftovers and caches


def _json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return loaded


def stages(run: Path) -> list[tuple[Path, dict[str, Any]]]:
    """The run and every run it continued from (`--init-from`), first stage first, while their run.json is here."""
    chain: list[tuple[Path, dict[str, Any]]] = []
    at: Path | None = run
    while at is not None and (at / "run.json").is_file() and len(chain) < 16:
        record = _json(at / "run.json")
        chain.append((at, record))
        at = Path(record["init_from"]) if record.get("init_from") else None
    return chain[::-1]


def card(run: Path, repo: str) -> str:
    """The model card: what the model is, its measured numbers, and how to use it."""
    trained, evaluated = _json(run / "run.json"), _json(run / "eval.json")
    history = "\n".join(
        f"| `{path.name}` | {', '.join(r.get('data', []))}"
        f"{f' + {r["replay"]} replayed' if r.get('replay') else ''} | {r.get('epochs', '?')} | {r.get('lr', '?')} | "
        f"{r.get('steps', '?')} | {r.get('train_minutes', '?')} | {r.get('temperature', float('nan')):.3f} |"
        for path, r in stages(run)
    )
    base = MODELS[trained["base"]].source.repo if trained.get("base") in MODELS else str(trained.get("base", "?"))
    nan = float("nan")
    rows = "\n".join(
        f"| `{name}` | {m['questions']:.0f} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m.get('brier', nan):.4f} "
        f"| {m['ece']:.4f} | {m.get('coverage_at_5%_error', nan):.3f} |"
        for name, m in sorted(evaluated.items())
    )
    versions = ", ".join(f"{k} {v}" for k, v in trained.get("versions", {}).items())
    return f"""---
base_model: {base}
library_name: peft
tags: [den, pointer-head, lora, unsloth, calibrated, multiple-choice]
---

# {repo.rsplit("/", 1)[-1]}

A System One decision model. It reads a state and typed questions (`choice`, `score`, `noul`) and returns a calibrated
probability for every option. It runs in one forward pass and generates no text. Built from `{base}` with a bf16 LoRA
(rank {trained.get("lora_rank", "?")}, {trained.get("engine", "?")}) and a set-aware pointer head; probabilities
are divided by a temperature T = {trained.get("temperature", float("nan")):.3f} fitted on the calibration split.

## Use

```bash
git clone https://github.com/ramput-labs/den && cd den && make setup
echo '{{"state": "I was charged twice for my order.", "questions": {{"team": {{"type": "choice",
  "instructions": "Which team handles this?", "criteria": {{"billing": "payments", "tech": "bugs"}}}}}}}}' \\
  | uv run den predict --run hf:{repo}
```

It runs on Apple Silicon (MLX), NVIDIA (CUDA) and CPU. The first call downloads the run into the Hub cache.

## Results

At T, over hard-labelled questions: accuracy, NLL, Brier score, expected calibration error, and coverage at 5% error
(the share of questions answerable at <= 5% error, most confident first). `eval.json` also has accuracy per question
type and, for score questions, the expected level's mean absolute error. `probes` is mostly unknowable items, scored
on confidence, not accuracy.

| file | questions | accuracy | nll | brier | ece | coverage at 5% error |
|---|---|---|---|---|---|---|
{rows or "| (not evaluated yet) | | | | | | |"}

## Files

- `merged/`: the LoRA folded into bf16 weights; loads like the base checkpoint.
- `head.safetensors`, `head.json`: the pointer head's weights, and its config (kind, size, the backbone's hidden size)
  and T. No pickle: the whole repo is safetensors and JSON.
- `adapter_model.safetensors`, `adapter_config.json`: the LoRA alone, for use on top of the base.
- `run.json`, `training_config.json`, `eval.json`: how it was trained and measured.
- `stages/`: every earlier training stage (adapter, head, its run.json), to continue or compare from.
- `data/`: the exact training data of every stage (sampled files hold only the lines used), the dev, calibration and
  test files it was scored on, and `data/MANIFEST.json` with their sha256.
- `logs/`: the training logs.

## Training

Each stage continued from the one before it (Kev-4B's recipe, with Kev's none-of-the-above augmentation), on
{trained.get("device", "?")}. The last stage's temperature ships.

| stage | data | epochs | lr | steps | minutes | T |
|---|---|---|---|---|---|---|
{history or "| (no run.json) | | | | | | |"}

Code {trained.get("commit", "?")}; {versions}.
"""


STAGE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "head.safetensors",
    "head.json",
    "run.json",
    "training_config.json",
    "eval.json",
)
CLEAN = Path("data/clean")


def bundle(run: Path, logs: Sequence[Path] = ()) -> dict[str, Any]:
    """Gather everything worth keeping into the run directory, beside the model, so one upload carries it:

    - `stages/<name>/`: every earlier stage of the chain (adapter, head, run.json; no merged weights)
    - `data/`: the exact training data of every stage (whole files, or only the sampled lines), the dev and calibration
      files, and every file in eval.json (test included: it was read for this run), with `data/MANIFEST.json`
    - `logs/`: the given log files

    Returns the manifest."""
    import shutil

    from .fetch import digest

    chain = stages(run)
    for path, _ in chain[:-1]:
        target = run / "stages" / path.name
        target.mkdir(parents=True, exist_ok=True)
        for name in STAGE_FILES:
            if (path / name).is_file():
                shutil.copy2(path / name, target / name)
    wanted: dict[str, list[int] | str] = {}
    for _, record in chain:
        for used in record.get("data_used", []):
            lines = used["lines"]
            if wanted.get(used["path"]) != "all":
                wanted[used["path"]] = "all" if lines == "all" else sorted({*wanted.get(used["path"], []), *lines})
        for f in [*record.get("dev_files", []), *record.get("calibration", [])]:
            wanted[f] = "all"
    for f in _json(run / "eval.json"):
        wanted[f] = "all"
    manifest: dict[str, Any] = {}
    for rel, lines in sorted(wanted.items()):
        source, target = CLEAN / rel, run / "data" / rel
        if not source.is_file():
            manifest[rel] = {"missing": True}
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if lines == "all":
            shutil.copy2(source, target)
        else:
            keep = set(lines)
            with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
                dst.writelines(line for n, line in enumerate(src, 1) if n in keep)
        manifest[rel] = {"source_sha256": digest(source), "lines": lines if lines == "all" else len(lines),
                         "sha256": digest(target)}  # fmt: skip
    (run / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for log in logs:
        (run / "logs").mkdir(exist_ok=True)
        shutil.copy2(log, run / "logs" / log.name)
    return manifest


def publish(run: Path, repo: str, private: bool = True, logs: Sequence[Path] = ()) -> list[str]:
    from huggingface_hub import create_repo, list_repo_files, upload_large_folder

    if missing := [f for f in NEEDED if not (run / f).is_file()]:
        raise SystemExit(f"{run} is missing {missing}: train with --merge first")
    bundle(run, logs)
    (run / "README.md").write_text(card(run, repo), encoding="utf-8")
    create_repo(repo, private=private, repo_type="model", exist_ok=True)
    upload_large_folder(repo, run, repo_type="model", private=private, ignore_patterns=list(SKIP))
    files = list_repo_files(repo)
    if absent := [f for f in (*NEEDED, "README.md", "data/MANIFEST.json") if f not in files]:
        raise SystemExit(f"uploaded, but {repo} lacks {absent}: run the same command again")
    return files


def publish_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den publish")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--repo", required=True, help="<org>/<name> on the Hugging Face Hub")
    p.add_argument("--logs", nargs="*", type=Path, default=[], help="log files to upload under logs/")
    p.add_argument("--public", action="store_true", help="publish publicly; private by default")
    p.add_argument(
        "--local", action="store_true", help="write the card, stages/, data/ and logs/ into the run; no upload"
    )
    args = p.parse_args(argv)
    if args.local:
        manifest = bundle(args.run, args.logs)
        (args.run / "README.md").write_text(card(args.run, args.repo), encoding="utf-8")
        print(f"bundled {args.run}: {len(manifest)} data files, {len(stages(args.run)) - 1} earlier stages")
        return 0
    files = publish(args.run, args.repo, private=not args.public, logs=args.logs)
    print(f"https://huggingface.co/{args.repo}  {len(files)} files")
    return 0
