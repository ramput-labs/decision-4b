"""`den publish`: a trained run as a private Hub model that `--run hf:<repo>` can use. The card is written from the
run's own records, so the numbers on the Hub are the ones measured. The upload is resumable: rerun it if interrupted.

    den publish --run runs/round2 --repo <org>/<name>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .licences import RANK, Kind, allowed, classify
from .paths import CLEAN, read_json
from .pins import MODELS

NEEDED = ("head.safetensors", "head.json", "run.json", "merged/config.json", "merged/tokenizer.json")
SKIP = ("checkpoint-*", "*.tmp", ".cache/*", "runs/*")  # Trainer leftovers and caches


def stages(run: Path) -> list[tuple[Path, dict[str, Any]]]:
    """The run and every run it continued from (`--init-from`), first stage first, while their run.json is here."""
    chain: list[tuple[Path, dict[str, Any]]] = []
    at: Path | None = run
    while at is not None and (at / "run.json").is_file() and len(chain) < 16:
        record = read_json(at / "run.json")
        chain.append((at, record))
        at = Path(record["init_from"]) if record.get("init_from") else None
    return chain[::-1]


def card(run: Path, repo: str) -> str:
    """The model card: what the model is, its measured numbers, and how to use it."""
    from huggingface_hub import ModelCardData

    trained, evaluated = read_json(run / "run.json"), read_json(run / "eval.json")
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
    sections = (calibrated(trained), behavior(evaluated), compared(run), checked(run), trained_on(run))
    details = "\n".join(p for p in sections if p)
    metadata = ModelCardData(
        license=base_licence(run),
        base_model=base,
        base_model_relation="finetune",  # merged/ is the base with the LoRA folded in
        library_name="peft",
        tags=["den", "pointer-head", "lora", "unsloth", "calibrated", "multiple-choice"],
    )
    return f"""---
{metadata.to_yaml()}
---

# {repo.rsplit("/", 1)[-1]}

A System One decision model. It reads a state and typed questions (`choice`, `score`, `noul`) and returns a calibrated
probability for every option. It runs in one forward pass and generates no text. Built from `{base}` with a bf16 LoRA
(rank {trained.get("lora", {}).get("rank", "?")}, {trained.get("engine", "?")}) and a set-aware pointer head;
probabilities are divided by a temperature T = {trained.get("temperature", float("nan")):.3f} fitted on the
calibration split.

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

{details}

## Files

- `merged/`: the LoRA folded into bf16 weights; loads like the base checkpoint.
- `head.safetensors`, `head.json`: the pointer head's weights, and its config (kind, size, the backbone's hidden size)
  and T. No pickle: the whole repo is safetensors and JSON.
- `adapter_model.safetensors`, `adapter_config.json`: the LoRA alone, for use on top of the base.
- `run.json`, `training_config.json`, `eval.json`: how it was trained (loss log, dev curve, calibration) and measured.
- `best/` (when present): the checkpoint with the lowest dev NLL, adapter and head with its own T; the final
  weights above are the ones that ship.
- `integrity.json`: the merged run's checks and every model file's sha256; `baselines.json`: the mode comparison.
- `stages/`: every earlier training stage (adapter, head, its run.json), to continue or compare from.
- `data/`: the exact training data of every stage (sampled files hold only the lines used), the dev, calibration and
  test files it was scored on, and `data/MANIFEST.json` with their sha256. Files whose licences forbid
  redistribution are left out; the manifest lists them by hash and line numbers, so a `make data-download` copy
  rebuilds them exactly.
- `logs/`: the training logs.

## Training

Each stage continued from the one before it (Kev-4B's recipe, with Kev's none-of-the-above augmentation), on
{trained.get("device", "?")}. The last stage's temperature ships.

| stage | data | epochs | lr | steps | minutes | T |
|---|---|---|---|---|---|---|
{history or "| (no run.json) | | | | | | |"}

Code {trained.get("commit", "?")}; {versions}.
"""


def base_licence(run: Path) -> str | None:
    """The base model's licence, from the card `merged/` copies from it: the weights keep their base's licence."""
    from huggingface_hub import ModelCard

    readme = run / "merged" / "README.md"
    found = ModelCard.load(readme).data.license if readme.is_file() else None
    return str(found) if found else None


def trained_on(run: Path) -> str:
    """The training records' licence classes over every stage (`run.json` `licences`), and what that means for use."""
    counts: dict[str, int] = {}
    for _, record in stages(run):
        for k, n in (record.get("licences") or {}).items():
            counts[k] = counts.get(k, 0) + n
    if not counts:
        return ""
    rows = "\n".join(f"| {k} | {counts[k]} |" for k in RANK if k in counts)
    limit = read_json(run / "run.json").get("max_licence")
    beyond = [k for k in counts if RANK[k] > RANK["share-alike"]]  # type: ignore[index]
    note = (
        f"Some training records come from sources whose terms limit their use ({', '.join(beyond)}; see "
        "`data/LICENSES.md` in the data repo). The weights carry the base model's licence; whether those data terms "
        "reach a model trained on them is a legal question this card does not settle. `den train --max-licence "
        "share-alike` trains only on data whose terms allow commercial use."
        if beyond
        else "Every training record comes from a source whose terms allow commercial use (open or share-alike)."
    )
    cap = f" Trained with `--max-licence {limit}`." if limit else ""
    return f"## Training data licences\n\n| class | records |\n|---|---|\n{rows}\n\n{note}{cap}\n"


def calibrated(trained: dict[str, Any]) -> str:
    """Calibration: what T did, on the calibration split it was fitted on and on dev."""
    report = trained.get("calibration_report") or {}
    rows: list[str] = []
    for split, label in ((report.get("calibration_split") or {}, "calibration (fit)"), (report, "dev")):
        for when in ("before", "after"):
            if (r := split.get(when)) is not None:
                t = "T = 1" if when == "before" else f"T = {report.get('temperature', float('nan')):.3f}"
                rows.append(f"| {label} | {t} | {r['accuracy']:.4f} | {r['nll']:.4f} | {r['ece']:.4f} |")
    best = trained.get("best")
    note = ""
    if best:
        where = "the final weights" if best.get("path") == "." else "`best/`"
        note = (
            f"\n\nLowest dev NLL (T = 1, {best['questions']} {best['on']} questions): step {best['step']} of "
            f"{trained.get('steps', '?')}, nll {best['nll']:.4f}, accuracy {best['accuracy']:.4f}: {where}."
        )
    if not rows:
        return note.strip()
    table = "| split | at | accuracy | nll | ece |\n|---|---|---|---|---|\n" + "\n".join(rows)
    return f"## Calibration\n\n{table}{note}\n"


def behavior(evaluated: dict[str, Any]) -> str:
    """Kev's minimal pairs: both halves right, the true option present (a none option wrong) and removed."""
    rows = [
        f"| `{name}` | {m['pairs']} | {m['pair_accuracy']:.4f} | {m['accuracy_present']:.4f} "
        f"| {m['accuracy_absent']:.4f} |"
        for name, m in sorted(evaluated.items())
        if "pair_accuracy" in m
    ]
    head = "| file | pairs | pair accuracy | true option present | true option removed |\n|---|---|---|---|---|\n"
    return f"## Minimal pairs\n\n{head}" + "\n".join(rows) + "\n" if rows else ""


def compared(run: Path) -> str:
    """The baselines table from `den baselines`."""
    from .compare import markdown

    table = read_json(run / "baselines.json")
    if not table:
        return ""
    return (
        "## Baselines\n\nA zero-shot Qwen (answer-letter logits), B LoRA + letters, C frozen Qwen + pointer head, D "
        f"LoRA + pointer head; the same files, the same protocol.\n\n{markdown(table)}\n"
    )


def checked(run: Path) -> str:
    report = read_json(run / "integrity.json")
    if not report:
        return ""
    passed = sum(c["ok"] for c in report["checks"])
    return (
        f"## Integrity\n\n{passed} of {len(report['checks'])} checks passed (`integrity.json`); "
        f"{len(report['files'])} model files with their sha256.\n"
    )


STAGE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "head.safetensors",
    "head.json",
    "run.json",
    "training_config.json",
    "eval.json",
    "integrity.json",
)


def licence(rel: str) -> tuple[Kind, list[str]]:
    """A data/clean file's licence class and sources, read as `den licences` and the data mirror read them."""
    return classify(CLEAN.parent, [f"clean/{rel}"])[f"clean/{rel}"]


def bundle(run: Path, logs: Sequence[Path] = (), public: bool = False) -> dict[str, Any]:
    """Gather everything worth keeping into the run directory, beside the model, so one upload carries it:

    - `stages/<name>/`: every earlier stage of the chain (adapter, head, run.json; no merged weights)
    - `data/`: the exact training data of every stage (whole files, or only the sampled lines), the dev and calibration
      files, and every file in eval.json (test included: it was read for this run), with `data/MANIFEST.json`;
      a file the licences keep off this copy (`licences.allowed`) is listed there by hash and lines, never copied
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
    for f in read_json(run / "eval.json"):
        wanted[f] = "all"
    manifest: dict[str, Any] = {}
    for rel, lines in sorted(wanted.items()):
        source, target = CLEAN / rel, run / "data" / rel
        if not source.is_file():
            manifest[rel] = {"missing": True}
            continue
        k, keys = licence(rel)
        if not allowed(k, public):
            target.unlink(missing_ok=True)  # an earlier bundle may have copied it
            manifest[rel] = {"excluded": f"{k}: {', '.join(keys)}", "source_sha256": digest(source), "lines": lines}
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


def verify(run: Path) -> None:
    """The integrity checks, rerun on what is about to be uploaded; refuses a run that fails them."""
    from .integrity import base_of, write

    if not write(run, base_of(run))["ok"]:
        raise SystemExit(f"{run} failed its integrity checks (integrity.json): not uploading it")


def upload_folder(repo: str, folder: Path, repo_type: str, private: bool, ignore: Sequence[str]) -> None:
    """`upload_large_folder`, resumable per repo. Its resume cache (`<folder>/.cache/huggingface/upload`) marks files
    as committed without recording the repo, so a cache left by an upload to another repo would skip every file that
    hasn't changed since. A marker beside the cache names the repo it belongs to; any other repo starts clean."""
    import shutil

    from huggingface_hub import upload_large_folder

    cache = folder / ".cache" / "huggingface"
    marker, target = cache / "upload-repo", f"{repo_type}:{repo}"
    if (cache / "upload").exists() and (not marker.is_file() or marker.read_text(encoding="utf-8") != target):
        shutil.rmtree(cache / "upload")
    cache.mkdir(parents=True, exist_ok=True)
    marker.write_text(target, encoding="utf-8")
    upload_large_folder(repo, folder, repo_type=repo_type, private=private, ignore_patterns=list(ignore))


def publish(run: Path, repo: str, private: bool = True, logs: Sequence[Path] = ()) -> list[str]:
    from huggingface_hub import HfApi, create_repo, list_repo_files

    if missing := [f for f in NEEDED if not (run / f).is_file()]:
        raise SystemExit(f"{run} is missing {missing}: train with --merge first")
    verify(run)
    api = HfApi()
    public = not private or (api.repo_exists(repo) and not api.model_info(repo).private)  # public rules if it is
    manifest = bundle(run, logs, public=public)
    excluded = sorted(f"data/{rel}" for rel, entry in manifest.items() if "excluded" in entry)
    (run / "README.md").write_text(card(run, repo), encoding="utf-8")
    create_repo(repo, private=not public, repo_type="model", exist_ok=True)
    if stale := sorted(set(list_repo_files(repo)) & set(excluded)):  # an earlier upload carried them
        api.delete_files(repo, delete_patterns=stale, commit_message="den: remove data the licences exclude")
    upload_folder(repo, run, "model", private=not public, ignore=[*SKIP, *excluded])
    files = list_repo_files(repo)
    if leaked := sorted(set(files) & set(excluded)):
        raise SystemExit(f"{repo} holds data the licences exclude: {leaked}")
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
        verify(args.run)
        manifest = bundle(args.run, args.logs, public=args.public)
        (args.run / "README.md").write_text(card(args.run, args.repo), encoding="utf-8")
        print(f"bundled {args.run}: {len(manifest)} data files, {len(stages(args.run)) - 1} earlier stages")
        return 0
    files = publish(args.run, args.repo, private=not args.public, logs=args.logs)
    print(f"https://huggingface.co/{args.repo}  {len(files)} files")
    return 0
