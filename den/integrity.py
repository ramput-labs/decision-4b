"""`den check-run`: is a finished run intact? Checked from its files after the merge and before every upload: the head
and adapter load and are finite, and `merged/` has the base's exact layout with exactly the adapted weights changed.
`integrity.json` also lists every model file's sha256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .fetch import digest
from .paths import downloaded, model_dir, read_json

REPORT = "integrity.json"
MODEL_FILES = ("adapter_model.safetensors", "adapter_config.json", "head.safetensors", "head.json", "run.json",
               "training_config.json")  # fmt: skip
MODEL_DIRS = ("merged", "best")


def model_files(run: Path) -> dict[str, str]:
    """sha256 of every file that makes up the model (not the card, evaluations or bundled data, which change)."""
    found = [run / name for name in MODEL_FILES if (run / name).is_file()]
    found += [
        f
        for d in MODEL_DIRS
        if (run / d).is_dir()
        for f in sorted((run / d).rglob("*"))
        if f.is_file() and not f.name.startswith(".")
    ]
    return {f.relative_to(run).as_posix(): digest(f) for f in found}


def _finite(path: Path) -> tuple[int, list[str]]:
    """The tensor count of a safetensors file, and the names of any with a NaN or infinity."""
    from safetensors import safe_open

    bad = []
    with safe_open(path, "pt") as handle:
        names = list(handle.keys())
        for name in names:
            t = handle.get_tensor(name)
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                bad.append(name)
    return len(names), bad


def _merged(merged: Path, base: Path, modules: int) -> list[tuple[str, bool, str]]:
    from safetensors import safe_open

    checks: list[tuple[str, bool, str]] = []
    base_files = {f.name for f in base.iterdir() if f.is_file() and not f.name.startswith(".")}
    ours = {f.name for f in merged.iterdir() if f.is_file() and not f.name.startswith(".")}
    checks.append(("merged has the base's files", base_files == ours, f"missing {sorted(base_files - ours)[:3]}, "
                   f"extra {sorted(ours - base_files)[:3]}"))  # fmt: skip
    copied = sorted(n for n in base_files & ours if not n.endswith(".safetensors"))
    differ = [n for n in copied if digest(base / n) != digest(merged / n)]
    checks.append(("merged config and tokenizer are the base's", not differ, f"differ: {differ[:3]}"))
    changed: list[str] = []
    layout: list[str] = []
    broken: list[str] = []
    for shard in sorted(n for n in base_files & ours if n.endswith(".safetensors")):
        with safe_open(base / shard, "pt") as b, safe_open(merged / shard, "pt") as m:
            if set(b.keys()) != set(m.keys()):
                layout.append(shard)
                continue
            for key in b.keys():  # noqa: SIM118 (a safetensors handle is not a dict)
                old, new = b.get_tensor(key), m.get_tensor(key)
                if old.shape != new.shape or old.dtype != new.dtype:
                    layout.append(key)
                elif not torch.equal(old, new):
                    changed.append(key)
                    if not bool(torch.isfinite(new).all()):
                        broken.append(key)
    checks.append(("merged tensors keep the base's names, shapes and dtypes", not layout, f"differ: {layout[:3]}"))
    checks.append((f"exactly the {modules} adapted weights changed", len(changed) == modules, f"{len(changed)}"))
    checks.append(("changed weights are finite", not broken, f"non-finite: {broken[:3]}"))
    vision = [k for k in changed if ".visual." in k or not k.endswith(".weight")]
    checks.append(("no vision-tower or non-weight tensor changed", not vision, f"{vision[:3]}"))
    return checks


def check(run: Path, base: Path | None = None) -> dict[str, Any]:
    """Every integrity check on `run`, and its model files' sha256. `base` is the base checkpoint the merge was built
    from; without it, `merged/` is checked on its own (finite, right hidden size), not against the base."""
    from .device import hidden_size

    trained, head = read_json(run / "run.json"), read_json(run / "head.json")
    rank, modules = int(trained["lora"]["rank"]), int(trained["lora"]["modules"])
    checks: list[tuple[str, bool, str]] = [("head.json is a den head", head.get("format") == "den-pointer-head", "")]
    count, bad = _finite(run / "head.safetensors")
    checks.append(("head weights are finite", not bad, f"{count} tensors, non-finite: {bad[:3]}"))
    if rank:
        count, bad = _finite(run / "adapter_model.safetensors")
        a = sum(".lora_A." in k for k in _names(run / "adapter_model.safetensors"))
        checks.append(("adapter weights are finite", not bad, f"{count} tensors, non-finite: {bad[:3]}"))
        checks.append((f"adapter covers the {modules} adapted modules", a == modules, f"{a} lora_A tensors"))
    merged = run / "merged"
    if (merged / "config.json").is_file():
        size = hidden_size(merged)
        checks.append(("head fits the merged hidden size", head.get("hidden_size") == size, f"backbone {size}"))
        if base is not None:
            checks += _merged(merged, base, modules)
        else:
            bad = [n for shard in sorted(merged.glob("*.safetensors")) for n in _finite(shard)[1]]
            checks.append(("merged weights are finite", not bad, f"non-finite: {bad[:3]}"))
    elif rank:
        checks.append(("merged/ exists (train with --merge)", False, "a LoRA run serves from merged/"))
    if (run / "best" / "head.json").is_file():
        count, bad = _finite(run / "best" / "head.safetensors")
        checks.append(("best/ head weights are finite", not bad, f"non-finite: {bad[:3]}"))
    return {
        "ok": all(ok for _, ok, _ in checks),
        "checks": [{"check": name, "ok": ok, **({"detail": d} if not ok and d else {})} for name, ok, d in checks],
        "base": str(base) if base else None,
        "files": model_files(run),
    }


def _names(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, "pt") as handle:
        return list(handle.keys())


def write(run: Path, base: Path | None = None) -> dict[str, Any]:
    """`check`, saved as the run's integrity.json, with one line per check printed."""
    report = check(run, base)
    (run / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for c in report["checks"]:
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['check']}" + (f"  ({c['detail']})" if "detail" in c else ""))
    print(f"integrity {'ok' if report['ok'] else 'FAILED'}: {run / REPORT}, {len(report['files'])} files hashed")
    return report


def base_of(run: Path) -> Path | None:
    """The base checkpoint a run was merged from, when it is here (`models/<key>`)."""
    key = str(read_json(run / "run.json").get("base"))
    return model_dir(key) if downloaded(key) else None


def check_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den check-run")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--base", type=Path, help="the base checkpoint (default: models/<run.json base>, when downloaded)")
    p.add_argument("--load", action="store_true", help="also load the run and answer one request")
    args = p.parse_args(argv)
    report = write(args.run, args.base or base_of(args.run))
    if args.load and report["ok"]:
        from .api import Json
        from .runtime import Model

        team: Json = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "pay", "tech": "bugs"}}
        request: dict[str, Json] = {"state": "I was charged twice for my order.", "questions": {"team": team}}
        answer = Model(args.run).predict(request)["answers"]["team"]
        total = sum(answer["probabilities"].values())
        print(f"  {'ok ' if abs(total - 1) < 1e-3 else 'BAD'} loads and answers: {answer}")
        return 0 if abs(total - 1) < 1e-3 else 1
    return 0 if report["ok"] else 1
