"""systemone: fetch, verify and check everything a systemone model trains on.

systemone models                  list the backbones (any size) and reference checkpoints
systemone model [MODEL ...]       download models: a key from `systemone models`, or hf:<org>/<name>@<commit>
systemone list [SET ...]          show the dataset catalog
systemone data [SET ...]          download dataset sets (default: suites; all = every set except raw-bulk)
systemone verify                  re-hash every downloaded file against locks/
systemone normalize               raw sources -> canonical train / dev / test records under data/*/sources/
systemone audit [--model M]       check every record and source is fit to train and evaluate on
systemone env                     show the backend this machine uses
systemone smoke [--model M] [--backend B]
                                  run a downloaded model once on this machine

MODEL defaults to $SYSTEM_ONE_MODEL, else qwen3.5-4b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

from .catalog import SET_DESCRIPTIONS, SET_NAMES, Model, SetName, hub_model
from .fetch import (
    IntegrityError,
    Lock,
    entry,
    fetch_item,
    fetch_model,
    locked_digests,
    read_lock,
    verify_lock,
    write_lock,
)
from .pins import DEFAULT_MODEL, ITEMS, MODELS

DATA = Path("data")
MODELS_DIR = Path("models")
LOCKS = Path("locks")
REPORTS = Path("reports")


def _default_model() -> str:
    return os.environ.get("SYSTEM_ONE_MODEL", DEFAULT_MODEL)


def _model(spec: str) -> Model:
    if spec.startswith("hf:"):
        return hub_model(spec)
    if spec not in MODELS:
        raise SystemExit(f"unknown model {spec!r}; see `systemone models`, or pass hf:<org>/<name>@<commit>")
    return MODELS[spec]


def _downloaded(key: str) -> bool:
    return (MODELS_DIR / key / "config.json").is_file()


def _model_dir(key: str) -> Path:
    if not _downloaded(key):
        raise SystemExit(f"model {key!r} is not downloaded: systemone model {key}")
    return MODELS_DIR / key


def _write_report(name: str, text: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(text, encoding="utf-8")


def _sets(names: list[str]) -> tuple[SetName, ...]:
    if not names:
        return ("suites",)
    if names == ["all"]:
        return tuple(s for s in SET_NAMES if s != "raw-bulk")
    if unknown := sorted(set(names) - set(SET_NAMES)):
        raise SystemExit(f"unknown set(s) {unknown}; choose from {list(SET_NAMES)} or 'all'")
    return tuple(cast(SetName, n) for n in names)


def _size(n: int) -> str:
    return f"{n / 1e9:,.1f} GB" if n >= 1e9 else f"{n / 1e6:,.1f} MB"


def cmd_models() -> int:
    default = _default_model()
    for role in ("base", "reference"):
        print(f"\n[{role}s]")
        for m in (m for m in MODELS.values() if m.role == role):
            here = "downloaded" if _downloaded(m.key) else ""
            mark = "*" if m.key == default else " "
            print(f" {mark}{m.key:<18}{_size(m.bytes):>10}  {m.note:<52}{here}")
    print(f"\n* default (SYSTEM_ONE_MODEL={default})")
    return 0


def cmd_model(specs: list[str]) -> int:
    lock_path = LOCKS / "models.json"
    lock: Lock = read_lock(lock_path) or {"root": MODELS_DIR.as_posix(), "entries": {}}
    for model in map(_model, specs or [_default_model()]):
        placed = fetch_model(model, MODELS_DIR, locked_digests(lock))
        lock["entries"][model.key] = entry(f"model/{model.role}", model.source.ref, placed)
        write_lock(lock_path, lock)
        print(f"  ok  {model.key:<20} {model.source.ref}  {_size(sum(p.bytes for p in placed))}")
    return 0


def cmd_list(names: list[str]) -> int:
    for s in _sets(names or list(SET_NAMES)):
        print(f"\n[{s}] {SET_DESCRIPTIONS[s]}")
        for item in (i for i in ITEMS if i.set == s):
            print(f"  {item.name:<36} {item.use.value:<12} {item.source.ref}")
            if item.note:
                print(f"  {'':<36} {item.note}")
    return 0


def cmd_data(names: list[str]) -> int:
    for s in _sets(names):
        lock_path = LOCKS / f"{s}.json"
        lock: Lock = read_lock(lock_path) or {"root": DATA.as_posix(), "entries": {}}
        known = locked_digests(lock)
        print(f"\n[{s}] {SET_DESCRIPTIONS[s]}")
        for item in (i for i in ITEMS if i.set == s):
            placed = fetch_item(item, DATA, known)
            lock["entries"][item.key] = entry(item.use.value, item.source.ref, placed)
            write_lock(lock_path, lock)
            print(f"  ok  {item.key:<46} {len(placed):>4} file(s)  {_size(sum(p.bytes for p in placed)):>12}")
    return 0


def cmd_verify() -> int:
    problems: list[str] = []
    for path in sorted(LOCKS.glob("*.json")):
        if (lock := read_lock(path)) is None:
            continue
        found = verify_lock(lock, Path(lock["root"]))
        files = sum(len(e["files"]) for e in lock["entries"].values())
        print(f"  {'BAD' if found else 'ok '} {path.name:<16} {files} file(s)")
        problems += found
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    return 1 if problems else 0


def cmd_normalize() -> int:
    from .normalize.pipeline import run

    report = run(DATA)
    _write_report("normalize.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{'source':<32}{'rows':>9}{'rejected':>10}{'conflict':>10}{'dupes':>8}{'train':>9}{'dev':>7}{'test':>7}")
    for name, s in report["sources"].items():
        n = {split: s["splits"].get(split, {}).get("records", 0) for split in ("train", "dev", "test")}
        counts = f"{s['rows']:>9}{s['rejected']:>10}{s['conflicting']:>10}{s['duplicates']:>8}"
        print(f"{name:<32}{counts}{n['train']:>9}{n['dev']:>7}{n['test']:>7}")
    return 0


def cmd_audit(model: str) -> int:
    from .audit import audit, summary, to_json

    report = audit(DATA, MODELS_DIR / model / "tokenizer.json", REPORTS / "normalize.json")
    _write_report("data-audit.json", to_json(report))
    print("\n".join(summary(report)))
    return 0 if report.ok else 1


def cmd_env() -> int:
    from .backends import default_dtype, detect, is_apple_silicon

    backend = detect()
    print(f"backend  {backend}\ndtype    {default_dtype(backend)}\napple    {is_apple_silicon()}")
    print(f"model    {_default_model()}")
    return 0


def cmd_smoke(model: str, backend: str | None) -> int:
    from tokenizers import Tokenizer

    from .backends import BACKENDS, load

    if backend is not None and backend not in BACKENDS:
        raise SystemExit(f"unknown backend {backend!r}; choose from {BACKENDS}")
    path = _model_dir(model)
    ids = Tokenizer.from_file(str(path / "tokenizer.json")).encode("I was charged twice. Which team handles this?").ids
    started = time.perf_counter()
    backbone = load(path, backend)
    loaded = time.perf_counter()
    hidden = backbone.hidden_states(ids)
    ran = time.perf_counter()
    print(f"model    {model}\ndevice   {backbone.device}")
    print(f"hidden   {hidden.shape}  finite={bool(np.isfinite(hidden).all())}")
    print(f"timing   load {loaded - started:.1f}s  forward {ran - loaded:.2f}s for {len(ids)} tokens")
    return 0 if hidden.shape == (len(ids), backbone.hidden_size) and np.isfinite(hidden).all() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="systemone", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    sub.add_parser("model").add_argument("models", nargs="*")
    sub.add_parser("list").add_argument("sets", nargs="*")
    sub.add_parser("data").add_argument("sets", nargs="*")
    sub.add_parser("verify")
    sub.add_parser("normalize")
    sub.add_parser("audit").add_argument("--model", default=_default_model())
    sub.add_parser("env")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--model", default=_default_model())
    smoke.add_argument("--backend")
    args = parser.parse_args(argv)
    commands: dict[str, Callable[[], int]] = {
        "models": cmd_models,
        "model": lambda: cmd_model(args.models),
        "list": lambda: cmd_list(args.sets),
        "data": lambda: cmd_data(args.sets),
        "verify": cmd_verify,
        "normalize": cmd_normalize,
        "audit": lambda: cmd_audit(args.model),
        "env": cmd_env,
        "smoke": lambda: cmd_smoke(args.model, args.backend),
    }
    try:
        return commands[args.cmd]()
    except (IntegrityError, MemoryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
