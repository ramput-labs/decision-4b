"""Runbook phase 3: a merged run loads like the base checkpoint, so its config must keep the base's model type.

uv run python -m scripts.check_merged runs/timing/round2      # prints qwen3_5; exit 1 otherwise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_type(run: Path) -> str:
    config = run / "merged" / "config.json"
    if not config.is_file():
        raise SystemExit(f"{config} is missing: train the last stage with --merge")
    return str(json.loads(config.read_text(encoding="utf-8")).get("model_type"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.check_merged")
    p.add_argument("runs", nargs="+", type=Path, help="run directories with a merged/ folder")
    p.add_argument("--expect", default="qwen3_5", help="the base checkpoint's model_type")
    args = p.parse_args(argv)
    found = {run: model_type(run) for run in args.runs}
    for run, kind in found.items():
        print(f"{'ok ' if kind == args.expect else 'BAD'} {run}: {kind}")
    return 0 if all(kind == args.expect for kind in found.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
