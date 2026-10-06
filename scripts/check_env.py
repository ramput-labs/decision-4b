"""Runbook phase 1: is this the environment we validated? torch with CUDA, and, after `make train-setup`, Unsloth on
top without having replaced torch or transformers.

    uv run python -m scripts.check_env              # before Unsloth: torch, CUDA, the GPU
    uv run python -m scripts.check_env --unsloth    # after: + unsloth, transformers, peft, BF16
"""

from __future__ import annotations

import argparse
import importlib

MIN_TRANSFORMERS = (5, 17)  # Qwen3.5 needs transformers v5


def version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("+")[0].split(".")[:2] if part.isdigit())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.check_env")
    p.add_argument("--unsloth", action="store_true", help="also check Unsloth, transformers, peft and BF16")
    args = p.parse_args(argv)
    import torch

    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "none"
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available {cuda}  device {device}")
    problems = [] if cuda else ["CUDA is not available"]
    if args.unsloth:
        unsloth = importlib.import_module("unsloth")  # before transformers and peft, as training imports it
        import peft
        import transformers

        bf16 = bool(unsloth.is_bfloat16_supported())
        print(f"unsloth {unsloth.__version__}  transformers {transformers.__version__}  peft {peft.__version__}  "
              f"bf16 {bf16}")  # fmt: skip
        if version(transformers.__version__) < MIN_TRANSFORMERS:
            problems.append(f"transformers {transformers.__version__} < 5.17: restore it from .unsloth-pins.txt")
        if not bf16:
            problems.append("BF16 is not supported here")
    for problem in problems:
        print(f"BAD {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
