"""`den doctor`: is this machine ready, and with what? Versions come from package metadata, so nothing heavy is
imported (Unsloth is only imported, to check it, on a CUDA machine).

    den doctor                      # report; MLX on a Mac, CUDA on Linux
    den doctor --require cuda       # exit 1 unless CUDA training will work: GPU, BF16, driver, Unsloth, kernels
    den doctor --verify             # also re-hash the model's weight shards against locks/models.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "huggingface-hub",
    "unsloth", "unsloth_zoo", "flash-linear-attention", "causal-conv1d", "mlx", "mlx-lm",
)  # fmt: skip
MIN_DRIVER = 580  # the locked torch is the CUDA 13 build
MIN_TRANSFORMERS = (5, 17)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _major_minor(version: str | None) -> tuple[int, int]:
    parts = (version or "0.0").split(".")
    return int(parts[0]), int("".join(c for c in parts[1] if c.isdigit()) or 0)


def _driver() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
    )
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def _cuda() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if not info["available"]:
        info["mps"] = torch.backends.mps.is_available()
        return info
    props = torch.cuda.get_device_properties(0)
    info |= {
        "devices": torch.cuda.device_count(),
        "gpu": props.name,
        "memory_gb": round(props.total_memory / 2**30, 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "bf16": torch.cuda.is_bf16_supported(),
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "driver": _driver(),
    }
    return info


def _unsloth_import() -> str:
    try:
        import unsloth

        return f"ok ({getattr(unsloth, '__version__', '?')})"
    except Exception as e:  # its import fails loudly without CUDA or on a version clash: report, don't crash
        return f"failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}"


def _model(key: str, verify: bool) -> dict[str, Any]:
    from .fetch import digest
    from .pins import MODELS

    info: dict[str, Any] = {"key": key}
    if key in MODELS:
        info["pinned"] = f"{MODELS[key].source.repo}@{MODELS[key].source.revision}"
    path = Path("models") / key
    info["local"] = str(path) if (path / "config.json").is_file() else None
    lock = Path("locks/models.json")
    files = (
        json.loads(lock.read_text(encoding="utf-8"))["entries"].get(key, {}).get("files", {}) if lock.is_file() else {}
    )
    info["locked_files"] = len(files)
    if info["local"] and files:
        missing = [f for f in files if not (Path("models") / f).is_file()]
        wrong_size = [
            f for f, m in files.items() if f not in missing and (Path("models") / f).stat().st_size != m["bytes"]
        ]
        info["files_ok"] = not missing and not wrong_size
        if missing or wrong_size:
            info["files_problem"] = {"missing": missing[:3], "wrong_size": wrong_size[:3]}
        if verify and info["files_ok"]:
            bad = [f for f, m in files.items() if digest(Path("models") / f) != m["sha256"]]
            info["sha256_ok"] = not bad
    if info["local"]:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        info |= {"model_type": config.get("model_type"), "hidden_size": text.get("hidden_size"),
                 "layers": text.get("num_hidden_layers")}  # fmt: skip
    return info


def report(model: str, verify: bool = False) -> dict[str, Any]:
    from .device import detect

    cuda = _cuda()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "backend": detect(),
        "packages": {name: _version(name) for name in PACKAGES},
        "cuda": cuda,
        "model": _model(model, verify),
        "env": {k: os.environ.get(k) for k in ("UV_NO_SYNC", "DEN_BACKEND", "DEN_MODEL", "HF_HOME")}
        | {"HF_TOKEN set": bool(os.environ.get("HF_TOKEN"))},
        "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 1),
    }
    if cuda["available"]:
        info["unsloth_import"] = _unsloth_import() if info["packages"]["unsloth"] else "not installed"
    return info


def _driver_ok(driver: object) -> bool:
    return bool(driver) and int(str(driver).split(".")[0]) >= MIN_DRIVER


def checks(info: dict[str, Any], require: str | None) -> list[tuple[str, bool, str]]:
    """(name, ok, detail) for what the required platform needs."""
    out: list[tuple[str, bool, str]] = []
    packages, cuda, model = info["packages"], info["cuda"], info["model"]
    out.append(("model downloaded", bool(model.get("local")), str(model.get("local"))))
    if model.get("local") and "files_ok" in model:
        out.append(("model files match the lock", bool(model["files_ok"]), str(model.get("files_problem", ""))))
    if "sha256_ok" in model:
        out.append(("model sha256 match the lock", bool(model["sha256_ok"]), ""))
    out.append(
        (
            "transformers >= 5.17",
            _major_minor(packages["transformers"]) >= MIN_TRANSFORMERS,
            str(packages["transformers"]),
        )
    )
    if require == "cuda":
        driver = cuda.get("driver")
        out += [
            ("CUDA available", bool(cuda["available"]), str(cuda.get("gpu"))),
            ("BF16 supported", bool(cuda.get("bf16")), ""),
            (f"NVIDIA driver >= {MIN_DRIVER}", _driver_ok(driver), str(driver)),
            ("Unsloth imports", str(info.get("unsloth_import", "")).startswith("ok"), str(info.get("unsloth_import"))),
            ("flash-linear-attention installed", bool(packages["flash-linear-attention"]), "DeltaNet kernels"),
            ("UV_NO_SYNC=1", info["env"]["UV_NO_SYNC"] == "1", "a plain `uv run` would undo the Unsloth install"),
            ("disk >= 80 GB free", info["disk_free_gb"] >= 80, f"{info['disk_free_gb']} GB"),
        ]  # fmt: skip
    elif require == "mlx":
        out += [("MLX installed", bool(packages["mlx"] and packages["mlx-lm"]), str(packages["mlx"]))]
    return out


def doctor_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den doctor")
    p.add_argument("--model", default=os.environ.get("DEN_MODEL", "qwen3.5-4b"))
    p.add_argument("--require", choices=("cuda", "mlx"), help="fail unless this platform is ready to train/serve")
    p.add_argument("--verify", action="store_true", help="re-hash the weight shards (slow: ~9 GB)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    info = report(args.model, args.verify)
    results = checks(info, args.require)
    info["checks"] = [{"check": n, "ok": ok, "detail": d} for n, ok, d in results]
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"python {info['python']}  {info['platform']}  backend {info['backend']}")
        print("packages " + "  ".join(f"{k} {v}" for k, v in info["packages"].items() if v))
        print("cuda     " + "  ".join(f"{k} {v}" for k, v in info["cuda"].items()))
        if "unsloth_import" in info:
            print(f"unsloth  import {info['unsloth_import']}")
        print("model    " + "  ".join(f"{k} {v}" for k, v in info["model"].items()))
        print(f"disk     {info['disk_free_gb']} GB free")
        for name, ok, detail in results:
            print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail and not ok else ""))
    return 0 if all(ok for _, ok, _ in results) else 1
