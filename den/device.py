"""Inference backbones: the hidden states of a checkpoint on MLX (Apple Silicon), CUDA or CPU, one interface."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, get_args

import numpy as np
from numpy.typing import NDArray

type Backend = Literal["mlx", "cuda", "cpu"]
type DType = Literal["bfloat16", "float32"]

BACKENDS: tuple[Backend, ...] = get_args(Backend.__value__)


@dataclass(frozen=True, slots=True)
class Device:
    backend: Backend
    name: str
    memory_gb: float
    dtype: DType


class Backbone(Protocol):
    @property
    def device(self) -> Device: ...

    @property
    def hidden_size(self) -> int: ...

    def hidden_states(self, ids: Sequence[int]) -> NDArray[np.float32]:
        """Shape (len(ids), hidden_size), after the final norm."""
        ...


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def detect() -> Backend:
    if forced := os.environ.get("DEN_BACKEND"):
        if forced not in BACKENDS:
            raise ValueError(f"DEN_BACKEND={forced!r}; choose from {BACKENDS}")
        return forced
    if is_apple_silicon():
        return "mlx"
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def default_dtype(backend: Backend) -> DType:
    return "float32" if backend == "cpu" else "bfloat16"


def text_config(path: Path) -> dict[str, object]:
    """Vision-language checkpoints nest the language config under `text_config`."""
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def hidden_size(path: Path) -> int:
    return int(str(text_config(path)["hidden_size"]))


def weight_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.glob("*.safetensors"))


def check_fits(path: Path, available_gb: float, dtype: DType) -> None:
    stored = weight_bytes(path) / 2**30
    needed = stored * (2 if dtype == "float32" else 1) * 1.2  # float32 doubles bf16 weights; 20% headroom
    if needed > available_gb:
        raise MemoryError(
            f"{path.name} needs about {needed:.0f} GB in {dtype}; this device has {available_gb:.0f} GB. "
            "Choose a smaller model (den models) or a larger device."
        )


def load(path: Path, backend: Backend | None = None, dtype: DType | None = None) -> Backbone:
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} is not a downloaded model (den model <key>)")
    backend = backend or detect()
    dtype = dtype or default_dtype(backend)
    if backend == "mlx":
        from .mlx_model import MlxBackbone

        return MlxBackbone(path, dtype)
    from .torch_model import TorchBackbone

    return TorchBackbone(path, backend, dtype)
