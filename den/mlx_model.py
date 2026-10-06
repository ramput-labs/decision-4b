from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
from numpy.typing import NDArray

from .device import Device, DType, check_fits, hidden_size

_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


class Tower(Protocol):
    def __call__(self, inputs: mx.array) -> mx.array: ...


def text_tower(model: object) -> Tower:
    """Vision-language checkpoints keep it at `language_model.model`."""
    outer = getattr(model, "language_model", model)
    tower: Tower | None = getattr(outer, "model", None)
    if tower is None or not callable(tower):
        raise TypeError(f"{type(model).__name__} has no text tower")
    return tower


class MlxBackbone:
    def __init__(self, path: Path, dtype: DType) -> None:
        info = mx.device_info()
        available = int(info["max_recommended_working_set_size"]) / 2**30
        check_fits(path, available, dtype)
        model, _ = load_model(path)
        model.set_dtype(_DTYPES[dtype])
        self._text = text_tower(model)
        self._hidden_size = hidden_size(path)
        self._device = Device("mlx", str(info["device_name"]), available, dtype)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def hidden_states(self, ids: Sequence[int]) -> NDArray[np.float32]:
        hidden = self._text(mx.array([list(ids)]))[0].astype(mx.float32)
        mx.eval(hidden)
        return np.asarray(hidden)
