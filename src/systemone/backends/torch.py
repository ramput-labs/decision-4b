from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM

from . import Backend, Device, DType, check_fits, hidden_size

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _device(backend: Backend, dtype: DType) -> Device:
    if backend == "cuda":
        props = torch.cuda.get_device_properties(0)
        return Device("cuda", props.name, props.total_memory / 2**30, dtype)
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return Device("cpu", platform.processor() or platform.machine(), memory, dtype)


class TorchBackbone:
    def __init__(self, path: Path, backend: Backend, dtype: DType) -> None:
        if backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SYSTEM_ONE_BACKEND=cuda but torch sees no CUDA device")
        self._device = _device(backend, dtype)
        check_fits(path, self._device.memory_gb, dtype)
        self._torch_device = torch.device("cuda:0" if backend == "cuda" else "cpu")
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=_DTYPES[dtype])
        self._text = lm.base_model.to(self._torch_device).eval()
        self._hidden_size = hidden_size(path)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @torch.inference_mode()
    def hidden_states(self, ids: Sequence[int]) -> NDArray[np.float32]:
        tokens = torch.tensor([list(ids)], device=self._torch_device)
        hidden = self._text(input_ids=tokens).last_hidden_state[0]
        return np.asarray(hidden.float().cpu(), dtype=np.float32)
