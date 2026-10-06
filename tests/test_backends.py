from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from systemone.backends import check_fits, detect, is_apple_silicon, load, text_config

BASE = Path("models/qwen3.5-4b")


def test_detect_honours_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "cpu")
    assert detect() == "cpu"
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "tpu")
    with pytest.raises(ValueError):
        detect()


def test_apple_silicon_uses_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEM_ONE_BACKEND", raising=False)
    if is_apple_silicon():
        assert detect() == "mlx"


def test_text_config_reads_nested_and_flat_configs(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"text_config": {"hidden_size": 2560}}')
    assert text_config(tmp_path)["hidden_size"] == 2560
    (tmp_path / "config.json").write_text('{"hidden_size": 1024}')
    assert text_config(tmp_path)["hidden_size"] == 1024


def test_models_too_large_for_the_device_are_refused(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 2**20)
    check_fits(tmp_path, available_gb=1.0, dtype="bfloat16")
    with pytest.raises(MemoryError):
        check_fits(tmp_path, available_gb=0.001, dtype="bfloat16")


@pytest.mark.model
@pytest.mark.skipif(not (BASE / "config.json").exists(), reason="base model not downloaded")
@pytest.mark.skipif(not is_apple_silicon() or importlib.util.find_spec("torch") is None, reason="needs MLX and torch")
def test_mlx_matches_torch() -> None:
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(BASE / "tokenizer.json")).encode("Refund one of the two charges.").ids
    mlx = load(BASE, "mlx").hidden_states(ids)
    torch = load(BASE, "cpu").hidden_states(ids)
    cosine = (mlx * torch).sum(-1) / (np.linalg.norm(mlx, axis=-1) * np.linalg.norm(torch, axis=-1))
    assert mlx.shape == torch.shape and cosine.min() > 0.99
