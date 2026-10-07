from __future__ import annotations

from typing import Any

from den.doctor import checks


def box(memory_gb: float, model_gb: float, disk_gb: float) -> dict[str, Any]:
    """A CUDA box `den doctor --require cuda` would pass but for memory and disk."""
    packages = {"transformers": "5.17.0", "flash-linear-attention": "0.4", "mlx": None, "mlx-lm": None}
    return {
        "packages": packages,
        "cuda": {"available": True, "gpu": "NVIDIA", "bf16": True, "driver": "580.65", "memory_gb": memory_gb},
        "model": {"key": "m", "local": "models/m", "gb": model_gb},
        "env": {"UV_NO_SYNC": "1"},
        "disk_free_gb": disk_gb,
        "unsloth_import": "ok",
    }


def failed(info: dict[str, Any], min_disk: float = 80) -> list[str]:
    return [name for name, ok, _ in checks(info, "cuda", min_disk) if not ok]


def test_an_8gb_card_trains_the_small_model_not_the_4b() -> None:
    assert failed(box(7.7, 1.6, 40), min_disk=30) == []  # qwen3.5-0.8b on an RTX 3070
    assert failed(box(7.7, 8.7, 40), min_disk=30) == ["GPU memory fits m in bf16 (~10.7 GB)"]  # qwen3.5-4b
    assert failed(box(79.2, 8.7, 200)) == []  # qwen3.5-4b on an H100


def test_the_disk_gate_follows_the_profile() -> None:
    assert failed(box(79.2, 8.7, 40)) == ["disk >= 80 GB free"]
    assert failed(box(7.7, 1.6, 40), min_disk=30) == []
