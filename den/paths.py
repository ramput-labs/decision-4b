"""Where things live, relative to the repo root: one place to change a layout, not a constant per module."""

from __future__ import annotations

from pathlib import Path

DATA = Path("data")
CLEAN = DATA / "clean"  # training and evaluation read only here
MODELS = Path("models")
LOCKS = Path("locks")
REPORTS = Path("reports")


def model_dir(key: str) -> Path:
    """`models/<key>`, where `den model` puts a base checkpoint (it may not be downloaded)."""
    return MODELS / key


def downloaded(key: str) -> bool:
    return (model_dir(key) / "config.json").is_file()
