"""Where things live, relative to the repo root, and how a run or base model spec becomes a directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA = Path("data")
CLEAN = DATA / "clean"  # training and evaluation read only here
MODELS = Path("models")
LOCKS = Path("locks")
REPORTS = Path("reports")
RELEASES = Path("releases")


def read_json(path: Path) -> dict[str, Any]:
    """A JSON object file, or {} when it doesn't exist."""
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return loaded


def model_dir(key: str) -> Path:
    """`models/<key>`, where `den model` puts a base checkpoint (it may not be downloaded)."""
    return MODELS / key


def downloaded(key: str) -> bool:
    return (model_dir(key) / "config.json").is_file()


def base_weights(key: str) -> Path:
    """A base checkpoint by its pinned key: `models/<key>` when downloaded, else the pinned Hub revision (cached)."""
    if downloaded(key):
        return model_dir(key)
    from huggingface_hub import snapshot_download

    from .pins import MODELS as PINNED

    if key not in PINNED:
        raise SystemExit(f"unknown base {key!r}: `den models` lists them")
    return Path(snapshot_download(PINNED[key].source.repo, revision=PINNED[key].source.revision))


def release_record(version: str, root: Path | None = None) -> dict[str, Any]:
    if not (path := (root or RELEASES) / f"{version}.json").is_file():
        raise SystemExit(f"no release {version}: {path} is missing (den release list)")
    return read_json(path)


def resolve(spec: str, root: Path | None = None) -> str:
    """`release:<version>` -> `hf:<repo>@<version>`, the tagged Hub commit; anything else unchanged."""
    if not spec.startswith("release:"):
        return spec
    record = release_record(spec.removeprefix("release:"), root)
    return f"hf:{record['hub']['repo']}@{record['version']}"


def locate(run: str, files: list[str] | None = None) -> Path:
    """A run directory: a local path, or `hf:<org>/<name>[@<revision>]` / `release:<version>` downloaded once into the
    Hub cache (only `files`, when given)."""
    run = resolve(run)
    if not run.startswith("hf:"):
        return Path(run)
    from huggingface_hub import snapshot_download

    repo, _, revision = run.removeprefix("hf:").partition("@")
    return Path(snapshot_download(repo, revision=revision or None, allow_patterns=files))
