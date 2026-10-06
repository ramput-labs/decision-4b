from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from den.fetch import digest
from scripts import mirror


def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """data/ with a pinned suite, a normalized source, and leftovers that never belong in the copy."""
    root = tmp_path / "data"
    (root / "train" / "sources" / "yelp").mkdir(parents=True)
    (root / "train" / "core.jsonl").write_text('{"a": 1}\n')
    (root / "train" / "sources" / "yelp" / "train.jsonl").write_text('{"b": 2}\n')
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "x.lock").write_text("")
    (root / "train" / "core.jsonl.part").write_text("partial")
    locks = tmp_path / "locks"
    locks.mkdir()
    pin = {"bytes": 9, "sha256": digest(root / "train" / "core.jsonl")}
    lock = {"root": "data", "entries": {"suites/core/train": {"use": "train", "source": "x",
            "files": {"train/core.jsonl": pin}}}}  # fmt: skip
    (locks / "suites.json").write_text(json.dumps(lock))
    (locks / "models.json").write_text(json.dumps({"root": "models", "entries": {"m": {"files": {"w": pin}}}}))
    monkeypatch.setattr(mirror, "LOCKS", locks)
    monkeypatch.setattr(mirror, "DATA", Path("data"))
    return root


def test_the_copy_lists_every_data_file_and_no_leftovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = data_dir(tmp_path, monkeypatch)
    listed = mirror.files(root)
    assert set(listed) == {"train/core.jsonl", "train/sources/yelp/train.jsonl"}
    assert listed["train/core.jsonl"] == {"bytes": 9, "sha256": digest(root / "train" / "core.jsonl")}


def test_pins_are_checked_against_data_locks_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = data_dir(tmp_path, monkeypatch)
    assert mirror.pinned(root, present_only=True) == []  # models.json's root is models/: not checked here
    (root / "train" / "core.jsonl").write_text('{"a": 2}\n')
    assert mirror.pinned(root, present_only=True) == ["suites.json suites/core/train: train/core.jsonl differs from "
                                                      "its pin"]  # fmt: skip
    (root / "train" / "core.jsonl").unlink()
    assert mirror.pinned(root, present_only=True) == []
    assert mirror.pinned(root, present_only=False) == ["suites.json suites/core/train: missing train/core.jsonl"]


def test_upload_refuses_data_that_differs_from_its_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = data_dir(tmp_path, monkeypatch)
    (root / "train" / "core.jsonl").write_text("edited by hand\n")
    with pytest.raises(SystemExit, match="differ from locks"):
        mirror.upload("org/den-data", root)


def test_download_checks_every_file_against_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = data_dir(tmp_path, monkeypatch)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": mirror.files(source)}))
    calls: dict[str, Any] = {}

    def snapshot(repo: str, **kwargs: Any) -> str:  # noqa: ANN401
        calls.update(repo=repo, **kwargs)
        shutil.copytree(source / "train", Path(kwargs["local_dir"]) / "train")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    target = tmp_path / "fresh"
    assert mirror.download("org/den-data@abc123", target) == []
    assert calls["repo"] == "org/den-data" and calls["revision"] == "abc123" and calls["repo_type"] == "dataset"
    (source / "train" / "sources" / "yelp" / "train.jsonl").write_text("corrupted\n")
    shutil.rmtree(target)
    assert mirror.download("org/den-data", target) == ["train/sources/yelp/train.jsonl differs from the uploaded file"]


def test_the_mirror_needs_a_direction() -> None:
    with pytest.raises(SystemExit, match="upload,download"):
        mirror.main(["sideways"])
