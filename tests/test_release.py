from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from den import paths, release


def run_dir(path: Path, dev: dict[str, float], base: str = "qwen3.5-4b", ok: bool = True, tested: bool = True) -> Path:
    path.mkdir(parents=True)
    evaluated = {f: {"accuracy": a, "questions": 100} for f, a in dev.items()}
    if tested:
        evaluated["test/core.jsonl"] = {"accuracy": 0.8, "questions": 100}
    files = {
        "run.json": {
            "base": base,
            "model": base,
            "lora": {"rank": 16},
            "data": ["train/new.jsonl"],
            "commit": "abc",
            "dataset_sha256": "d" * 64,
            "data_used": [{"path": "train/new.jsonl", "sha256": "e" * 64, "lines": "all"}],
        },
        "eval.json": evaluated,
        "head.json": {"kind": "set", "dim": 256, "layers": 1, "heads": 4, "hidden_size": 2560, "temperature": 1.3},
        "integrity.json": {"ok": ok, "files": {"head.safetensors": "f" * 64}},
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value))
    return path


def released(root: Path, version: str, dev: dict[str, float], base: str = "qwen3.5-4b") -> None:
    root.mkdir(exist_ok=True)
    entry = {
        "version": version,
        "parent": None,
        "base": base,
        "hub": {"repo": "org/den", "revision": "1" * 40},
        "dev": {f: {"accuracy": a} for f, a in dev.items()},
        "test": {},
        "created": "2026-10-01T00:00:00Z",
    }
    (root / f"{version}.json").write_text(json.dumps(entry))


def test_release_names_resolve_to_their_tagged_hub_commit(tmp_path: Path) -> None:
    released(tmp_path, "v1", {"dev/core.jsonl": 0.86})
    assert paths.resolve("release:v1", tmp_path) == "hf:org/den@v1"
    assert paths.resolve("runs/v2", tmp_path) == "runs/v2" and paths.resolve("hf:x/y@z", tmp_path) == "hf:x/y@z"
    with pytest.raises(SystemExit, match="no release v9"):
        paths.resolve("release:v9", tmp_path)


def test_regressions_count_dev_files_both_saw_never_test_or_augmented() -> None:
    parent = {
        "dev/core.jsonl": {"accuracy": 0.86},
        "dev/skills.jsonl": {"accuracy": 0.80},
        "dev/core.jsonl+pairs": {"accuracy": 0.9},
        "test/core.jsonl": {"accuracy": 0.9},
    }
    child = {
        "dev/core.jsonl": {"accuracy": 0.84},
        "dev/skills.jsonl": {"accuracy": 0.805},
        "dev/core.jsonl+pairs": {"accuracy": 0.5},
        "test/core.jsonl": {"accuracy": 0.1},
    }
    assert release.regressions(parent, child) == {"dev/core.jsonl": 0.02}
    assert release.regressions(parent, child, max_drop=0.05) == {}


def test_a_release_must_be_new_intact_tested_and_no_worse_than_its_parent(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    released(root, "v1", {"dev/core.jsonl": 0.86, "dev/skills.jsonl": 0.80})
    good = run_dir(tmp_path / "good", {"dev/core.jsonl": 0.87, "dev/skills.jsonl": 0.795})  # -0.5 pp: within 1 point
    assert release.check("v2", good, "v1", root) == {}
    with pytest.raises(SystemExit, match="immutable"):
        release.check("v1", good, None, root)
    with pytest.raises(SystemExit, match="use v1, v2"):
        release.check("version-2", good, None, root)
    with pytest.raises(SystemExit, match="integrity"):
        release.check("v2", run_dir(tmp_path / "broken", {"dev/core.jsonl": 0.9}, ok=False), "v1", root)
    with pytest.raises(SystemExit, match="no locked-test results"):
        release.check("v2", run_dir(tmp_path / "untested", {"dev/core.jsonl": 0.9}, tested=False), "v1", root)
    with pytest.raises(SystemExit, match="another base"):
        release.check("v2", run_dir(tmp_path / "other", {"dev/core.jsonl": 0.9}, base="qwen3.5-9b"), "v1", root)
    with pytest.raises(SystemExit, match="shares no dev file"):
        release.check("v2", run_dir(tmp_path / "apart", {"dev/new.jsonl": 0.9}), "v1", root)
    worse = run_dir(tmp_path / "worse", {"dev/core.jsonl": 0.90, "dev/skills.jsonl": 0.75})  # -5 pp on skills
    with pytest.raises(SystemExit, match=r"regresses on v1 beyond 0\.01: dev/skills\.jsonl -0\.0500"):
        release.check("v2", worse, "v1", root)
    assert release.check("v2", worse, "v1", root, accept="skills traded for breadth") == {"dev/skills.jsonl": 0.05}


def test_create_publishes_tags_and_records_the_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from den import publish

    root = tmp_path / "releases"
    released(root, "v1", {"dev/core.jsonl": 0.86})
    run = run_dir(tmp_path / "v2", {"dev/core.jsonl": 0.88})
    calls: dict[str, Any] = {"tags": []}

    class Api:
        def repo_exists(self, repo: str) -> bool:
            return True

        def list_repo_refs(self, repo: str) -> Any:  # noqa: ANN401
            return type("Refs", (), {"tags": [type("Tag", (), {"name": "v1"})()]})()

        def model_info(self, repo: str) -> Any:  # noqa: ANN401
            return type("Info", (), {"sha": "2" * 40})()

        def create_tag(self, repo: str, tag: str, revision: str, **kw: Any) -> None:  # noqa: ANN401
            calls["tags"].append((repo, tag, revision))

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(publish, "publish", lambda run, repo, **kw: calls.setdefault("published", (run, repo)))
    entry = release.create("v2", run, "org/den", "v1", "org/den-data@data-v2", True, None, [], root)
    assert calls["published"] == (run, "org/den") and calls["tags"] == [("org/den", "v2", "2" * 40)]
    saved = json.loads((root / "v2.json").read_text())
    assert (
        saved == entry
        and entry["parent"] == "v1"
        and entry["hub"] == {"repo": "org/den", "revision": "2" * 40, "tag": "v2"}
    )
    assert entry["data"] == {"repo": "org/den-data", "revision": "data-v2"}
    assert set(entry["dev"]) == {"dev/core.jsonl"} and set(entry["test"]) == {"test/core.jsonl"}
    assert entry["stages"][0]["dataset_sha256"] == "d" * 64 and entry["files"] == {"head.safetensors": "f" * 64}
    with pytest.raises(SystemExit, match="already has a tag v1"):
        release.create("v1", run, "org/den", None, None, True, None, [], tmp_path / "elsewhere")


def test_init_from_a_release_downloads_only_the_adapter_and_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from den import train

    released(tmp_path / "releases", "v1", {"dev/core.jsonl": 0.86})
    monkeypatch.setattr(paths, "RELEASES", tmp_path / "releases")
    seen: dict[str, Any] = {}

    def snapshot(repo: str, revision: str | None = None, allow_patterns: list[str] | None = None) -> str:
        seen.update(repo=repo, revision=revision, allow=allow_patterns)
        return str(tmp_path / "cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert paths.locate("runs/v1", train.CONTINUE_FILES) == Path("runs/v1")
    assert paths.locate("release:v1", train.CONTINUE_FILES) == tmp_path / "cache"
    assert seen["repo"] == "org/den" and seen["revision"] == "v1"
    assert "adapter_model.safetensors" in seen["allow"] and not any("merged" in a for a in seen["allow"])
