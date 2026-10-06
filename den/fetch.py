"""A file reaches its final path only when its digest matches its pin, or its lock entry if only a commit is pinned."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from .catalog import NOTICE_FILE, Hub, Item, Model, Sha256, Url

CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    pass


class LockedFile(TypedDict):
    bytes: int
    sha256: str


class LockedEntry(TypedDict):
    use: str
    source: str
    files: dict[str, LockedFile]


class Lock(TypedDict):
    root: str
    entries: dict[str, LockedEntry]


@dataclass(frozen=True, slots=True)
class Placed:
    path: str
    bytes: int
    sha256: Sha256


def read_lock(path: Path) -> Lock | None:
    if not path.exists():
        return None
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(raw, dict) and isinstance(raw.get("root"), str) and isinstance(raw.get("entries"), dict)):
        raise IntegrityError(f"{path}: not a lock file")
    return {"root": raw["root"], "entries": raw["entries"]}


def write_lock(path: Path, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered: Lock = {"root": lock["root"], "entries": dict(sorted(lock["entries"].items()))}
    path.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def locked_digests(lock: Lock | None) -> dict[str, Sha256]:
    if lock is None:
        return {}
    return {path: Sha256(f["sha256"]) for e in lock["entries"].values() for path, f in e["files"].items()}


def entry(use: str, source: str, placed: list[Placed]) -> LockedEntry:
    return {"use": use, "source": source, "files": {p.path: {"bytes": p.bytes, "sha256": p.sha256} for p in placed}}


def digest(path: Path) -> Sha256:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return Sha256(h.hexdigest())


def _mismatch(what: object, actual: str, expected: Sha256) -> IntegrityError:
    return IntegrityError(f"{what}: sha256 {actual} does not match the pin {expected}")


def _placed(root: Path, path: Path, expected: Sha256 | None) -> Placed:
    actual = digest(path)
    if expected is not None and actual != expected:
        raise _mismatch(path, actual, expected)
    return Placed(path.relative_to(root).as_posix(), path.stat().st_size, actual)


def _is_current(path: Path, expected: Sha256 | None) -> bool:
    return expected is not None and path.is_file() and digest(path) == expected


def _download_url(url: str, dst: Path, expected: Sha256 | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as r,
        part.open("wb") as f,
    ):
        r.raise_for_status()
        for chunk in r.iter_bytes(CHUNK):
            h.update(chunk)
            f.write(chunk)
    if expected is not None and h.hexdigest() != expected:
        part.unlink()
        raise _mismatch(url, h.hexdigest(), expected)
    part.replace(dst)


def _download_hub_file(hub: Hub, repo_path: str, dst: Path, staging: Path, expected: Sha256 | None) -> None:
    got = hf_hub_download(hub.repo, repo_path, repo_type=hub.repo_type, revision=hub.revision, local_dir=staging)
    if not isinstance(got, str):
        raise TypeError(f"unexpected dry-run result for {hub.repo}/{repo_path}")
    src = Path(got)
    if expected is not None and (actual := digest(src)) != expected:
        src.unlink()
        raise _mismatch(f"{hub.ref}/{repo_path}", actual, expected)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def _snapshot(hub: Hub, dest: Path) -> list[Path]:
    snapshot_download(
        hub.repo, repo_type=hub.repo_type, revision=hub.revision, allow_patterns=list(hub.patterns), local_dir=dest
    )
    skip = (".cache", NOTICE_FILE)
    files = sorted(p for p in dest.rglob("*") if p.is_file() and not set(p.relative_to(dest).parts) & set(skip))
    if not files:
        raise IntegrityError(f"{hub.ref}: patterns {hub.patterns} matched no files")
    return files


def fetch_item(item: Item, root: Path, locked: dict[str, Sha256]) -> list[Placed]:
    dest = root / item.dest

    def expected(path: Path, pin: Sha256 | None = None) -> Sha256 | None:
        return pin or locked.get(path.relative_to(root).as_posix())

    match item.source:
        case Url(url=url, name=name, sha256=pin):
            dst = dest / name
            if not _is_current(dst, want := expected(dst, pin)):
                _download_url(url, dst, want)
            return [_placed(root, dst, want)]
        case Hub(files=files) as hub if files:
            root.mkdir(parents=True, exist_ok=True)
            placed = []
            with tempfile.TemporaryDirectory(dir=root, prefix=".staging-") as staging:
                for f in files:
                    dst = dest / f.local
                    if not _is_current(dst, want := expected(dst, f.sha256)):
                        _download_hub_file(hub, f.path, dst, Path(staging), want)
                    placed.append(_placed(root, dst, want))
            return placed
        case Hub() as hub:
            return [_placed(root, p, expected(p)) for p in _snapshot(hub, dest)]


def fetch_model(model: Model, root: Path, locked: dict[str, Sha256]) -> list[Placed]:
    dest = root / model.dest
    pins = {(dest / f.path).relative_to(root).as_posix(): f.sha256 for f in model.verify}
    placed = []
    for path in _snapshot(model.source, dest):
        rel = path.relative_to(root).as_posix()
        placed.append(_placed(root, path, pins.get(rel) or locked.get(rel)))
    if missing := sorted(set(pins) - {p.path for p in placed}):
        raise IntegrityError(f"{model.key}: pinned files missing after download: {missing}")
    return placed


def verify_lock(lock: Lock, root: Path) -> list[str]:
    problems = []
    for key, e in lock["entries"].items():
        for rel, f in e["files"].items():
            path = root / rel
            if not path.is_file():
                problems.append(f"{key}: missing {rel}")
            elif (actual := digest(path)) != f["sha256"]:
                problems.append(f"{key}: {rel} sha256 {actual} != locked {f['sha256']}")
    return problems
