"""Catalog invariants: every source is pinned, only TRAIN/TRAINABLE data reaches a gradient step, each use writes
only under `USE_ROOTS`, and no repository is both trainable and eval-only (`NEVER_TRAIN` never is)."""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Literal, NewType, get_args

Sha256 = NewType("Sha256", str)
Rev = NewType("Rev", str)

type SetName = Literal["suites", "raw-train", "raw-new", "raw-eval", "raw-bulk"]
type RepoType = Literal["model", "dataset"]
type ModelRole = Literal["base", "reference"]

NOTICE_FILE = "SOURCE-LICENSE.md"  # `den licences` writes one into each raw source directory; never data, never locked
SET_NAMES: tuple[SetName, ...] = get_args(SetName.__value__)
ROLES = ("train", "calibration", "dev", "test")  # record directories under data/, least to most held out


def record_path(path: str) -> str:
    """A path under data/clean with `.` and `..` resolved, so a guard on its first part sees its real role:
    `calibration/../dev/core.jsonl` is dev. Refuses one that leaves data/clean (`../test/...` is the pinned test)."""
    normal = posixpath.normpath(path)
    if normal == "." or normal.startswith(("/", "../")) or normal == "..":
        raise SystemExit(f"{path}: not a file under data/clean")
    return normal


def role(path: str) -> str:
    """The partition a data/clean path reads: its first directory once `..` is resolved (`record_path`)."""
    return PurePosixPath(record_path(path)).parts[0]


def sha256(value: str) -> Sha256:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"not a sha256 digest: {value!r}")
    return Sha256(value)


def rev(value: str) -> Rev:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"not a full 40-hex commit: {value!r}")
    return Rev(value)


class Use(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    DEVELOPMENT = "development"
    TEST = "test"
    TRAINABLE = "trainable"
    EVAL_ONLY = "eval_only"
    METADATA = "metadata"

    @property
    def may_train(self) -> bool:
        return self in (Use.TRAIN, Use.TRAINABLE)


SET_USES: dict[SetName, frozenset[Use]] = {
    "suites": frozenset({Use.TRAIN, Use.CALIBRATION, Use.DEVELOPMENT, Use.TEST, Use.METADATA}),
    "raw-train": frozenset({Use.TRAINABLE, Use.METADATA}),
    "raw-new": frozenset({Use.TRAINABLE}),
    "raw-eval": frozenset({Use.EVAL_ONLY}),
    "raw-bulk": frozenset({Use.TRAINABLE}),
}

SET_DESCRIPTIONS: dict[SetName, str] = {
    "suites": "Kev 1.0's frozen suites, byte-identical: train / calibration / dev / locked test + manifests",
    "raw-train": "raw public sources Kev trained from, at Kev's pinned revisions",
    "raw-new": "new trainable sources aimed at Kev's measured weaknesses",
    "raw-eval": "eval-only raw sources: transfer, devtools eval-only, breadth-v1; never trained on",
    "raw-bulk": "multi-GB raw sources behind the documents and devtools suites",
}

USE_ROOTS: dict[Use, tuple[str, ...]] = {
    Use.TRAIN: ("train",),
    Use.CALIBRATION: ("calibration",),
    Use.DEVELOPMENT: ("dev",),
    Use.TEST: ("test",),
    Use.METADATA: ("manifests", "sources/train"),
    Use.TRAINABLE: ("sources/train",),
    Use.EVAL_ONLY: ("sources/eval",),
}

NEVER_TRAIN: dict[str, str] = {
    "tasksource/": "tasksource-heldout-v1 holds out 24 whole tasksource families (names private)",
    "theatticusproject/cuad": "longdoc-v1 is built from CUAD contracts",
    "stanfordnlp/snli": "MNLI sibling; Kev excluded it so transfer stays honest",
    "cornell-movie-review-data/rotten_tomatoes": "SST parent; Kev excluded it",
    "cais/mmlu": "knowledge probe; eval-only permanently",
    "TIGER-Lab/MMLU-Pro": "knowledge probe in the probes suite",
}


@dataclass(frozen=True, slots=True)
class File:
    """Without a sha256 it is pinned by the commit, and the lock freezes its digest."""

    path: str
    sha256: Sha256 | None
    name: str | None = None

    @property
    def local(self) -> str:
        return self.name or self.path


@dataclass(frozen=True, slots=True)
class Hub:
    repo: str
    revision: Rev
    files: tuple[File, ...] = ()
    patterns: tuple[str, ...] = ()
    repo_type: RepoType = "dataset"

    def __post_init__(self) -> None:
        if bool(self.files) == bool(self.patterns):
            raise ValueError(f"{self.repo}: give exactly one of files or patterns")

    @property
    def ref(self) -> str:
        return f"hf://{self.repo_type}s/{self.repo}@{self.revision}"


@dataclass(frozen=True, slots=True)
class Url:
    url: str
    name: str
    sha256: Sha256 | None

    @property
    def ref(self) -> str:
        return self.url


type Source = Hub | Url


@dataclass(frozen=True, slots=True)
class Item:
    set: SetName
    name: str
    use: Use
    source: Source
    dest: str
    licence: str
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.set}/{self.name}"

    def claims(self) -> tuple[PurePosixPath, ...]:
        """A pattern download claims its whole directory."""
        base = PurePosixPath(self.dest)
        match self.source:
            case Url(name=name):
                return (base / name,)
            case Hub(files=files) if files:
                return tuple(base / f.local for f in files)
            case Hub():
                return (base,)


@dataclass(frozen=True, slots=True)
class Model:
    key: str
    role: ModelRole
    source: Hub
    verify: tuple[File, ...]
    bytes: int
    note: str
    base: str | None = None

    @property
    def dest(self) -> str:
        return self.key


def model_hub(repo: str, revision: str) -> Hub:
    return Hub(repo, rev(revision), patterns=("*",), repo_type="model")


def hub_model(spec: str) -> Model:
    repo, _, revision = spec.removeprefix("hf:").partition("@")
    if not spec.startswith("hf:") or repo.count("/") != 1:
        raise ValueError(f"expected hf:<org>/<name>@<commit>, got {spec!r}")
    key = repo.split("/")[1].lower().removesuffix("-base")
    return Model(key, "base", model_hub(repo, revision), (), 0, f"custom: {repo}")


def _duplicates(keys: list[str]) -> list[str]:
    return sorted(k for k, n in Counter(keys).items() if n > 1)


def validate_models(models: tuple[Model, ...]) -> None:
    keys = [m.key for m in models]
    if dupes := _duplicates(keys):
        raise ValueError(f"duplicate model keys: {dupes}")
    for m in models:
        if m.role == "reference" and m.base not in keys:
            raise ValueError(f"{m.key}: reference to unknown base {m.base!r}")


def _under(path: PurePosixPath, roots: tuple[str, ...]) -> bool:
    return any(path == PurePosixPath(r) or PurePosixPath(r) in path.parents for r in roots)


def _check_unique(items: tuple[Item, ...]) -> None:
    if dupes := _duplicates([i.key for i in items]):
        raise ValueError(f"duplicate keys: {dupes}")
    claims = sorted((c, i.key) for i in items for c in i.claims())
    for (a, key_a), (b, key_b) in pairwise(claims):
        if a == b or a in b.parents:
            raise ValueError(f"{key_a} and {key_b} write to overlapping paths: {a}, {b}")


def _check_roles(items: tuple[Item, ...]) -> None:
    for item in items:
        if item.use not in SET_USES[item.set]:
            raise ValueError(f"{item.key}: use {item.use} not allowed in set {item.set}")
        if not _under(PurePosixPath(item.dest), USE_ROOTS[item.use]):
            raise ValueError(f"{item.key}: a {item.use} file must live under {USE_ROOTS[item.use]}, not {item.dest!r}")


def _check_contamination(items: tuple[Item, ...]) -> None:
    raw: dict[str, set[Use]] = {}
    for item in items:
        if isinstance(item.source, Hub) and item.use in (Use.TRAINABLE, Use.EVAL_ONLY):
            raw.setdefault(item.source.repo, set()).add(item.use)
    if mixed := sorted(repo for repo, uses in raw.items() if len(uses) > 1):
        raise ValueError(f"raw sources both trainable and eval-only: {mixed}")
    for item in items:
        if item.use.may_train and isinstance(item.source, Hub):
            repo = item.source.repo
            for prefix, reason in NEVER_TRAIN.items():
                if repo == prefix or (prefix.endswith("/") and repo.startswith(prefix)):
                    raise ValueError(f"{item.key}: {repo} may never be trained on: {reason}")


def validate(items: tuple[Item, ...]) -> None:
    _check_unique(items)
    _check_roles(items)
    _check_contamination(items)
