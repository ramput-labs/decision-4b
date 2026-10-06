"""Leakage rules, matched on rendered state and raw-text provenance across sources and Kev's suites:
test drops Kev's train/calibration/dev; dev drops Kev's train/calibration and any test;
train drops any calibration/dev/test."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from ..catalog import ROLES
from ..records import Json, Record, RecordError, parse, question_key, read
from .sources import SPLITS, Spec, Split, class_names, specs
from .text import provenance

SALT = "normalize/v1"
CAP = 5000
MAX_CARVE = 0.1
PRIORITY: dict[Split, int] = {"test": 0, "dev": 1, "train": 2}


@dataclass(slots=True)
class Entry:
    record: dict[str, Json]
    state: str
    origin: str
    split: Split

    @property
    def keys(self) -> tuple[str, str]:
        return self.state, self.origin


@dataclass(slots=True)
class Stats:
    rows: int = 0
    rejected: int = 0
    conflicting: int = 0
    duplicates: int = 0
    carved: dict[str, int] = field(default_factory=dict)
    capped: dict[str, int] = field(default_factory=dict)
    leaked: dict[str, int] = field(default_factory=dict)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)


def bucket(key: str) -> float:
    return int(hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()[:16], 16) / 2**64


def rows(path: Path) -> Iterator[dict[str, object]]:
    match path.suffix:
        case ".parquet":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                yield from batch.to_pylist()
        case ".jsonl":
            with path.open(encoding="utf-8") as f:
                yield from (json.loads(line) for line in f if line.strip())
        case ".json":
            yield from json.loads(path.read_text(encoding="utf-8"))
        case _:
            raise ValueError(f"no reader for {path}")


def _files(root: Path, spec: Spec, globs: tuple[str, ...]) -> list[Path]:
    base = root / "sources" / spec.raw
    found = sorted({p for g in globs for p in base.glob(g) if p.is_file()})
    if not found:
        raise FileNotFoundError(f"{spec.name}: nothing matches {globs} under {base}")
    return found


def _questions(record: Record) -> dict[str, str]:
    """Each question's identity (its state, wording and options) -> its label."""
    return {question_key(record.state, q): json.dumps([q.label, q.target]) for q in record.questions}


def _convert(root: Path, spec: Spec, stats: Stats) -> Iterator[tuple[dict[str, str], Entry]]:
    native_split = {native: split for split, natives in spec.plan.items() for native in natives}
    for native, globs in spec.native.items():
        if native not in native_split:
            continue
        for path in _files(root, spec, globs):
            labels = class_names(path)
            for n, row in enumerate(rows(path)):
                stats.rows += 1
                raw = spec.convert(row, labels)
                if isinstance(raw, dict) and isinstance(questions := raw.get("questions"), dict):
                    for question in questions.values():
                        if isinstance(question, dict):
                            question["src"] = spec.name
                record_id = f"{spec.name}/{native}/{path.stem}:{n}"
                try:
                    record = parse(raw, record_id) if raw is not None else None
                except RecordError:
                    record = None
                if record is None or not isinstance(raw, dict):
                    stats.rejected += 1
                    continue
                origin = provenance(row)
                raw["_meta"] = {"id": record_id, "source": spec.name, "native_split": native, "text_sha256": origin}
                yield _questions(record), Entry(raw, record.fingerprint, origin, native_split[native])


def _deduplicate(root: Path, spec: Spec, stats: Stats) -> list[Entry]:
    """Drop records with a label-conflicting question, then keep each question once: most held-out split first,
    then records asking more questions."""
    converted = list(_convert(root, spec, stats))
    labels: dict[str, set[str]] = defaultdict(set)
    for questions, _ in converted:
        for key, label in questions.items():
            labels[key].add(label)
    consistent = [(q, e) for q, e in converted if all(len(labels[k]) == 1 for k in q)]
    stats.conflicting = len(converted) - len(consistent)
    order = sorted(range(len(consistent)), key=lambda i: (PRIORITY[consistent[i][1].split], -len(consistent[i][0]), i))
    seen: set[str] = set()
    kept: list[tuple[int, Entry]] = []
    for i in order:
        questions, entry = consistent[i]
        if questions.keys() <= seen:
            stats.duplicates += 1
            continue
        seen.update(questions)
        kept.append((i, entry))
    return [entry for _, entry in sorted(kept, key=lambda pair: pair[0])]


def _carve(spec: Spec, entries: list[Entry], stats: Stats) -> None:
    """Move donor records into missing dev/test by hash bucket of their origin, so one document never spans splits."""
    missing = [s for s in ("dev", "test") if not spec.plan.get(s)]
    if not missing:
        return
    donor: Split = "train" if spec.plan.get("train") else "test"
    groups = {e.origin for e in entries if e.split == donor}
    share = 0.5 if donor == "test" else min(MAX_CARVE, CAP / max(1, len(groups)))
    for entry in entries:
        if entry.split == donor:
            index = int(bucket(entry.origin) / share)
            if index < len(missing):
                entry.split = cast(Split, missing[index])
    stats.carved = {s: sum(e.split == s for e in entries) for s in missing}


def _cap(entries: list[Entry], stats: Stats) -> list[Entry]:
    kept = [e for e in entries if e.split == "train"]
    for split in ("dev", "test"):
        part = sorted((e for e in entries if e.split == split), key=lambda e: bucket(json.dumps(e.record["_meta"])))
        stats.capped[split] = max(0, len(part) - CAP)
        kept += part[:CAP]
    return kept


def _staged(staging: Path, spec: Spec) -> Path:
    return staging / f"{spec.name.replace('/', '--')}.jsonl"


def _kev_keys(root: Path) -> dict[str, set[str]]:
    return {
        role: {
            k
            for path in sorted((root / role).glob("*.jsonl"))
            for r in read(path)
            for k in (r.fingerprint, r.provenance)
            if k
        }
        for role in ROLES
    }


def _write(path: Path, records: Iterator[dict[str, Json]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, count, questions = hashlib.sha256(), 0, 0
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode())
            count += 1
            for qid, q in cast(dict[str, dict[str, Json]], record["questions"]).items():
                questions += 1
                labels[qid][str(q["label"]).lower()] += 1
    return {
        "records": count,
        "questions": questions,
        "sha256": h.hexdigest(),
        "labels": {k: dict(v) for k, v in labels.items()},
    }


def run(root: Path) -> dict[str, Any]:
    for split in SPLITS:
        shutil.rmtree(root / split / "sources", ignore_errors=True)
    staging = root / ".normalize"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()

    kev = _kev_keys(root)
    all_specs = specs(root)
    stats: dict[str, Stats] = {}
    held: dict[str, dict[Split, list[Entry]]] = {}
    for spec in all_specs:
        s = stats[spec.name] = Stats()
        entries = _deduplicate(root, spec, s)
        _carve(spec, entries, s)
        entries = _cap(entries, s)
        held[spec.name] = {split: [e for e in entries if e.split == split] for split in ("dev", "test")}
        with _staged(staging, spec).open("w", encoding="utf-8") as f:
            # train is streamed back from disk later, after every source's dev/test is known
            for e in entries:
                if e.split == "train":
                    f.write(json.dumps([e.state, e.origin, e.record], ensure_ascii=False) + "\n")

    def keep(entries: list[Entry], banned: set[str], name: str, split: Split) -> list[Entry]:
        kept = [e for e in entries if not (set(e.keys) & banned)]
        stats[name].leaked[split] = len(entries) - len(kept)
        return kept

    for name, parts in held.items():
        parts["test"] = keep(parts["test"], kev["train"] | kev["calibration"] | kev["dev"], name, "test")
    tests = kev["test"] | {k for parts in held.values() for e in parts["test"] for k in e.keys}
    for name, parts in held.items():
        parts["dev"] = keep(parts["dev"], kev["train"] | kev["calibration"] | tests, name, "dev")
    evaluated = (
        kev["calibration"] | kev["dev"] | tests | {k for parts in held.values() for e in parts["dev"] for k in e.keys}
    )

    for spec in all_specs:
        parts = held[spec.name]
        for split in ("dev", "test"):
            if parts[split]:
                out = root / split / "sources" / f"{spec.name}.jsonl"
                stats[spec.name].splits[split] = _write(out, (e.record for e in parts[split]))
        if spec.trainable:
            staged = _staged(staging, spec)
            leaked = Counter[str]()

            def train_records(path: Path = staged, counter: Counter[str] = leaked) -> Iterator[dict[str, Json]]:
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        state, origin, record = json.loads(line)
                        if state in evaluated or origin in evaluated:
                            counter["train"] += 1
                        else:
                            yield record

            out = root / "train" / "sources" / f"{spec.name}.jsonl"
            stats[spec.name].splits["train"] = _write(out, train_records())
            stats[spec.name].leaked["train"] = leaked["train"]
    shutil.rmtree(staging)

    return {
        "salt": SALT,
        "cap": CAP,
        "sources": {
            spec.name: {
                "raw": f"sources/{spec.raw}",
                "trainable": spec.trainable,
                "plan": dict(spec.plan),
                **asdict(stats[spec.name]),
            }
            for spec in all_specs
        },
    }
