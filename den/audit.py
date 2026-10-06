from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from functools import cache
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .api import Record, RecordError, question_key, read
from .catalog import ROLES, Hub, Url, Use
from .fetch import digest
from .pins import SUITES

HELD_OUT = (
    "dev/transfer.jsonl",
    "test/transfer.jsonl",
    "dev/probes.jsonl",
    "test/probes.jsonl",
    "calibration/heldout.jsonl",
)
MAX_TRAIN_STATE_TOKENS = 7552

type Level = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Finding:
    level: Level
    where: str
    message: str


@dataclass(slots=True)
class FileStats:
    path: str
    records: int
    questions: int
    types: dict[str, int]
    noul_true_rate: float | None
    first_option_rate: float | None
    soft_targets: int
    duplicates: int
    state_tokens: dict[str, int]


@dataclass(frozen=True, slots=True)
class Keys:
    states: frozenset[str]
    items: frozenset[str]
    origins: frozenset[str]
    sources: Counter[str]

    @classmethod
    def of(cls, records: list[Record]) -> Keys:
        return cls(
            frozenset(r.fingerprint for r in records),
            frozenset(key for r in records for key, _ in _items(r)),
            frozenset(r.provenance for r in records if r.provenance),
            Counter(q.source for r in records for q in r.questions),
        )


@dataclass(slots=True)
class Report:
    files: list[FileStats] = field(default_factory=list)
    sources: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def flag(self, level: Level, where: str, message: str) -> None:
        self.findings.append(Finding(level, where, message))

    @property
    def ok(self) -> bool:
        return all(f.level != "error" for f in self.findings)


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def _items(record: Record) -> Iterator[tuple[str, int]]:
    for q in record.questions:
        if q.target is None:
            yield question_key(record.state, q), q.label


def _token_lengths(records: list[Record], tokenizer: Tokenizer | None) -> dict[str, int]:
    if tokenizer is None:
        return {}
    encoded = tokenizer.encode_batch([r.state for r in records], add_special_tokens=False)
    lengths = sorted(len(e.ids) for e in encoded)
    return {
        "median": int(statistics.median(lengths)),
        "p99": lengths[int(0.99 * (len(lengths) - 1))],
        "max": lengths[-1],
    }


def _stats(path: str, records: list[Record], tokenizer: Tokenizer | None) -> FileStats:
    questions = [q for r in records for q in r.questions]
    hard = [q for q in questions if q.target is None]
    copies = Counter((r.fingerprint, tuple((q.instructions, q.options, q.label) for q in r.questions)) for r in records)
    return FileStats(
        path=path,
        records=len(records),
        questions=len(questions),
        types=dict(Counter(q.type for q in questions)),
        noul_true_rate=_rate([q.label == 1 for q in hard if q.type == "noul"]),
        first_option_rate=_rate([q.label == 0 for q in hard if q.type == "choice"]),
        soft_targets=len(questions) - len(hard),
        duplicates=sum(n - 1 for n in copies.values()),
        state_tokens=_token_lengths(records, tokenizer),
    )


def _by_family(sources: Iterable[str]) -> str:
    """'devtools_commit_type', 'devtools_flaky' -> '2 devtools'"""
    return ", ".join(f"{n} {s}" for s, n in Counter(s.split("_")[0] for s in sources).most_common())


@cache
def _kev_names() -> dict[str, tuple[str, str]]:
    """partition path -> (manifest, Kev's file name)"""
    names = {}
    for item in SUITES:
        if item.use is Use.METADATA:
            continue
        match item.source:
            case Url(url=url, name=local):
                origin = PurePosixPath(url).name
            case Hub(files=files):
                origin, local = PurePosixPath(files[0].path).name, files[0].local
        names[f"{item.dest}/{local}"] = (f"manifests/{item.name.split('/')[0]}.json", origin)
    return names


def _built(normalized: Path) -> dict[str, dict[str, Any]]:
    if not normalized.is_file():
        return {}
    sources = json.loads(normalized.read_text(encoding="utf-8"))["sources"]
    return {f"{split}/sources/{name}.jsonl": v for name, s in sources.items() for split, v in s["splits"].items()}


def _check_counts(report: Report, root: Path, stats: FileStats, built: dict[str, dict[str, Any]]) -> None:
    if stats.path in built:
        expected, by = built[stats.path], "the normalize report"
        if digest(root / stats.path) != expected["sha256"]:
            report.flag("error", stats.path, "sha256 differs from the normalize report (rerun make normalize)")
    elif stats.path in (kev := _kev_names()):
        manifest, origin = kev[stats.path]
        expected = json.loads((root / manifest).read_text(encoding="utf-8")).get("files", {}).get(origin, {})
        by = "Kev's manifest"
    else:
        report.flag("error", stats.path, "not a Kev suite and not in the normalize report")
        return
    for key, actual in (("records", stats.records), ("questions", stats.questions)):
        if key in expected and expected[key] != actual:
            report.flag("error", stats.path, f"{actual} {key}; {by} says {expected[key]}")


def _check_file(report: Report, stats: FileStats, records: list[Record]) -> None:
    training = stats.path.startswith("train/")
    if stats.noul_true_rate is not None and not 0.25 <= stats.noul_true_rate <= 0.75:
        report.flag("warning", stats.path, f"noul labels are {stats.noul_true_rate:.0%} true")
    if stats.first_option_rate is not None and stats.first_option_rate > 0.5:
        report.flag("warning", stats.path, f"{stats.first_option_rate:.0%} of choice labels are the first option")
    if stats.duplicates:
        report.flag("warning", stats.path, f"{stats.duplicates} records repeat another record exactly")
    if training and stats.state_tokens.get("max", 0) > MAX_TRAIN_STATE_TOKENS:
        report.flag("warning", stats.path, f"longest state is {stats.state_tokens['max']} tokens; Kev trains at 7,552")

    labels: dict[str, set[int]] = defaultdict(set)
    sources: dict[str, str] = {}
    for record in records:
        for (key, label), q in zip(_items(record), (q for q in record.questions if q.target is None), strict=True):
            labels[key].add(label)
            sources[key] = q.source
    if conflicts := [sources[k] for k, v in labels.items() if len(v) > 1]:
        detail = _by_family(conflicts)
        report.flag("error" if training else "warning", stats.path, f"same question, different labels: {detail}")


def _check_leakage(report: Report, keys: dict[str, Keys]) -> None:
    for a, b in combinations(sorted(keys), 2):
        roles = {a.split("/")[0], b.split("/")[0]}
        if len(roles) == 1:
            continue
        level: Level = "error" if "train" in roles or roles == {"calibration", "test"} else "warning"
        if shared := len(keys[a].items & keys[b].items):
            report.flag(level, f"{a} x {b}", f"{shared} identical questions on identical states")
        elif shared := len(keys[a].states & keys[b].states):
            report.flag(level if "train" in roles else "warning", f"{a} x {b}", f"{shared} shared states")
        elif "train" in roles and (shared := len(keys[a].origins & keys[b].origins)):
            report.flag("error", f"{a} x {b}", f"{shared} records drawn from the same source text")


def _check_held_out(report: Report, keys: dict[str, Keys]) -> None:
    trained = {s for path, k in keys.items() if path.startswith("train/") for s in k.sources}
    for path in HELD_OUT:
        held = keys.get(path)
        seen = Counter({s: n for s, n in held.sources.items() if s in trained}) if held else Counter()
        if seen:
            detail = _by_family(seen.elements())
            report.flag("warning", path, f"held-out set has questions from trained generators: {detail}")


def _rows(path: Path) -> int | None:
    match path.suffix:
        case ".parquet":
            return int(pq.ParquetFile(path).metadata.num_rows)
        case ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                return max(0, sum(1 for _ in csv.reader(f)) - 1)
        case ".json" | ".jsonl":
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
                return len(data) if isinstance(data, list | dict) else None
            except json.JSONDecodeError:
                return sum(1 for line in text.splitlines() if line.strip() and json.loads(line) is not None)
        case _:
            return None


def _check_sources(report: Report, root: Path) -> None:
    for path in sorted((root / "sources").rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name in {".gitattributes", "README.md"}:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            rows = _rows(path)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            report.flag("error", rel, f"unreadable: {e}")
            continue
        if rows == 0:
            report.flag("error", rel, "no rows")
        parts = rel.split("/")
        dataset = "/".join(parts[: 5 if parts[2] == "breadth" else 4])
        totals = report.sources.setdefault(dataset, {"files": 0, "rows": 0})
        totals["files"] += 1
        totals["rows"] += rows or 0


def audit(root: Path, tokenizer_path: Path, normalized: Path) -> Report:
    report = Report()
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.is_file() else None
    if tokenizer is None:
        report.flag("warning", "tokenizer", "base model not downloaded, token lengths skipped (make model)")
    built = _built(normalized)
    keys: dict[str, Keys] = {}
    for role in ROLES:
        for path in sorted((root / role).rglob("*.jsonl")):
            rel = path.relative_to(root).as_posix()
            try:
                records = list(read(path))
            except RecordError as e:
                report.flag("error", rel, str(e))
                continue
            stats = _stats(rel, records, tokenizer)
            report.files.append(stats)
            _check_counts(report, root, stats, built)
            _check_file(report, stats, records)
            keys[rel] = Keys.of(records)
    _check_leakage(report, keys)
    _check_held_out(report, keys)
    _check_sources(report, root)
    return report


def to_json(report: Report) -> str:
    return json.dumps(asdict(report), indent=2) + "\n"


def summary(report: Report) -> Iterator[str]:
    columns = (("records", 9), ("questions", 11), ("noul true", 11), ("1st opt", 9), ("dupes", 7), ("tok p99", 9))
    yield f"{'file':<48}" + "".join(f"{name:>{width}}" for name, width in columns) + f"{'tok max':>9}"
    for s in report.files:
        noul = "-" if s.noul_true_rate is None else f"{s.noul_true_rate:.2f}"
        first = "-" if s.first_option_rate is None else f"{s.first_option_rate:.2f}"
        p99, longest = s.state_tokens.get("p99", "-"), s.state_tokens.get("max", "-")
        yield f"{s.path:<48}{s.records:>9}{s.questions:>11}{noul:>11}{first:>9}{s.duplicates:>7}{p99:>9}{longest:>9}"
    yield ""
    yield f"{'source':<58}{'files':>6}{'rows':>12}"
    for name, t in report.sources.items():
        yield f"{name:<58}{t['files']:>6}{t['rows'] or '-':>12}"
    yield ""
    yield from (f"{f.level.upper():<8} {f.where}: {f.message}" for f in report.findings)
    yield f"\n{'ok' if report.ok else 'FAILED'}: {sum(f.level == 'error' for f in report.findings)} error(s), " + (
        f"{sum(f.level == 'warning' for f in report.findings)} warning(s)"
    )
