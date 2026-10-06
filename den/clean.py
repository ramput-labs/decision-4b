"""Training-ready copies of every suite and normalized source, under data/clean/.

Kev's suites are sha256-pinned and `data/*/sources/` match the normalize report, so neither is edited. This writes
cleaned copies beside them and reports what changed. Whitespace is only collapsed in prose: diffs, code, templated
policies and MMLU tables mean their indentation, so those get the safe repairs only (NFC, control and zero-width
characters, CRLF). Train files are deduplicated; dev, test and calibration keep every record, so Kev's counts and the
deliberate a/b "unknowable" pairs (same state, different labels, to test abstention) stay as built.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .api import Json
from .catalog import ROLES
from .text import _CONTROL, clean

# sources whose text is prose; everything else (code, diffs, templates, MMLU, buried-fact probes) keeps its layout
PROSE = frozenset(
    {
        *("agnews", "ag-news", "amazon", "amazon-reviews", "banking77", "boolq", "cfpb", "dbpedia14", "emotion"),
        *("imdb", "mnli", "multinli", "paws", "qnli", "sciq", "sst5", "trec", "yelp", "yelp-full", "aegis"),
        *("tweet_offensive", "tweeteval-offensive", "when2call", "prompt_injection", "prompt-injection"),
    }
)
ESCAPED = frozenset({"yelp", "agnews"})  # Kev's suites store these with literal \n and \" in place of newlines
ENTITIES = frozenset({"agnews", "ag-news", "mnli", "multinli", "tweet_offensive", "tweeteval-offensive"})


def _name(record: dict[str, Json]) -> str:
    meta = record.get("_meta")
    source = meta.get("source", "") if isinstance(meta, dict) else ""
    return str(source).rsplit("/", 1)[-1]


def repair(text: str, source: str) -> str:
    if source in PROSE:
        return clean(text, escapes=source in ESCAPED and "\n" not in text, markup=source in ENTITIES)
    text = _CONTROL.sub("", unicodedata.normalize("NFC", text))
    return text.replace("\r\n", "\n")


def _walk(value: Json, source: str) -> Json:
    match value:
        case str():
            return repair(value, source)
        case list():
            return [_walk(v, source) for v in value]
        case dict():
            return {k: _walk(v, source) for k, v in value.items()}
        case _:
            return value


def _question(question: Json, source: str) -> Json:
    if source in PROSE and isinstance(question, dict) and isinstance(text := question.get("instructions"), str):
        return {**question, "instructions": repair(text, source)}
    return question


def _clean(record: dict[str, Json]) -> dict[str, Json]:
    source = _name(record)
    questions = record["questions"]
    if isinstance(questions, dict):
        questions = {qid: _question(q, source) for qid, q in questions.items()}
    return {**record, "state": _walk(record["state"], source), "questions": questions}


def _identity(record: dict[str, Json]) -> str:
    body = {k: v for k, v in record.items() if k != "_meta"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


_PROBLEMS = {
    "literal_escapes": re.compile(r"\\n"),
    "edge_space": re.compile(r"\A\s|\s\Z"),
    "runs_of_space": re.compile(r"[ \t]{2,}"),
    "blank_runs": re.compile(r"\n{3,}"),
    "control_or_zero_width": _CONTROL,
    "entities": re.compile(r"&(?:amp|quot|lt|gt|#\d+);|<br\s*/?>"),
}


def _strings(value: Json) -> Iterator[str]:
    match value:
        case str():
            yield value
        case list():
            for v in value:
                yield from _strings(v)
        case dict():
            for v in value.values():
                yield from _strings(v)


def _problems(record: dict[str, Json], counts: Counter[str]) -> None:
    text = list(_strings(record["state"]))
    for name, pattern in _PROBLEMS.items():
        if any(pattern.search(t) for t in text):
            counts[name] += 1


def run(root: Path) -> dict[str, dict[str, object]]:
    out = root / "clean"
    shutil.rmtree(out, ignore_errors=True)
    report: dict[str, dict[str, object]] = {}
    for role in ROLES:
        for path in sorted((root / role).rglob("*.jsonl")):
            rel = path.relative_to(root)
            before: Counter[str] = Counter()
            after: Counter[str] = Counter()
            changed = dropped = total = 0
            seen: set[str] = set()
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with path.open(encoding="utf-8") as src, dst.open("w", encoding="utf-8") as f:
                for line in filter(str.strip, src):
                    record: dict[str, Json] = json.loads(line)
                    total += 1
                    _problems(record, before)
                    fixed = _clean(record)
                    changed += fixed != record
                    _problems(fixed, after)
                    if role == "train":
                        if (key := _identity(fixed)) in seen:
                            dropped += 1
                            continue
                        seen.add(key)
                    f.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            report[rel.as_posix()] = {
                "records": total,
                "rewritten": changed,
                "duplicates_dropped": dropped,
                "written": total - dropped,
                "problems_before": dict(before),
                "problems_after": dict(after),
            }
    return report
