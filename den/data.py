"""What a run trains on: files under data/clean sampled reproducibly, and `Shuffled`, the augmented training items.

Every file a run reads and the exact lines it took go into `data_used`, the run's data manifest (`den publish`
uploads the same data from it).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

from .api import Record, option_text, read
from .fetch import digest
from .licences import RANK, Kind, record_kind
from .paths import CLEAN
from .prompt import DISTRACTORS, NONE_OPTIONS, Example, Style, augment, encode, none_pair, shuffle_options, split

KEV_TRAIN = ("core", "dates-unknowable", "documents", "skills", "devtools")  # Kev's stages 1-4
UNCAPPED = 1 << 30  # dev, calibration and serving read states of any length

type Manifest = list[dict[str, object]]


def load(paths: Sequence[Path]) -> list[Record]:
    return [record for path in paths for record in read(path)]


def kinds(path: Path) -> dict[int, Kind]:
    """Each non-empty line's licence class, by 1-based line number."""
    with path.open(encoding="utf-8") as f:
        return {n: record_kind(json.loads(line)) for n, line in enumerate(f, 1) if line.strip()}


def licensed(path: Path, worst: Kind | None) -> list[int]:
    """The non-empty lines of `path` whose sources are no more restrictive than `worst` (all of them if None)."""
    if worst is None:
        with path.open(encoding="utf-8") as f:
            return [n for n, line in enumerate(f, 1) if line.strip()]
    return [n for n, k in kinds(path).items() if RANK[k] <= RANK[worst]]


def licence_counts(used: Manifest) -> dict[str, int]:
    """Trained records per licence class, over the run's data manifest."""
    counts: dict[str, int] = {}
    for entry in used:
        lines, classes = entry["lines"], kinds(CLEAN / str(entry["path"]))
        for n in lines if isinstance(lines, list) else classes:
            counts[classes[n]] = counts.get(classes[n], 0) + 1
    return {k: counts[k] for k in RANK if k in counts}


def sample(path: Path, cap: int, seed: int, used: Manifest | None = None, worst: Kind | None = None) -> list[Record]:
    """Up to `cap` records of one file, the same ones for the same seed (only those lines are parsed), recorded in
    `used`. With `worst`, only records licensed at most that class are eligible."""
    numbered = licensed(path, worst)
    rng = random.Random(f"{seed}:{path.as_posix()}")
    lines = sorted(rng.sample(numbered, min(cap, len(numbered))))
    if used is not None:
        used.append({"path": path.relative_to(CLEAN).as_posix(), "sha256": digest(path), "lines": lines})
    return list(read(path, set(lines)))


def sources(
    cap: int, seed: int, known: Sequence[Record] = (), used: Manifest | None = None, worst: Kind | None = None
) -> list[Record]:
    """At most `cap` records from each normalized trainable source, leaving out texts already in `known` (Kev's core
    holds 1,000 of each of its public datasets, and a text trained on twice would count double)."""
    seen = {r.provenance for r in known if r.provenance} | {r.fingerprint for r in known}
    return [
        r
        for path in sorted((CLEAN / "train" / "sources").rglob("*.jsonl"))
        for r in sample(path, cap, seed, used, worst)
        if r.provenance not in seen and r.fingerprint not in seen
    ]


def gather(args: argparse.Namespace, used: Manifest) -> list[Record]:
    """The run's training rows, one per question: `--data`, `--sources-cap` of every public source, and `--replay`
    from each `--replay-from` suite. A source text in any suite the run trains on or replays from is left out."""
    worst: Kind | None = args.max_licence
    if args.limit:  # a rehearsal: a seeded slice of every file
        args.replay, args.sources_cap = min(args.replay, args.limit), min(args.sources_cap, args.limit)
        records = [r for d in args.data for r in sample(CLEAN / d, args.limit, args.seed, used, worst)]
    elif worst:
        records = [r for d in args.data for r in sample(CLEAN / d, 1 << 62, args.seed, used, worst)]
    else:
        records = load([CLEAN / d for d in args.data])
        used += [{"path": d, "sha256": digest(CLEAN / d), "lines": "all"} for d in args.data]
    suites = args.replay_from if args.replay else []
    replayed = [r for f in suites for r in sample(CLEAN / f, args.replay, args.seed, used, worst)]
    if args.sources_cap:
        if not (CLEAN / "train" / "sources").is_dir():
            raise SystemExit(f"{CLEAN}/train/sources is missing: make data-download")
        known = load([CLEAN / f for f in [*args.data, *suites]])
        records += sources(args.sources_cap, args.seed, known, used, worst)
    return [one for r in [*records, *replayed] for one in split(r)]


def examples(
    records: Sequence[Record],
    tokenizer: Tokenizer,
    max_state: int,
    rng: random.Random | None = None,
    style: Style = "dash",
) -> tuple[list[Example], int]:
    """Encoded records, one per question (choice options shuffled when `rng` is given), and how many were skipped."""
    out, skipped = [], 0
    for record in (one for r in records for one in split(r)):
        if (example := encode(shuffle_options(record, rng) if rng else record, tokenizer, max_state, style)) is None:
            skipped += 1
        else:
            out.append(example)
    return out, skipped


def describe(name: str, lengths: Sequence[int], questions: int, skipped: int) -> str:
    ordered = sorted(lengths) or [0]
    p99 = ordered[int(0.99 * (len(ordered) - 1))]
    return (
        f"{name:<8}{len(lengths):>8} records{questions:>9} questions  "
        f"tokens median {int(statistics.median(ordered))} p99 {p99} max {ordered[-1]} total {sum(ordered) / 1e6:.1f}M  "
        f"skipped {skipped}"
    )


def added(tokenizer: Tokenizer, style: Style) -> int:
    """The most tokens Kev's augmentation adds to a question: its longest added option line, plus one for a merge
    across the line boundary."""
    marker = "Z) " if style == "letters" else "- "
    lines = [f"{marker}{option_text(k, d)}\n" for k, d in (*NONE_OPTIONS, *DISTRACTORS)]
    return 1 + max(len(tokenizer.encode(line, add_special_tokens=False).ids) for line in lines)


class Shuffled:
    """Training items encoded on access, a fresh variant each epoch.

    `augment`: "kev" (none-of-the-above and distractor options, shuffled order), "shuffle" (order only) or "none".
    With `p_none_pair`, that share of eligible records also yields Kev's minimal pair: two more items, the question
    with the true option present and removed. Records whose state is too long are dropped.

    A variant depends on (seed, item, epoch) only, never on how often an item was read: a resumed run skips batches
    without reading them, and dataloader workers read copies. The Trainer sets the epoch through `set_epoch`.
    """

    def __init__(
        self,
        records: Sequence[Record],
        tokenizer: Tokenizer,
        max_state: int,
        seed: int,
        augment: str = "kev",
        p_none_pair: float = 0.0,
        style: Style = "dash",
    ) -> None:
        self.records: list[Record] = []
        self.longest = 0  # the longest an item can get once augmented: the backbone's max_seq_length
        self.items: list[tuple[int, int]] = []  # (record, part): part 0 the record, 1 and 2 its pair's halves
        self.lengths: list[int] = []  # for the length-grouped sampler
        self.tokenizer, self.seed, self.augment, self.style = tokenizer, seed, augment, style
        self.epoch = 0
        grown = added(tokenizer, style) if augment == "kev" else 0
        for record in records:
            if (example := encode(record, tokenizer, max_state, style)) is None:
                continue
            i = len(self.records)
            self.records.append(record)
            self._add(i, 0, len(example.ids), grown)
            if random.Random(f"{seed}:{i}:pair").random() < p_none_pair and (
                pair := none_pair(record, random.Random(0))
            ):
                halves = [encode(half, tokenizer, UNCAPPED, style) for half in pair]
                for part, half in enumerate(halves, 1):
                    if all(halves) and half is not None:  # letters: the added option can take a question past Z
                        self._add(i, part, len(half.ids))

    def _add(self, record: int, part: int, length: int, grown: int = 0) -> None:
        self.items.append((record, part))
        self.lengths.append(length)
        self.longest = max(self.longest, length + grown)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, n: int) -> Example:
        i, part = self.items[n]
        rng = random.Random(f"{self.seed}:{i}:{part and 'pair'}:{self.epoch}")  # both halves share one stream
        record = self.records[i]
        if part:
            pair = none_pair(record, rng)
            assert pair is not None
            record = pair[part - 1]
        elif self.augment == "kev":
            record = augment(record, rng)
        elif self.augment == "shuffle":
            record = shuffle_options(record, rng)
        example = encode(record, self.tokenizer, UNCAPPED, self.style)
        if example is None:  # letters: an added option took the question past 26; train it without the addition
            example = encode(self.records[i], self.tokenizer, UNCAPPED, self.style)
        assert example is not None
        return example
