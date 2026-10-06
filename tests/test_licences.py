from __future__ import annotations

import json
from pathlib import Path

import pytest

from den.catalog import NOTICE_FILE
from den.licences import BY_KEY, SOURCES, TEXTS, allowed, classify, data_files, kind, notices, source_key, write
from den.pins import ITEMS


def test_every_pinned_raw_item_has_a_recorded_licence() -> None:
    """A new pin can't land without its terms: each raw catalog item resolves to a registry source."""
    for item in ITEMS:
        if item.dest.startswith("sources/"):
            assert source_key(item.name) in BY_KEY


def test_the_registry_is_well_formed() -> None:
    assert len({s.key for s in SOURCES}) == len(SOURCES)
    names = [n for s in SOURCES for n in s.names]
    assert len(names) == len(set(names))  # one name, one source
    for s in SOURCES:
        assert s.evidence and s.url.startswith("http") and s.attribution
        assert all((TEXTS / f"{t}.txt").is_file() for t in s.texts)


def test_names_resolve_and_unknown_ones_are_refused() -> None:
    assert source_key("yelp") == source_key("sentiment/yelp-full") == "yelp"
    assert source_key("hard_temporal_numeric") == "kev"  # Kev's skills families, by prefix
    with pytest.raises(KeyError, match="add it to den"):
        source_key("brand_new_dataset")


def test_a_file_takes_its_most_restrictive_source() -> None:
    assert kind(["mmlu", "boolq"]) == "share-alike"
    assert kind(["kev", "sciq", "trec"]) == "unspecified"
    assert kind(["banking77", "yelp"]) == "restricted"
    assert kind([]) == "open"
    for k, private, public in [("open", True, True), ("share-alike", True, True), ("non-commercial", True, True),
                               ("unspecified", True, False), ("restricted", False, False)]:  # fmt: skip
        assert allowed(k, public=False) is private and allowed(k, public=True) is public  # type: ignore[arg-type]


def _record(source: str, **meta: str) -> str:
    q = {"q": {"type": "noul", "instructions": "?", "label": True}}
    return json.dumps({"state": "s", "questions": q, "_meta": {"source": source, **meta}}) + "\n"


def tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for rel, lines in {
        "train/core.jsonl": [_record("banking77"), _record("yelp"), _record("compositional")],
        "dev/probes.jsonl": [_record("mmlu"), _record("buried", parent_source="paws")],
        "dev/sources/transfer/sciq.jsonl": [_record("transfer/sciq")],
        "dev/sources/topic/trec.jsonl": [_record("topic/trec")],
        "clean/train/core.jsonl": [_record("banking77")],  # a cleaned copy holds what its original holds
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("".join(lines))
    raw = root / "sources/train/sentiment/yelp-full/yelp_review_full/train.parquet"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    (root / "manifests").mkdir()
    (root / "manifests/core.json").write_text("{}")
    (root / ".normalize").mkdir()
    (root / ".normalize/x.jsonl").write_text("staging")
    return root


def test_classify_reads_each_files_sources(tmp_path: Path) -> None:
    root = tree(tmp_path)
    got = classify(root, data_files(root))
    assert ".normalize/x.jsonl" not in got  # staging never counts
    assert got["train/core.jsonl"] == ("restricted", ["banking77", "kev", "yelp"])
    assert got["clean/train/core.jsonl"] == got["train/core.jsonl"]
    assert got["dev/probes.jsonl"] == ("open", ["mmlu", "paws"])  # a buried probe is its parent dataset's text
    assert got["dev/sources/transfer/sciq.jsonl"] == ("non-commercial", ["sciq"])
    assert got["sources/train/sentiment/yelp-full/yelp_review_full/train.parquet"] == ("restricted", ["yelp"])
    assert got["manifests/core.json"] == ("open", ["kev"])


def test_notices_are_written_and_deterministic(tmp_path: Path) -> None:
    root = tree(tmp_path)
    classified, excluded = notices(root)
    assert set(excluded) == {"train/core.jsonl", "clean/train/core.jsonl",
                             "sources/train/sentiment/yelp-full/yelp_review_full/train.parquet"}  # fmt: skip
    _, public = notices(root, public=True)
    assert set(public) == {*excluded, "dev/sources/topic/trec.jsonl"}  # unlicensed: kept to private copies
    card = (root / "README.md").read_text()
    assert (
        card.startswith("---\nlicense: other\n")
        and "license_link" not in card
        and "| Yelp Review Full | Yelp Dataset Terms of Use | restricted |" in card
    )
    assert "`train/core.jsonl`: restricted" in card  # the card says what was left out
    assert (root / "LICENSES/texts/CC-BY-NC-3.0.txt").read_text() == (TEXTS / "CC-BY-NC-3.0.txt").read_text()
    assert "`train/core.jsonl` (not in this copy)" in (root / "LICENSES/yelp.md").read_text()
    raw = (
        root / "sources/train/sentiment/yelp-full" / NOTICE_FILE
    ).read_text()  # one notice per dataset, subfolders too
    assert (
        "| `yelp_review_full/train.parquet` | restricted | [Yelp Review Full](../../../../LICENSES/yelp.md) | no ("
        in raw
    )
    beside = (root / "train" / NOTICE_FILE).read_text()  # and beside every folder of suite or normalized files
    assert "| `core.jsonl` | restricted |" in beside and "- Licence: **CC-BY-4.0** (open)" in beside
    assert "[CC-BY-4.0](../LICENSES/texts/CC-BY-4.0.txt)" in beside
    assert (
        "| `sciq.jsonl` | non-commercial | [SciQ](../../../LICENSES/sciq.md) | yes |"
        in (root / "dev/sources/transfer" / NOTICE_FILE).read_text()
    )
    assert "| `dev/probes.jsonl` | open | mmlu, paws |" in (root / "LICENSES.md").read_text()
    (root / "gone").mkdir()
    (root / "gone" / NOTICE_FILE).write_text("a folder that no longer holds data")
    write(root, classified, excluded)
    assert not (root / "gone" / NOTICE_FILE).exists()  # stale notices go
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    write(root, classified, excluded)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert not any(rel.endswith(NOTICE_FILE) or rel.startswith("LICENSES") for rel in classified)  # notices aren't data


def test_every_source_in_the_real_data_is_known() -> None:
    root = Path("data")
    if not (root / "train").is_dir():
        pytest.skip("no data/ here")
    rels = [r for r in data_files(root) if r.endswith(".jsonl") and "/sources/" not in r and not r.startswith("clean/")]
    classified = classify(root, rels)  # raises on any source without a recorded licence
    if "train/core.jsonl" in classified:  # a Hub copy (`make download-data`) leaves restricted files out
        assert classified["train/core.jsonl"][0] == "restricted"  # Kev's core holds Yelp and Amazon reviews
