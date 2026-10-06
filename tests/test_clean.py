from __future__ import annotations

import json
from pathlib import Path

from den.clean import repair, run


def _record(state: object, source: str, label: str = "a") -> dict[str, object]:
    question = {"type": "choice", "instructions": "Pick?", "criteria": {"a": None, "b": None}, "label": label}
    return {"state": state, "questions": {"q": question}, "_meta": {"source": source, "id": source}}


def test_prose_is_collapsed_and_code_keeps_its_layout() -> None:
    assert repair("  a   b \n\n\n\n c ", "cfpb") == "a b\n\nc"
    diff = " def f():\n     return 1  \n"
    assert repair(diff, "codereviewer") == diff
    assert repair("x​\r\ny", "commitpackft") == "x\ny"


def test_escapes_are_undone_only_where_the_suite_stored_them_literally() -> None:
    assert repair('Great!\\n\\nSaid \\"wow\\"', "yelp") == 'Great!\n\nSaid "wow"'
    assert repair("print('a\\nb')", "commitpackft") == "print('a\\nb')"
    assert repair("Real\nnewline and \\n literal", "yelp") == "Real\nnewline and \\n literal"


def test_run_dedupes_train_only_and_never_drops_eval(tmp_path: Path) -> None:
    rows = [_record("same  text", "yelp"), _record("same text", "yelp"), _record("other", "yelp")]
    for role in ("train", "dev", "calibration", "test"):
        (tmp_path / role).mkdir()
        (tmp_path / role / "core.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = run(tmp_path)
    assert report["train/core.jsonl"]["written"] == 2
    assert report["train/core.jsonl"]["duplicates_dropped"] == 1
    assert report["dev/core.jsonl"]["written"] == 3
    out = [json.loads(line) for line in (tmp_path / "clean/train/core.jsonl").read_text().splitlines()]
    assert [r["state"] for r in out] == ["same text", "other"]


def test_conflicting_labels_in_eval_are_kept(tmp_path: Path) -> None:
    for role in ("train", "dev", "calibration", "test"):
        (tmp_path / role).mkdir()
    pair = [_record("unknowable", "unknowable", "a"), _record("unknowable", "unknowable", "b")]
    (tmp_path / "dev" / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pair), encoding="utf-8")
    assert run(tmp_path)["dev/probes.jsonl"]["written"] == 2
