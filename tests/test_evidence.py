from __future__ import annotations

import json
from pathlib import Path

import pytest

from den import evidence
from den.metrics import Answer, Row, paired_bootstrap


def _row(record: str, label: int, p: list[float], **kw: str) -> Row:
    return Row(Answer(p, label, "choice", "src"), record, "q", ("a", "b"), **kw)


def test_runs_name_their_evidence_folder() -> None:
    assert evidence.name("runs/kev-recipe/4-skills") == "kev-recipe/4-skills"
    assert evidence.name("/box/den/runs/round2") == "round2"
    assert evidence.name("release:v2") == "release-v2"
    assert evidence.name("hf:org/den@abc123") == "hf-org-den-abc123"
    assert evidence.name("evidence:kev-recipe/4-skills") == "kev-recipe/4-skills"
    assert evidence.folder("runs/round2", "dev/core.jsonl+pairs", Path("r")) == Path("r/round2/dev/core+pairs")


def test_rows_round_trip_and_keys_are_listed(tmp_path: Path) -> None:
    rows = [
        _row("r1", 0, [0.81234567, 0.18765433], variant="permuted", parent="r0"),
        Row(Answer([0.5, 0.5], 0, "noul", "unk", (0.5, 0.5)), "r2", "q", ("false", "true"), origin="unknowable"),
    ]
    where = evidence.write("runs/x", "dev/core.jsonl", {"accuracy": 1.0}, rows, tmp_path)
    evidence.write("runs/x", "dev/core.jsonl+pairs", {}, [], tmp_path)
    assert json.loads((where / "report.json").read_text()) == {"accuracy": 1.0}
    back = evidence.rows("runs/x", "dev/core.jsonl", tmp_path)
    assert back[0].answer.probs == [0.812346, 0.187654] and back[0].parent == "r0" and back[0].variant == "permuted"
    assert back[1].answer.target == [0.5, 0.5] and back[1].origin == "unknowable" and back[1].pair is None
    assert set(evidence.keys("runs/x", tmp_path)) == {"dev/core.jsonl", "dev/core+pairs.jsonl"}


def test_paired_bootstrap_by_hand() -> None:
    got = paired_bootstrap(["r1", "r1", "r2", "r3"], [0, 1, 0, 1], [1, 1, 1, 1], resamples=500)
    assert got["diff"] == 0.5 and got["questions"] == 4 and got["groups"] == 3
    assert 0 <= got["low"] <= got["diff"] <= got["high"] <= 1
    assert paired_bootstrap(["r1"], [1], [1]) == {"diff": 0.0, "low": 0.0, "high": 0.0, "questions": 1, "groups": 1}
    assert got == paired_bootstrap(["r1", "r1", "r2", "r3"], [0, 1, 0, 1], [1, 1, 1, 1], resamples=500)  # seeded


def test_paired_tells_a_real_gain_from_noise(tmp_path: Path) -> None:
    n = 200
    worse = [_row(f"r{i}", 0, [0.4, 0.6] if i % 2 else [0.6, 0.4]) for i in range(n)]  # half right
    better = [_row(f"r{i}", 0, [0.9, 0.1] if i % 10 else [0.4, 0.6]) for i in range(n)]  # 90% right
    for run, rows in (("runs/a", worse), ("runs/b", better), ("runs/c", worse[:150])):
        evidence.write(run, "dev/core.jsonl", {}, rows, tmp_path)
    up = evidence.paired("runs/a", "runs/b", "dev/core.jsonl", tmp_path)
    assert up is not None and up["questions"] == n and up["accuracy"]["b_vs_a"] == "better"
    assert up["accuracy"]["diff"] == pytest.approx(0.4) and up["nll"]["b_vs_a"] == "better"  # lower NLL is better
    down = evidence.paired("runs/b", "runs/a", "dev/core.jsonl", tmp_path)
    assert down is not None and down["accuracy"]["b_vs_a"] == "worse" and down["nll"]["b_vs_a"] == "worse"
    same = evidence.paired("runs/a", "runs/c", "dev/core.jsonl", tmp_path)
    assert same is not None and same["accuracy"]["b_vs_a"] == "no clear difference" and same["only_in_a"] == 50


def test_provenance_merges_every_evaluation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "head.json").write_text(json.dumps({"base": "qwen3.5-4b", "lora": 16, "kind": "set", "temperature": 1.4}))
    (run / "run.json").write_text(json.dumps({"commit": "abc"}))
    evidence.provenance("runs/v2", run, {"dev/core.jsonl": {"sha256": "1"}}, "def", tmp_path)
    evidence.provenance("runs/v2", run, {"test/core.jsonl": {"sha256": "2"}}, "def", tmp_path)
    saved = json.loads((tmp_path / "v2" / "provenance.json").read_text())
    assert set(saved["files"]) == {"dev/core.jsonl", "test/core.jsonl"}
    assert saved["trained_commit"] == "abc" and saved["evaluated_commit"] == "def" and saved["temperature"] == 1.4


def test_committed_evidence_locks_the_test_set_even_for_hub_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from den import evaluate

    monkeypatch.setattr(evidence, "EVIDENCE", tmp_path)
    evidence.write("release:v1", "test/core.jsonl", {}, [], tmp_path)
    loaded: list[Path] = []
    monkeypatch.setattr(evaluate, "Model", lambda path, backend: loaded.append(path))
    with pytest.raises(SystemExit, match="read once per run"):
        evaluate.evaluate_main(["--run", "release:v1", "--final", "test/core.jsonl"])
    assert not loaded


def test_compare_paired_prints_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from den import evaluate

    monkeypatch.setattr(evidence, "EVIDENCE", tmp_path)
    rows = [_row(f"r{i}", 0, [0.9, 0.1]) for i in range(20)]
    for run in ("runs/a", "runs/b"):
        evidence.write(run, "dev/core.jsonl", {}, rows, tmp_path)
        evidence.write(run, "test/core.jsonl", {}, rows, tmp_path)
    evaluate.compare_main(["--paired", "runs/a", "runs/b"])
    out = capsys.readouterr().out
    assert "dev/core.jsonl" in out and "test/core.jsonl" not in out and "no clear difference" in out
    saved = json.loads((tmp_path / "b" / "comparison-vs-a.json").read_text())
    assert set(saved["files"]) == {"dev/core.jsonl"} and saved["files"]["dev/core.jsonl"]["accuracy"]["diff"] == 0
    with pytest.raises(SystemExit, match="exactly two"):
        evaluate.compare_main(["--paired", "runs/a"])


def test_a_release_records_its_paired_comparison_with_the_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from den import release

    monkeypatch.setattr(evidence, "EVIDENCE", tmp_path / "evidence")
    rows = [_row(f"r{i}", 0, [0.9, 0.1]) for i in range(10)]
    evidence.write("runs/round2", "dev/core.jsonl", {}, rows, tmp_path / "evidence")
    evidence.write("runs/v2", "dev/core.jsonl", {}, rows, tmp_path / "evidence")
    (tmp_path / "releases").mkdir()
    (tmp_path / "releases" / "v1.json").write_text(json.dumps({"version": "v1", "evidence": "round2"}))
    got = release.against("v1", "runs/v2", tmp_path / "releases")
    assert got is not None and got["dev/core.jsonl"]["questions"] == 10
    assert release.against(None, "runs/v2", tmp_path / "releases") is None
