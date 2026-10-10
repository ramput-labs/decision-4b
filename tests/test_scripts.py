"""The runbook's check scripts on small inputs (scripts/estimate_time.py, scripts/check_env.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import estimate_time
from scripts.check_env import version


def test_estimate_time_and_the_phase_3_decision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for stage in [*estimate_time.ROUND1, "round2"]:
        (tmp_path / stage).mkdir()
        (tmp_path / stage / "run.json").write_text(json.dumps({"tokens_per_second": 10_000}))
    # round 1: 23.2M tokens at 10k/s = 38.7 min + 4 stages x 2 min; round 2: 10.2M = 17 min + 2
    estimate_time.main(["--runs", str(tmp_path), "--budget-hours", "2.5"])
    assert capsys.readouterr().out == "round 1 47 min, round 2 19 min\ndecision: run both rounds\n"
    assert estimate_time.plan(47, 17, 130) == "decision: run round 1 only, skip phase 6, and say so in the report"
    assert estimate_time.plan(47, 17, 100) == "decision: stop and report the estimate"


def test_env_version_parsing() -> None:
    assert version("5.17.0") == (5, 17) and version("2.14.1+cu130") == (2, 14) and version("5.9") < (5, 17)


def test_claims_trace_to_their_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import verify_claims

    monkeypatch.chdir(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"clean": {"accuracy": 0.86249}}))
    (tmp_path / "README.md").write_text("den v1 scores 0.862 on core (86.2%).")
    good: list[dict[str, Any]] = [
        {"printed": "0.862", "in": ["README.md"], "source": "report.json", "path": ["clean", "accuracy"]},
        {"printed": "86.2%", "in": ["README.md"], "source": "report.json", "path": ["clean", "accuracy"], "scale": 100},
    ]
    (tmp_path / "claims.json").write_text(json.dumps(good))
    assert verify_claims.main(["--claims", "claims.json"]) == 0
    bad = [{**good[0], "printed": "0.871"}, {**good[0], "path": ["clean", "nll"]}]
    (tmp_path / "claims.json").write_text(json.dumps(bad))
    assert verify_claims.main(["--claims", "claims.json"]) == 1
    assert verify_claims.problems(bad[0]) == [
        "'0.871' from report.json: not printed in README.md",
        "'0.871' from report.json: the source says 0.862, the docs print 0.871",
    ]
