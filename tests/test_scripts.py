"""The runbook's check scripts on small inputs (scripts/check_merged.py, scripts/estimate_time.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_merged, estimate_time
from scripts.check_env import version


def test_check_merged_wants_the_base_model_type(tmp_path: Path) -> None:
    (tmp_path / "run" / "merged").mkdir(parents=True)
    (tmp_path / "run" / "merged" / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert check_merged.main([str(tmp_path / "run")]) == 0
    (tmp_path / "run" / "merged" / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    assert check_merged.main([str(tmp_path / "run")]) == 1
    with pytest.raises(SystemExit, match="--merge"):
        check_merged.main([str(tmp_path / "missing")])


def test_estimate_time_and_the_phase_3_decision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for stage in [*estimate_time.ROUND1, "round2"]:
        (tmp_path / stage).mkdir()
        (tmp_path / stage / "run.json").write_text(json.dumps({"tokens_per_second": 10_000}))
    # round 1: 23.2M tokens at 10k/s = 38.7 min + 4 stages x 2 min; round 2: 8.8M = 14.7 min + 2
    estimate_time.main(["--runs", str(tmp_path), "--budget-hours", "2.5"])
    assert capsys.readouterr().out == "round 1 47 min, round 2 17 min\ndecision: run both rounds\n"
    assert estimate_time.plan(47, 17, 130) == "decision: run round 1 only, skip phase 6, and say so in the report"
    assert estimate_time.plan(47, 17, 100) == "decision: stop and report the estimate"


def test_env_version_parsing() -> None:
    assert version("5.17.0") == (5, 17) and version("2.14.1+cu130") == (2, 14) and version("5.9") < (5, 17)
