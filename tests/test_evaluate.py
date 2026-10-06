from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from den.evaluate import request


def test_request_accepts_unlabelled_questions_of_every_type() -> None:
    record = request(
        {
            "state": "The invoice is late.",
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}},
                "level": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
                "late": {"type": "noul", "instructions": "Is it late?"},
            },
        }
    )
    assert [q.keys for q in record.questions] == [("billing", "tech"), ("0", "1"), ("false", "true")]


def test_request_still_validates_the_schema() -> None:
    with pytest.raises(ValueError, match="choice needs"):
        request({"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}})


def test_model_card_reports_the_measured_numbers(tmp_path: Path) -> None:
    from den.publish import card

    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "base": "qwen3.5-4b",
                "lora_rank": 16,
                "engine": "unsloth",
                "temperature": 1.23,
                "steps": 1883,
                "records_per_step": 16,
                "train_minutes": 71.5,
                "train_loss": 0.41,
                "data": ["train/core.jsonl"],
                "device": "NVIDIA H100",
                "commit": "abc",
                "versions": {"torch": "2.14.1"},
            }
        )
    )
    (tmp_path / "eval.json").write_text(
        json.dumps({"dev/core.jsonl": {"questions": 1400, "accuracy": 0.9, "nll": 0.3, "ece": 0.02}})
    )
    text = card(tmp_path, "org/systemone-test")
    assert "base_model: Qwen/Qwen3.5-4B-Base" in text
    assert "| `dev/core.jsonl` | 1400 | 0.9000 | 0.3000 | nan | 0.0200 | nan |" in text
    assert "T = 1.230" in text and "hf:org/systemone-test" in text and "| 1883 | 71.5 | 1.230 |" in text


def test_model_card_lists_every_stage(tmp_path: Path) -> None:
    from den.publish import card

    for i, name in enumerate(("1-base", "2-dates")):
        (tmp_path / name).mkdir()
        previous = str(tmp_path / "1-base") if i else None
        (tmp_path / name / "run.json").write_text(
            json.dumps(
                {
                    "base": "qwen3.5-4b",
                    "init_from": previous,
                    "data": [f"train/{name}.jsonl"],
                    "epochs": 2 - i,
                    "lr": 5e-5 if i == 0 else 2e-5,
                    "steps": 10 * (i + 1),
                    "train_minutes": 1.0,
                    "temperature": 2.0,
                    "replay": 2000 * i,
                }
            )
        )
    text = card(tmp_path / "2-dates", "org/x")
    assert text.index("| `1-base` | train/1-base.jsonl | 2 | 5e-05 |") < text.index(
        "| `2-dates` | train/2-dates.jsonl + 2000 replayed |"
    )


def test_locate_keeps_local_paths() -> None:
    from den.evaluate import locate

    assert locate("runs/x") == Path("runs/x")


def _eval(run: Path, scores: dict[str, float]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    report = {f: {"questions": 100, "accuracy": a, "nll": 0.5, "ece": 0.02} for f, a in scores.items()}
    (run / "eval.json").write_text(json.dumps(report))


def test_compare_ships_the_better_run_without_regressions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from den.evaluate import compare_main

    _eval(tmp_path / "round1", {"dev/core.jsonl": 0.86, "dev/documents.jsonl": 0.89})
    _eval(tmp_path / "round2", {"dev/core.jsonl": 0.88, "dev/documents.jsonl": 0.885})  # better mean, -0.5 pp
    _eval(tmp_path / "round3", {"dev/core.jsonl": 0.95, "dev/documents.jsonl": 0.85})  # best mean, -4 pp on documents
    compare_main([str(tmp_path / r) for r in ("round1", "round2", "round3")])
    assert capsys.readouterr().out.strip().endswith(f"ship: {tmp_path / 'round2'}")
    with pytest.raises(SystemExit, match="never on test"):
        compare_main([str(tmp_path / "round1"), "--files", "test/core.jsonl"])


def test_bundle_collects_stages_and_exact_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from den import publish

    clean = tmp_path / "clean"
    (clean / "train").mkdir(parents=True)
    (clean / "dev").mkdir()
    (clean / "train" / "core.jsonl").write_text("a\nb\nc\nd\n")
    (clean / "train" / "big.jsonl").write_text("1\n2\n3\n4\n5\n")
    (clean / "dev" / "core.jsonl").write_text("dev\n")
    monkeypatch.setattr(publish, "CLEAN", clean)
    first, last = tmp_path / "runs" / "1-base", tmp_path / "runs" / "2-more"
    for run, used, previous in (
        (first, [{"path": "train/core.jsonl", "sha256": "x", "lines": "all"}], None),
        (last, [{"path": "train/big.jsonl", "sha256": "y", "lines": [2, 5]}], str(first)),
    ):
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps({"init_from": previous, "data_used": used, "dev_files": ["dev/core.jsonl"]})
        )
        (run / "head.json").write_text("{}")
    manifest = publish.bundle(last)
    assert (last / "stages" / "1-base" / "head.json").read_text() == "{}"
    assert (last / "data" / "train" / "core.jsonl").read_text() == "a\nb\nc\nd\n"
    assert (last / "data" / "train" / "big.jsonl").read_text() == "2\n5\n"  # only the sampled lines
    assert manifest["train/big.jsonl"]["lines"] == 2 and manifest["dev/core.jsonl"]["lines"] == "all"
    assert json.loads((last / "data" / "MANIFEST.json").read_text()) == manifest


def test_metrics_brier_coverage_and_score_error() -> None:
    from den.metrics import Answer, summarize

    answers = [
        Answer([0.9, 0.1], 0, "choice", "a"),  # right, confident
        Answer([0.6, 0.4], 1, "choice", "a"),  # wrong, less confident
        Answer([0.0, 0.3, 0.7, 0.0, 0.0], 1, "score", "b"),  # wrong: expected level 1.7, truth 1
        Answer([0.5, 0.5], 0, "noul", "c", target=[0.5, 0.5]),  # unknowable: counted, not scored
    ]
    m = summarize(answers)
    assert m["questions"] == 3 and m["soft_skipped"] == 1 and abs(m["accuracy"] - 1 / 3) < 1e-9
    assert abs(m["nll"] + (math.log(0.9) + math.log(0.4) + math.log(0.3)) / 3) < 1e-9
    assert abs(m["brier"] - ((0.1**2 + 0.1**2) + (0.6**2 + 0.6**2) + ((0.3 - 1) ** 2 + 0.7**2)) / 3) < 1e-9
    assert m["accuracy_by_type"] == {"choice": 0.5, "score": 0.0}
    assert m["accuracy_by_source"] == {"a": 0.5, "b": 0.0}
    assert abs(m["score_level_mae"] - 0.7) < 1e-9
    # RPS: cumulative (0, .3, 1, 1) against the truth's (0, 1, 1, 1): (0 + .49 + 0 + 0) / 4
    assert abs(m["score_rps"] - 0.49 / 4) < 1e-9
    assert abs(m["coverage_at_5%_error"] - 1 / 3) < 1e-9  # only the most confident answer keeps error <= 5%
    # ECE bins: (.8, .9] holds (0.9, right): gap .1; (.6, .7] (0.7, wrong): .7; (.5, .6] (0.6, wrong): .6
    assert abs(m["ece"] - (0.1 + 0.7 + 0.6) / 3) < 1e-9


def test_metric_pieces_on_edge_cases() -> None:
    from den.metrics import coverage, ece

    assert ece([]) == 0.0 and coverage([]) == 0.0
    assert ece([(1.0, True)] * 10) == 0.0  # confident and always right: perfectly calibrated
    assert coverage([(0.9, True)] * 19 + [(0.8, False)]) == 1.0  # 1 error in 20 is exactly 5%
    assert coverage([(0.9, False)] + [(0.8, True)] * 19) == 1.0  # the error comes first, but 20 answers reach 5%


def test_probe_refuses_test_files_and_mixed_modes() -> None:
    from den.overfit import probe_main

    with pytest.raises(SystemExit, match="never reads test"):
        probe_main(["--dev", "test/core.jsonl"])
    with pytest.raises(SystemExit, match="zero-shot"):
        probe_main(["--head-kind", "letters", "--train", "train/core.jsonl", "--dev", "dev/core.jsonl"])
    with pytest.raises(SystemExit, match="need --train"):
        probe_main(["--dev", "dev/core.jsonl"])
