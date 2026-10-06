from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def test_the_locked_test_set_is_read_once_per_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from den import evaluate

    run = tmp_path / "run"
    _eval(run, {"test/core.jsonl": 0.8, "dev/core.jsonl": 0.9})
    loaded: list[Path] = []
    monkeypatch.setattr(evaluate, "Model", lambda path, backend: loaded.append(path))
    with pytest.raises(SystemExit, match="read once per run"):
        evaluate.evaluate_main(["--run", str(run), "--final", "test/core.jsonl"])
    assert not loaded  # refused before the model is even loaded


def _run(path: Path, kind: str, lora: int, scores: dict[str, float]) -> Path:
    _eval(path, scores)
    (path / "head.json").write_text(json.dumps({"kind": kind, "lora": lora, "temperature": 1.5}))
    return path


def test_baselines_label_each_mode_and_land_on_the_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from den.evaluate import baselines_main
    from den.publish import card

    files = {"test/core.jsonl": 0.86}
    ship = _run(tmp_path / "ship", "set", 16, files)
    a = _run(tmp_path / "a", "letters", 0, {"test/core.jsonl": 0.55})
    c = _run(tmp_path / "c", "set", 0, {"test/core.jsonl": 0.71})
    b = _run(tmp_path / "b", "letters", 16, {"test/core.jsonl": 0.80})
    baselines_main([f"D={ship}", f"A={a}", f"C={c}", f"B={b}"])
    table = json.loads((ship / "baselines.json").read_text())
    assert {k: r["mode"] for k, r in table["runs"].items()} == {"D": "D", "A": "A", "C": "C", "B": "B"}
    assert table["runs"]["C"]["results"]["test/core.jsonl"]["accuracy"] == 0.71
    assert "| `test/core.jsonl` | A | A | 100 | 0.5500 |" in capsys.readouterr().out
    (ship / "run.json").write_text(json.dumps({"base": "qwen3.5-4b"}))
    assert "## Baselines" in card(ship, "org/x") and "| `test/core.jsonl` | C | C | 100 | 0.7100 |" in card(
        ship, "org/x"
    )
    with pytest.raises(SystemExit, match="not evaluated on"):
        baselines_main([f"D={ship}", f"A={a}", "--files", "dev/core.jsonl"])


def test_card_reports_calibration_best_checkpoint_pairs_and_integrity(tmp_path: Path) -> None:
    from den.publish import card

    split = {"accuracy": 0.8, "nll": 0.7, "ece": 0.09}
    (tmp_path / "run.json").write_text(json.dumps({
        "base": "qwen3.5-4b", "steps": 900,
        "calibration_report": {"temperature": 1.4, "before": split, "after": {**split, "nll": 0.5, "ece": 0.02},
                               "calibration_split": {"before": split, "after": {**split, "ece": 0.01}}},
        "best": {"step": 800, "nll": 0.51, "accuracy": 0.81, "on": "dev sample", "questions": 600, "path": "best"},
    }))  # fmt: skip
    pairs = {"questions": 10, "accuracy": 0.9, "nll": 0.3, "ece": 0.02, "pairs": 5, "pair_accuracy": 0.6,
             "accuracy_present": 0.8, "accuracy_absent": 0.7}  # fmt: skip
    (tmp_path / "eval.json").write_text(json.dumps({"dev/core.jsonl+pairs": pairs}))
    (tmp_path / "integrity.json").write_text(json.dumps({"checks": [{"ok": True}, {"ok": True}], "files": {"a": 1}}))
    text = card(tmp_path, "org/x")
    assert "| calibration (fit) | T = 1.400 | 0.8000 | 0.7000 | 0.0100 |" in text
    assert (
        "| dev | T = 1 | 0.8000 | 0.7000 | 0.0900 |" in text
        and "| dev | T = 1.400 | 0.8000 | 0.5000 | 0.0200 |" in text
    )
    assert "step 800 of 900, nll 0.5100, accuracy 0.8100: `best/`" in text
    assert "| `dev/core.jsonl+pairs` | 5 | 0.6000 | 0.8000 | 0.7000 |" in text
    assert "2 of 2 checks passed" in text


def test_train_history_is_plain_json() -> None:
    from den.train import train_history

    log: list[dict[str, Any]] = [
        {"loss": 1.23456789, "grad_norm": 0.5, "learning_rate": 5e-5, "epoch": 0.1, "step": 10},
        {"train_runtime": 12.0, "step": 20, "total_flos": None},
    ]
    assert train_history(log) == [{"loss": 1.234568, "grad_norm": 0.5, "learning_rate": 5e-05, "epoch": 0.1,
                                   "step": 10}, {"train_runtime": 12.0, "step": 20}]  # fmt: skip
    json.dumps(train_history(log))


def _row(p: list[float], label: int, record: str, keys: tuple[str, ...] = ("a", "b"), **kw: Any) -> Any:  # noqa: ANN401
    from den.metrics import Answer, Row

    return Row(Answer(p, label), record, "q", keys, **kw)


def test_robustness_matches_kevs_checks_by_hand() -> None:
    from den.metrics import robustness

    rows = [
        _row([0.8, 0.2], 0, "r1"),
        _row(
            [0.4, 0.6], 0, "r1/permuted", ("b", "a"), variant="permuted", parent="r1"
        ),  # aligned [0.6, 0.4]: 0.2, kept
        _row([0.3, 0.7], 1, "r2"),
        _row(
            [0.1, 0.9], 1, "r2/permuted", ("b", "a"), variant="permuted", parent="r2"
        ),  # aligned [0.9, 0.1]: 0.6, flip
        _row([0.9, 0.1], 0, "p/a", pair="p", sibling="a"),  # answers differ, prediction flips: right
        _row([0.2, 0.8], 1, "p/b", pair="p", sibling="b"),
        _row([0.6, 0.4], 0, "s/a", pair="s", sibling="a"),  # answers agree, prediction holds: invariant
        _row([0.7, 0.3], 0, "s/b", pair="s", sibling="b"),
        _row([0.6, 0.4], 0, "lonely/a", pair="lonely", sibling="a"),  # its sibling was not sampled
        _row([0.95, 0.05], 0, "u1", origin="unknowable", control="c1"),
        _row([0.55, 0.45], 0, "u2", origin="unknowable", control="c2"),
        _row([0.99, 0.01], 0, "c1", origin="unknowable_control"),
        _row([0.40, 0.60], 0, "c2", origin="unknowable_control"),
    ]
    out = robustness(rows)
    assert out["permutation"] == {"n": 2, "mean_max_delta": pytest.approx(0.4), "flip_rate": 0.5}
    assert out["paired_flip"] == {
        "pairs": 1,
        "flip_rate": 1.0,
        "both_correct_rate": 1.0,
        "incomplete_pairs": 1,
        "invariant_pairs": 1,
        "invariant_both_correct_rate": 1.0,
        "invariance_rate": 1.0,
    }
    u = out["unknowable"]
    assert u["n"] == 2 and u["mean_max_p"] == pytest.approx(0.75) and u["share_at_0_9"] == 0.5
    assert u["control_acc"] == 0.5 and u["paired_confidence_drop"] == pytest.approx((0.04 + 0.05) / 2)
    assert u["share_less_confident_than_control"] == 1.0  # 0.95 < 0.99 and 0.55 < 0.60
    assert out["variants"]["permuted"] == {"n": 2, "accuracy": 0.5}
    assert out["clean"]["questions"] == 9  # clean rows, the unknowable records left out
    assert robustness([_row([0.8, 0.2], 0, "r1")]) == {}  # no structure, nothing to add


def test_rotate_options_moves_every_option_and_keeps_the_answer() -> None:
    import random

    from den.api import Json, parse
    from den.prompt import rotate_options

    raw: Json = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "label": "c",
           "criteria": {"a": "first", "b": "second", "c": "third", "d": "fourth"}},
           "n": {"type": "noul", "instructions": "?", "label": True}}}  # fmt: skip
    record = parse(raw, "r")
    for seed in range(20):
        q, n = rotate_options(record, random.Random(seed)).questions
        assert all(a != b for a, b in zip(q.keys, record.questions[0].keys, strict=True))  # nothing stays put
        assert q.keys[q.label] == "c" and q.options[q.label] == "c: third"
        assert n == record.questions[1]  # noul is never reordered
