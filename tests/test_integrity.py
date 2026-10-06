from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from den.integrity import check, check_main
from den.model import HeadConfig, make_head, save_head

HIDDEN = 16
Q = "model.language_model.layers.{}.self_attn.q_proj.weight"


def run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A two-layer base checkpoint (one with a vision tensor) and a run that adapted and merged both q_proj."""
    torch.manual_seed(0)
    base, run = tmp_path / "base", tmp_path / "run"
    base.mkdir()
    weights = {Q.format(i): torch.randn(HIDDEN, HIDDEN) for i in range(2)}
    weights["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(2, 2)
    save_file(weights, base / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": HIDDEN}}))
    (base / "tokenizer.json").write_text("{}")
    shutil.copytree(base, run / "merged")
    merged = {k: v + 0.01 if "q_proj" in k else v for k, v in weights.items()}
    save_file(merged, run / "merged" / "model.safetensors")
    adapter = {f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_{ab}.weight": torch.randn(4, HIDDEN)
               for i in range(2) for ab in "AB"}  # fmt: skip
    save_file(adapter, run / "adapter_model.safetensors")
    save_head(run, make_head(HIDDEN, HeadConfig(kind="set", dim=8, heads=2)), base="tiny", lora=4, temperature=1.2)
    (run / "run.json").write_text(json.dumps({"base": "tiny", "lora_rank": 4, "lora": {"modules": 2}}))
    return run, base


def failed(report: dict[str, object]) -> list[str]:
    return [c["check"] for c in report["checks"] if not c["ok"]]  # type: ignore[attr-defined]


def test_an_intact_merged_run_passes_and_lists_its_files(tmp_path: Path) -> None:
    run, base = run_dir(tmp_path)
    report = check(run, base)
    assert report["ok"], failed(report)
    assert {"head.safetensors", "head.json", "adapter_model.safetensors", "run.json", "merged/model.safetensors",
            "merged/config.json", "merged/tokenizer.json"} == set(report["files"])  # fmt: skip


def test_check_run_writes_the_report_and_exits_nonzero_on_failure(tmp_path: Path) -> None:
    run, base = run_dir(tmp_path)
    assert check_main(["--run", str(run), "--base", str(base)]) == 0
    assert json.loads((run / "integrity.json").read_text())["ok"]
    (run / "merged" / "tokenizer.json").write_text('{"changed": 1}')
    assert check_main(["--run", str(run), "--base", str(base)]) == 1


@pytest.mark.parametrize(
    ("damage", "names"),
    [
        ("vision", "exactly the 2 adapted weights changed"),
        ("unmerged", "exactly the 2 adapted weights changed"),
        ("nan", "changed weights are finite"),
        ("shape", "merged tensors keep the base's names, shapes and dtypes"),
        ("missing", "merged has the base's files"),
        ("head", "head weights are finite"),
        ("hidden", "head fits the merged hidden size"),
    ],
)
def test_each_kind_of_damage_is_caught(tmp_path: Path, damage: str, names: str) -> None:
    run, base = run_dir(tmp_path)
    shard = run / "merged" / "model.safetensors"
    tensors = load_file(shard)
    if damage == "vision":
        tensors["model.visual.blocks.0.attn.qkv.weight"] += 1
    elif damage == "unmerged":
        tensors[Q.format(1)] = load_file(base / "model.safetensors")[Q.format(1)]
    elif damage == "nan":
        tensors[Q.format(0)][0, 0] = float("nan")
    elif damage == "shape":
        tensors[Q.format(0)] = tensors[Q.format(0)][:4]
    elif damage == "missing":
        (run / "merged" / "tokenizer.json").unlink()
    elif damage == "head":
        head = load_file(run / "head.safetensors")
        next(iter(head.values())).fill_(float("inf"))
        save_file(head, run / "head.safetensors")
    elif damage == "hidden":
        (run / "merged" / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 32}}))
    save_file(tensors, shard)
    report = check(run, base)
    assert not report["ok"] and names in failed(report)


def test_a_lora_run_without_merged_weights_fails_and_without_base_merged_is_checked_alone(tmp_path: Path) -> None:
    run, _ = run_dir(tmp_path)
    assert check(run)["ok"]  # no base here: finite weights and the hidden size only
    shutil.rmtree(run / "merged")
    assert failed(check(run)) == ["merged/ exists (train with --merge)"]
