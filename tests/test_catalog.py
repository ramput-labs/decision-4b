from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from systemone.catalog import USE_ROOTS, File, Hub, Item, SetName, Url, Use, hub_model, rev, sha256, validate
from systemone.pins import DEFAULT_MODEL, ITEMS, MODELS, SUITES

GOOD_REV = "0" * 40
GOOD_SHA = "0" * 64


def _raw(name: str, use: Use, repo: str, set_: SetName = "raw-train", dest: str | None = None) -> Item:
    root = "sources/eval" if use is Use.EVAL_ONLY else "sources/train"
    return Item(set_, name, use, Hub(repo, rev(GOOD_REV), patterns=("*",)), dest or f"{root}/{name}", "x")


def test_catalog_is_valid() -> None:
    validate(ITEMS)


def test_pins_reject_short_or_mutable_refs() -> None:
    with pytest.raises(ValueError):
        rev("main")
    with pytest.raises(ValueError):
        rev("1001bb4d")
    with pytest.raises(ValueError):
        sha256("dd633435")


def test_hub_needs_exactly_one_of_files_or_patterns() -> None:
    with pytest.raises(ValueError):
        Hub("a/b", rev(GOOD_REV))
    with pytest.raises(ValueError):
        Hub("a/b", rev(GOOD_REV), files=(File("x", None),), patterns=("*",))


def test_training_suites_are_kev_stages_and_byte_pinned() -> None:
    train = [i for i in SUITES if i.use is Use.TRAIN]
    assert {i.name for i in train} == {
        "core/train",
        "dates-unknowable/train",
        "documents/train",
        "skills/train",
        "devtools/train",
    }
    for item in train:
        match item.source:
            case Url(sha256=pin):
                assert pin is not None, item.key
            case Hub(files=files):
                assert files and all(f.sha256 is not None for f in files), item.key


def test_every_locked_test_partition_is_pinned() -> None:
    for item in (i for i in SUITES if i.use is Use.TEST):
        pins = [item.source.sha256] if isinstance(item.source, Url) else [f.sha256 for f in item.source.files]
        assert all(pins), item.key


def test_every_path_shows_its_role() -> None:
    for item in ITEMS:
        roots = [PurePosixPath(r) for r in USE_ROOTS[item.use]]
        for claim in item.claims():
            assert any(r in claim.parents or r == claim for r in roots), (item.key, claim)


def test_a_file_outside_its_role_directory_is_refused() -> None:
    with pytest.raises(ValueError, match="must live under"):
        validate((_raw("x", Use.EVAL_ONLY, "org/x", set_="raw-eval", dest="sources/train/x"),))


def test_eval_only_and_trainable_never_share_a_repo() -> None:
    with pytest.raises(ValueError, match="both trainable and eval-only"):
        validate((_raw("a", Use.TRAINABLE, "org/data"), _raw("b", Use.EVAL_ONLY, "org/data", set_="raw-eval")))


def test_never_train_repos_are_refused() -> None:
    with pytest.raises(ValueError, match="never be trained on"):
        validate((_raw("mmlu", Use.TRAINABLE, "cais/mmlu"),))
    with pytest.raises(ValueError, match="never be trained on"):
        validate((_raw("ts", Use.TRAINABLE, "tasksource/bigbench"),))


def test_set_uses_are_enforced() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        validate((_raw("x", Use.EVAL_ONLY, "org/x"),))  # eval-only source in the raw-train set


def test_overlapping_destinations_are_refused() -> None:
    a = _raw("a", Use.TRAINABLE, "org/a")
    b = Item("raw-train", "b", Use.TRAINABLE, Url("https://x/y", "f.bin", sha256(GOOD_SHA)), "sources/train/a", "x")
    with pytest.raises(ValueError, match="overlapping"):
        validate((a, b))


def test_every_model_pins_its_weights() -> None:
    for model in MODELS.values():
        assert any(f.path.endswith((".safetensors", ".pt")) for f in model.verify), model.key


def test_default_and_reference_bases_exist() -> None:
    assert DEFAULT_MODEL in MODELS and MODELS[DEFAULT_MODEL].role == "base"
    for model in MODELS.values():
        assert model.role == "base" or MODELS[str(model.base)].role == "base"


def test_any_hub_model_can_be_named_with_a_full_commit() -> None:
    model = hub_model(f"hf:Qwen/Qwen3-1.7B-Base@{GOOD_REV}")
    assert model.key == "qwen3-1.7b" and model.source.revision == GOOD_REV
    with pytest.raises(ValueError):
        hub_model("hf:Qwen/Qwen3-1.7B-Base@main")
    with pytest.raises(ValueError):
        hub_model("Qwen/Qwen3-1.7B-Base")
