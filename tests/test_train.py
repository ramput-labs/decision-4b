from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer

from den.api import Record, parse
from den.calibrate import fit_temperature
from den.model import HeadConfig, PointerHead, question_loss
from den.prompt import Example, encode, shuffle_options, split

TOKENIZER = Path("models/qwen3.5-4b/tokenizer.json")


def _record() -> Record:
    return parse(
        {
            "state": "The invoice is late.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "money", "tech": None},
                    "label": "tech",
                },
                "late": {"type": "noul", "instructions": "Is it late?", "label": True},
            },
        },
        "r",
    )


@pytest.mark.skipif(not TOKENIZER.is_file(), reason="qwen3.5-4b tokenizer not downloaded")
def test_encode_marks_final_and_closing_tokens() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    example = encode(_record(), tokenizer)
    assert example is not None
    newline = tokenizer.encode("\n", add_special_tokens=False).ids
    colon = tokenizer.encode(":", add_special_tokens=False).ids
    assert len(example.finals) == len(example.closes) == 2
    assert [len(c) for c in example.closes] == [2, 2]
    for closes in example.closes:
        assert all(example.ids[i : i + len(newline)] == tuple(newline) or example.ids[i] == newline[-1] for i in closes)
    assert all(example.ids[f] == colon[-1] for f in example.finals)
    assert example.labels == (1, 1)
    assert example.finals[0] < example.closes[1][0]  # questions are laid out in order
    for starts, closes in zip(example.starts, example.closes, strict=True):
        assert all(s <= c for s, c in zip(starts, closes, strict=True))
        assert all(c < s for c, s in zip(closes, starts[1:], strict=False))  # spans do not overlap
        assert tokenizer.decode(list(example.ids[starts[0] : closes[0] + 1])).startswith("- ")


@pytest.mark.skipif(not TOKENIZER.is_file(), reason="qwen3.5-4b tokenizer not downloaded")
def test_long_state_is_skipped() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    assert encode(_record(), tokenizer, max_state=2) is None


def _example(closes: tuple[int, ...], starts: tuple[int, ...], label: int = 1) -> Example:
    return Example(
        ids=tuple(range(10)), finals=(9,), closes=(closes,), starts=(starts,), labels=(label,), targets=(None,)
    )


def test_pointer_head_scores_and_loss() -> None:
    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(dim=4, layers=1, heads=2))
    example = _example((3, 6), (1, 4))
    hidden = torch.randn(1, 10, 8, requires_grad=True)
    (scores,) = head(hidden, [example])[0]
    assert scores.shape == (2,)
    loss = question_loss(scores, 1, None)
    assert torch.isclose(loss, -torch.log_softmax(scores, 0)[1])
    soft = question_loss(scores, 0, (0.25, 0.75))
    assert torch.isclose(soft, -(0.25 * torch.log_softmax(scores, 0)[0] + 0.75 * torch.log_softmax(scores, 0)[1]))
    loss.backward()  # type: ignore[no-untyped-call]
    assert hidden.grad is not None and hidden.grad[0, 3].abs().sum() > 0 and hidden.grad[0, 0].abs().sum() == 0


def test_head_starts_as_kevs_pointer() -> None:
    """Span values, mixing and prior start at zero, so initial scores are q(final) . k(close) / sqrt(dim)."""
    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(dim=4, layers=2, heads=2)).eval()
    hidden = torch.randn(1, 10, 8)
    (scores,) = head(hidden, [_example((3, 6), (1, 4))])[0]
    h = head.norm(hidden[0])
    expected = head.k(h[[3, 6]]) @ head.q(h[9]) / 2
    assert torch.allclose(scores, expected, atol=1e-5)


def test_scores_follow_the_options_not_their_order() -> None:
    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(dim=4, layers=1, heads=2))
    for p in head.parameters():  # leave the zero init so mixing and the prior actually act
        torch.nn.init.normal_(p, std=0.3)
    head.eval()
    hidden = torch.randn(1, 10, 8)
    (forward,) = head(hidden, [_example((2, 5, 8), (1, 3, 6))])[0]
    (backward,) = head(hidden, [_example((8, 5, 2), (6, 3, 1))])[0]
    assert torch.allclose(forward, backward.flip(0), atol=1e-5)


def test_head_batches_questions_with_different_option_counts() -> None:
    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(dim=4))
    two = Example((0,) * 10, (9, 9), ((3, 6), (2, 4, 7)), ((1, 4), (1, 3, 5)), (0, 2), (None, None))
    one = _example((3, 6), (1, 4))
    scores = head(torch.randn(2, 10, 8), [two, one])
    assert [[len(s) for s in per] for per in scores] == [[2, 3], [2]]
    assert all(torch.isfinite(s).all() for per in scores for s in per)


def test_shuffle_options_keeps_the_answer() -> None:
    record = parse(
        {
            "state": "s",
            "questions": {
                "c": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {k: None for k in "abcdef"},
                    "label": "d",
                    "target": {"d": 3, "a": 1},
                },
                "n": {"type": "noul", "instructions": "Yes?", "label": True},
                "s": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"], "label": 2},
            },
        },
        "r",
    )
    seen = set()
    for seed in range(20):
        shuffled = shuffle_options(record, random.Random(seed))
        c, n, s = shuffled.questions
        assert c.keys[c.label] == "d" and c.target is not None
        assert dict(zip(c.keys, c.target, strict=True)) == {"a": 0.25, "b": 0, "c": 0, "d": 0.75, "e": 0, "f": 0}
        assert (n, s) == record.questions[1:]  # noul and score keep their order
        seen.add(c.label)
    assert len(seen) > 3


def test_fit_temperature_recovers_overconfidence() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4000, 4) * 2
    labels = torch.multinomial(logits.softmax(-1), 1).squeeze(1).tolist()
    t = fit_temperature(list(logits * 3), labels, [None] * len(labels))  # three times too sharp
    assert 2.6 < t < 3.4


def test_shuffled_reorders_options_on_every_visit() -> None:
    from den.train import Shuffled

    class Chars:  # a stand-in tokenizer: one token per character
        def encode(self, text: str, add_special_tokens: bool = False) -> object:
            return type(
                "Encoding", (), {"ids": [ord(c) for c in text], "offsets": [(i, i + 1) for i in range(len(text))]}
            )

    record = parse(
        {
            "state": "s",
            "questions": {
                "c": {"type": "choice", "instructions": "Pick", "criteria": dict.fromkeys("abcdef"), "label": "d"}
            },
        },
        "r",
    )
    long = parse({"state": "x" * 50, "questions": {"n": {"type": "noul", "instructions": "?", "label": True}}}, "l")
    data = Shuffled([record, long], Chars(), max_state=10, seed=0, augment="shuffle")  # type: ignore[arg-type]
    assert len(data) == 1 and data.lengths == [len(data[0].ids)]
    picks = [data[0] for _ in range(8)]
    assert len({e.ids for e in picks}) > 1  # a new order on each visit
    assert all(chr(e.ids[e.starts[0][e.labels[0]] + 2]) == "d" for e in picks)  # the label follows its option


MODEL = Path("models/qwen3.5-4b")


@pytest.mark.model
@pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="qwen3.5-4b not downloaded")
def test_lora_adapts_every_text_linear_and_nothing_else() -> None:
    """The PEFT engine wraps the same targets Unsloth gets: every attention, DeltaNet and MLP projection of the text
    decoder, none of the vision tower. Starts as the identity, and gradients reach the adapters and the head."""
    from den.device import hidden_size
    from den.model import LoraConfig, adapted, load_backbone, text_tower

    model = load_backbone(MODEL, LoraConfig(rank=4, alpha=8), 64, 0, engine="peft")
    names = adapted(model)
    kinds = {name.rsplit(".", 1)[-1] for name in names}
    assert len(names) == 8 * 4 + 24 * 5 + 32 * 3  # full attention: q k v o; DeltaNet: qkv z a b out; MLP: 3
    assert kinds == {*("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")} | {
        *("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    }
    assert all(p.requires_grad == ("lora_" in n) for n, p in model.named_parameters())
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}  # bf16 would round updates away
    tower = text_tower(model)
    assert type(tower).__name__ == "Qwen3_5TextModel"

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    example = encode(_record(), tokenizer)
    assert example is not None
    ids = torch.tensor([example.ids])
    with model.disable_adapter(), torch.no_grad():
        base = tower(input_ids=ids, use_cache=False).last_hidden_state
    hidden = tower(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(hidden, base)  # lora_B starts at zero

    head = PointerHead(hidden_size(MODEL), HeadConfig(dim=16, heads=2))
    (scores,) = head(hidden, [example])
    torch.stack(
        [question_loss(s, label, None) for s, label in zip(scores, example.labels, strict=True)]
    ).sum().backward()  # type: ignore[no-untyped-call]
    lora_b = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert all(g is not None for g in lora_b) and any(float(g.abs().sum()) > 0 for g in lora_b if g is not None)
    assert all(p.grad is not None for p in head.parameters() if p.requires_grad)


def test_fit_temperature_stays_bounded_when_every_answer_is_wrong() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0]] * 50)  # confident and always wrong: the unbounded optimum is T -> inf
    assert fit_temperature(list(logits), [1] * 50, [None] * 50) <= 20.0 + 1e-4


def test_sample_is_seeded_and_capped(tmp_path: Path) -> None:
    import json

    from den.train import sample

    rows = [
        {"state": f"s{i}", "questions": {"q": {"type": "noul", "instructions": "?", "label": i % 2 == 0}}}
        for i in range(50)
    ]
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    first = [r.state for r in sample(path, 10, seed=1)]
    assert len(first) == 10 and len(set(first)) == 10
    assert first == [r.state for r in sample(path, 10, seed=1)]  # same seed, same slice
    assert first != [r.state for r in sample(path, 10, seed=2)]
    assert len(sample(path, 500, seed=1)) == 50  # a cap above the file size takes everything


@pytest.mark.skipif(not TOKENIZER.is_file(), reason="qwen3.5-4b tokenizer not downloaded")
def test_limit_caps_every_file_for_a_rehearsal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import shutil

    from den.train import main

    for rel in ("train/a.jsonl", "train/b.jsonl", "dev/a.jsonl", "calibration/a.jsonl"):
        path = tmp_path / "data" / "clean" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"state": f"{rel} {i}", "questions": {"q": {"type": "noul", "instructions": "?", "label": True}}}
                for i in range(50)]  # fmt: skip
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "models" / "m").mkdir(parents=True)
    (tmp_path / "models" / "m" / "config.json").write_text("{}")
    shutil.copy(TOKENIZER, tmp_path / "models" / "m" / "tokenizer.json")
    monkeypatch.chdir(tmp_path)
    common = ["--model", "m", "--data", "train/a.jsonl", "--dev", "dev/a.jsonl", "--calibration", "calibration/a.jsonl"]
    replay = ["--replay", "30", "--replay-from", "train/b.jsonl", "--dry-run"]
    assert main([*common, *replay]) == 0
    assert capsys.readouterr().out.split()[:2] == ["train", "80"]  # 50 + 30 replayed
    assert main([*common, *replay, "--limit", "10"]) == 0
    assert capsys.readouterr().out.split()[:2] == ["train", "20"]  # 10 from each file, replay capped too


def _choice(n: int = 5, label: str = "c") -> Record:
    keys = "abcdefgh"[:n]
    return parse(
        {
            "state": "s",
            "questions": {
                "q": {"type": "choice", "instructions": "Pick", "criteria": dict.fromkeys(keys), "label": label}
            },
        },
        "r",
    )


def test_kev_augment_keeps_labels_true() -> None:
    from den.prompt import NONE_OPTIONS, augment

    nones = {k for k, _ in NONE_OPTIONS}
    kinds = {"replaced": 0, "added": 0, "plain": 0}
    for seed in range(400):
        (q,) = augment(_choice(), random.Random(seed)).questions
        assert len(set(q.options)) == len(q.options) and len(q.keys) == len(q.options)
        if q.keys[q.label] in nones:  # the true option was swapped for a none option, which is now right
            assert "c" not in q.keys and len(q.keys) == 5
            kinds["replaced"] += 1
        else:
            assert q.keys[q.label] == "c"
            kinds["added" if len(q.keys) == 6 else "plain"] += 1
    assert 20 < kinds["replaced"] < 70 and 80 < kinds["added"] < 140  # Kev's 10% and 12% + 15%, out of 400


def test_augment_leaves_soft_targets_and_other_types_alone() -> None:
    from den.prompt import augment

    record = parse(
        {
            "state": "s",
            "questions": {
                "c": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": dict.fromkeys("abc"),
                    "label": "a",
                    "target": {"a": 1, "b": 1},
                },
                "n": {"type": "noul", "instructions": "?", "label": True},
            },
        },
        "r",
    )
    for seed in range(50):
        c, n = augment(record, random.Random(seed)).questions
        assert sorted(c.keys) == ["a", "b", "c"] and c.target is not None
        assert dict(zip(c.keys, c.target, strict=True)) == {"a": 0.5, "b": 0.5, "c": 0}
        assert n == record.questions[1]


def test_none_pair_differs_only_in_the_true_option() -> None:
    from den.prompt import none_pair

    for seed in range(30):
        pair = none_pair(_choice(), random.Random(seed))
        assert pair is not None
        (present,), (absent,) = pair[0].questions, pair[1].questions
        none_key = absent.keys[absent.label]
        assert present.keys[present.label] == "c" and none_key in present.keys and "c" not in absent.keys
        assert [k for k in present.keys if k != "c"] == list(absent.keys)  # same order, one option fewer
    assert none_pair(_choice(2, "a"), random.Random(0)) is None  # needs three options


def test_shuffled_adds_pair_halves_that_agree() -> None:
    from den.train import Shuffled

    class Chars:
        def encode(self, text: str, add_special_tokens: bool = False) -> object:
            return type(
                "Encoding", (), {"ids": [ord(c) for c in text], "offsets": [(i, i + 1) for i in range(len(text))]}
            )

    records = [_choice() for _ in range(40)]
    data = Shuffled(records, Chars(), max_state=99, seed=0, augment="kev", p_none_pair=0.25)  # type: ignore[arg-type]
    pairs = [n for n, (_, part) in enumerate(data.items) if part == 1]
    assert 4 <= len(pairs) <= 18 and len(data) == 40 + 2 * len(pairs)
    for n in pairs:
        present, absent = data[n], data[n + 1]
        assert len(present.closes[0]) == len(absent.closes[0]) + 1  # the same visit: one option fewer, nothing else


def test_warm_start_refuses_a_different_shape(tmp_path: Path) -> None:
    from den.model import LoraConfig, save_head, warm_start

    head = PointerHead(8, HeadConfig(dim=4, layers=1, heads=2))
    save_head(tmp_path, head, base="qwen3.5-4b", lora=0, temperature=1.5)
    fresh = PointerHead(8, HeadConfig(dim=4, layers=1, heads=2))
    assert warm_start(None, fresh, tmp_path, "qwen3.5-4b", LoraConfig(rank=0)) == 1.5
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), head.state_dict().values(), strict=True))
    with pytest.raises(SystemExit, match="lora rank"):
        warm_start(None, fresh, tmp_path, "qwen3.5-4b", LoraConfig(rank=16))
    save_head(tmp_path, head, base="qwen3.5-4b", lora=16, temperature=1.5)  # an older head: no alpha, no rsLoRA
    for other in (LoraConfig(rank=16, alpha=16), LoraConfig(rank=16, rslora=True)):
        with pytest.raises(SystemExit, match="lora scale"):
            warm_start(None, fresh, tmp_path, "qwen3.5-4b", other)


def test_lora_scale_follows_alpha_and_rslora() -> None:
    from den.model import LoraConfig

    assert LoraConfig(rank=16, alpha=32).scale == 2.0 and LoraConfig(rank=16, alpha=32, rslora=True).scale == 8.0
    assert LoraConfig(rank=0).scale == 0.0


def test_head_saves_as_safetensors_and_checks_the_backbone(tmp_path: Path) -> None:
    import json

    from den.model import load_head, save_head

    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(dim=4, layers=1, heads=2))
    for p in head.parameters():
        torch.nn.init.normal_(p)
    save_head(tmp_path, head, temperature=2.0)
    assert {f.name for f in tmp_path.iterdir()} == {"head.safetensors", "head.json"}  # no pickle
    meta = json.loads((tmp_path / "head.json").read_text())
    assert meta["format"] == "den-pointer-head" and meta["hidden_size"] == 8 and meta["kind"] == "set"
    loaded, _ = load_head(tmp_path, hidden=8)
    hidden = torch.randn(1, 10, 8)
    example = _example((3, 6), (1, 4))
    assert torch.allclose(loaded.eval()(hidden, [example])[0][0], head.eval()(hidden, [example])[0][0])
    with pytest.raises(SystemExit, match="hidden size 8"):
        load_head(tmp_path, hidden=2560)


def test_pointer_kind_is_kevs_head() -> None:
    torch.manual_seed(0)
    head = PointerHead(8, HeadConfig(kind="pointer", dim=4))
    assert {name.split(".")[0] for name, _ in head.named_parameters()} == {"norm", "q", "k"}
    hidden = torch.randn(1, 10, 8)
    (scores,) = head(hidden, [_example((3, 6), (1, 4))])[0]
    h = head.norm(hidden[0])
    assert torch.allclose(scores, head.k(h[[3, 6]]) @ head.q(h[9]) / 2, atol=1e-5)


def test_ordinal_loss_charges_far_misses_more() -> None:
    near = torch.log(torch.tensor([0.05, 0.05, 0.8, 0.05, 0.05]))  # mass on level 2, the truth is level 1
    far = torch.log(torch.tensor([0.05, 0.05, 0.05, 0.05, 0.8]))  # mass on level 4
    plain = question_loss(near, 1, None)
    assert torch.isclose(plain, question_loss(far, 1, None))  # cross-entropy alone can't tell them apart
    assert question_loss(near, 1, None, ordinal=1.0) < question_loss(far, 1, None, ordinal=1.0)
    assert question_loss(near, 1, (0.5, 0.5, 0, 0, 0), ordinal=1.0) == question_loss(near, 1, (0.5, 0.5, 0, 0, 0))


def test_split_gives_each_question_the_state_alone() -> None:
    from den.prompt import split, text

    record = _record()
    rows = split(record)
    assert [r.questions for r in rows] == [(q,) for q in record.questions]
    assert all(r.state == record.state for r in rows)
    assert "Is it late?" not in text(rows[0]) and "Which team?" not in text(rows[1])  # no question sees another


def test_save_merged_folds_lora_into_the_base_layout(tmp_path: Path) -> None:
    import json

    import peft
    from safetensors.torch import load_file, save_file

    from den.model import save_merged

    class Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 3, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attn()

    class Text(torch.nn.Module):  # the text-only class's names: model.layers.N...
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

    class Causal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Text()

    torch.manual_seed(0)
    base_model: Any = Causal()  # attribute access through nn.Module is untyped
    base = tmp_path / "base"  # the checkpoint is the vision-language layout: model.language_model.layers.N...
    base.mkdir()
    weights = {
        f"model.language_model.layers.{i}.self_attn.q_proj.weight": base_model.model.layers[i]
        .self_attn.q_proj.weight.detach()
        .clone()
        for i in range(2)
    }
    weights["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(2, 2)
    save_file(weights, base / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (base / "tokenizer.json").write_text("{}")
    (base / ".gitattributes").write_text("*.safetensors filter=lfs\n")

    adapted = peft.get_peft_model(base_model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    lora: Any
    for lora in adapted.modules():
        if hasattr(lora, "lora_B"):
            torch.nn.init.normal_(lora.lora_B["default"].weight)  # lora_B starts at zero; make the update real
    x = torch.randn(5, 4)
    expected = [base_model.model.layers[i].self_attn.q_proj(x) for i in range(2)]  # wrapped in place by peft

    assert save_merged(adapted, base, tmp_path / "merged") == 2
    merged = load_file(tmp_path / "merged" / "model.safetensors")
    for i in range(2):
        got = x @ merged[f"model.language_model.layers.{i}.self_attn.q_proj.weight"].T
        assert torch.allclose(got, expected[i].detach(), atol=1e-5)
    assert torch.equal(
        merged["model.visual.blocks.0.attn.qkv.weight"], weights["model.visual.blocks.0.attn.qkv.weight"]
    )
    assert json.loads((tmp_path / "merged" / "config.json").read_text()) == {"model_type": "qwen3_5"}
    assert (tmp_path / "merged" / "tokenizer.json").is_file()
    assert not (tmp_path / "merged" / ".gitattributes").exists()


def test_evaluation_perturbations_keep_labels_true() -> None:
    from den.prompt import NONE_OPTIONS, perturb

    nones = {k for k, _ in NONE_OPTIONS}
    record = _choice()
    for seed in range(20):
        (replaced,) = perturb(record, random.Random(seed), "none-replace").questions
        assert "c" not in replaced.keys and replaced.keys[replaced.label] in nones and len(replaced.keys) == 5
        (added,) = perturb(record, random.Random(seed), "none-add").questions
        assert added.keys[added.label] == "c" and len(added.keys) == 6 and any(k in nones for k in added.keys)
        (distracted,) = perturb(record, random.Random(seed), "distract").questions
        assert distracted.keys[distracted.label] == "c" and len(distracted.keys) == 6
        assert not any(k in nones for k in distracted.keys)


def test_letters_layout_marks_the_same_positions(tmp_path: Path) -> None:
    from den.prompt import encode, text

    if not TOKENIZER.is_file():
        pytest.skip("qwen3.5-4b tokenizer not downloaded")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    record = _choice(4, "b")
    assert "A) a\nB) b\nC) c\nD) d\nAnswer:" in text(record, "letters")
    example = encode(record, tokenizer, style="letters")
    assert example is not None and tokenizer.decode([example.ids[example.finals[0]]]) == ":"
    assert encode(_choice(8, "a"), tokenizer, style="letters") is not None
    many = parse(
        {
            "state": "s",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": {f"o{i}": None for i in range(27)},
                    "label": "o0",
                }
            },
        },
        "r",
    )
    assert encode(many, tokenizer, style="letters") is None  # 27 options: no 27th letter


@pytest.mark.model
@pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="qwen3.5-4b not downloaded")
def test_merged_export_answers_like_the_adapter(tmp_path: Path) -> None:
    """Behavioral merge equivalence on the real 4B weights: base + LoRA (torch, the training path) and the exported
    merged/ + head (loaded from a copied directory through the serving path) give the same probabilities and the same
    predictions, and repeated reads are identical. Tolerance: 0.02 in probability, for bf16 arithmetic on two
    different engines (torch CPU vs MLX/torch)."""
    import gc
    import shutil

    from den.evaluate import Model
    from den.model import LoraConfig, load_backbone, save_head, save_merged, text_tower

    torch.manual_seed(0)
    model = load_backbone(MODEL, LoraConfig(rank=8, alpha=16), 256, 0, engine="peft")
    for m in model.modules():
        if hasattr(m, "lora_B"):
            torch.nn.init.normal_(m.lora_B["default"].weight, std=0.02)  # a real update, not the zero init
    head = PointerHead(2560, HeadConfig(dim=64, heads=2))
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.02)
    head.eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    requests = [
        _record(),
        parse(
            {
                "state": "Rated it 4 out of 5.",
                "questions": {
                    "s": {
                        "type": "score",
                        "instructions": "Stars?",
                        "criteria": ["one", "two", "three", "four", "five"],
                        "label": 3,
                    }
                },
            },
            "s",
        ),
    ]
    want = []
    with torch.no_grad():
        for record in requests:
            for one in split(record):
                example = encode(one, tokenizer)
                assert example is not None
                hidden = text_tower(model)(input_ids=torch.tensor([example.ids]), use_cache=False).last_hidden_state
                want.append(head(hidden, [example])[0][0].float())
    run = tmp_path / "train-workspace"
    save_merged(model, MODEL, run / "merged")
    save_head(run, head, base="qwen3.5-4b", lora=8, temperature=1.0)
    del model
    gc.collect()
    exported = tmp_path / "somewhere-else"
    shutil.copytree(run, exported)
    shutil.rmtree(run)  # nothing may point back at the training directory
    served = Model(exported)
    got = [s for record in requests for s in served.read(record)[0]]  # T = 1: the logits themselves
    again = [s for record in requests for s in served.read(record)[0]]
    assert len(got) == len(want) == 3
    logit_gap = max(float((a - b).abs().max()) for a, b in zip(got, want, strict=True))
    prob_gap = max(float((a.softmax(-1) - b.softmax(-1)).abs().max()) for a, b in zip(got, want, strict=True))
    print(f"\nmerge: max |logit diff| {logit_gap:.5f}  max |prob diff| {prob_gap:.5f}  "
          f"argmax equal {[int(a.argmax()) == int(b.argmax()) for a, b in zip(got, want, strict=True)]}")  # fmt: skip
    assert prob_gap < 0.02 and all(int(a.argmax()) == int(b.argmax()) for a, b in zip(got, want, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(got, again, strict=True))


def test_fit_temperature_survives_flat_and_extreme_logits() -> None:
    """Regression: LBFGS's strong-Wolfe line search divided by zero when the loss went flat at the bound, as it
    does for zero-shot LM logits. The bounded search must return a finite T in range, and the true optimum inside."""
    torch.manual_seed(0)
    logits = torch.randn(500, 5) * 40  # LM-sized logits, labels unrelated: the best T sits at the upper bound
    labels = torch.randint(0, 5, (500,)).tolist()
    t = fit_temperature(list(logits), labels, [None] * 500)
    assert math.isfinite(t) and 19.9 <= t <= 20.0 + 1e-6
    same = torch.zeros(10, 3)  # every option tied: any T is optimal, and it must still return one
    assert 0.05 <= fit_temperature(list(same), [0] * 10, [None] * 10) <= 20
    ragged = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 3.0, 0.0, 0.0])]  # padded options never count
    assert 0.05 <= fit_temperature(ragged, [0, 1], [None, None]) <= 20


def test_fit_temperature_finds_the_minimum() -> None:
    from den.calibrate import calibration_nll

    torch.manual_seed(1)
    logits = torch.randn(2000, 4) * 2
    labels = torch.multinomial(logits.softmax(-1), 1).squeeze(1).tolist()
    t = fit_temperature(list(logits * 2.5), labels, [None] * 2000)
    goal = torch.nn.functional.one_hot(torch.tensor(labels), 4).float()
    here = calibration_nll(logits * 2.5, goal, math.log(t))
    assert all(here <= calibration_nll(logits * 2.5, goal, math.log(t) + d) + 1e-9 for d in (-0.01, 0.01))
    assert 2.2 < t < 2.8


def test_run_metadata_helpers() -> None:
    from den.train import dataset_hash, pinned, runtime

    used: list[dict[str, object]] = [{"path": "train/b.jsonl", "sha256": "2", "lines": [3, 1]},
                                     {"path": "train/a.jsonl", "sha256": "1", "lines": "all"}]  # fmt: skip
    assert dataset_hash(used) == dataset_hash(list(reversed(used)))  # order of listing doesn't matter
    assert dataset_hash(used) != dataset_hash([{**used[0], "lines": [3]}, used[1]])  # the lines taken do
    assert pinned("qwen3.5-4b") == {"key": "qwen3.5-4b", "repo": "Qwen/Qwen3.5-4B-Base",
                                    "revision": "1001bb4d826a52d1f399e183466143f4da7b741b"}  # fmt: skip
    assert {"platform", "python", "cuda", "cudnn", "driver", "gpu_count"} <= set(runtime())


def test_max_licence_keeps_only_records_the_licence_allows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from den import train

    question = {"answer": {"type": "noul", "instructions": "Ok?", "label": True}}
    rows = [{"state": f"s{i}", "questions": question, "_meta": {"source": s}} for i, s in enumerate(
        ("banking77", "yelp", "dbpedia14", "agnews")
    )] + [{"state": "no source", "questions": question}]  # fmt: skip
    path = tmp_path / "train" / "mixed.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(train, "CLEAN", tmp_path)
    assert train.licensed(path, None) == [1, 2, 3, 4, 5]
    assert train.licensed(path, "open") == [1]
    assert train.licensed(path, "share-alike") == [1, 3]
    assert train.licensed(path, "restricted") == [1, 2, 3, 4, 5]  # a record naming no source counts as unspecified
    used: list[dict[str, object]] = []
    kept = train.sample(path, 99, 0, used, "non-commercial")
    assert [r.state for r in kept] == ["s0", "s2", "s3"] and used[0]["lines"] == [1, 3, 4]
    assert train.licence_counts(used) == {"open": 1, "share-alike": 1, "non-commercial": 1}
    assert train.licence_counts([{"path": "train/mixed.jsonl", "lines": "all"}]) == {
        "open": 1, "share-alike": 1, "non-commercial": 1, "unspecified": 1, "restricted": 1
    }  # fmt: skip


def test_new_training_flags_default_to_kevs_recipe() -> None:
    from den.train import parser

    args = parser().parse_args([])
    assert (args.lora, args.lora_alpha, args.rslora, args.max_licence, args.report_to) == (16, 0, False, None, ["none"])
