"""The training wiring on a tiny real PEFT model: which parameters train in each mode, that gradients reach them,
that the optimizer holds exactly them, and that the head scores exactly the options it is given."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import peft
import pytest
import torch
from torch import nn

from den.model import (
    HeadConfig,
    LetterHead,
    PointerHead,
    SystemOne,
    collate,
    make_head,
    param_groups,
    question_loss,
    text_tower,
)
from den.prompt import Example

HIDDEN = 16


class Tower(nn.Module):
    """A stand-in text decoder: embeddings, one attention-named and one MLP-named linear, a final norm."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)
        self.up_proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None, use_cache: bool = False) -> Any:  # noqa: ANN401
        h = self.embed_tokens(input_ids)
        h = h + torch.tanh(self.q_proj(h)).cumsum(1) / input_ids.shape[1]  # causal-ish mixing along the sequence
        return SimpleNamespace(last_hidden_state=self.norm(h + self.up_proj(h)))


class Causal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Tower()

    @property
    def base_model(self) -> nn.Module:
        return self.model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens


def example(n_options: int, label: int, offset: int = 0) -> Example:
    """A 40-token row: options start at token 2, each 3 tokens long; the final token is the last."""
    starts = tuple(2 + 3 * i for i in range(n_options))
    closes = tuple(s + 2 for s in starts)
    ids = tuple((offset + 7 * i) % 64 for i in range(40))
    return Example(ids, (39,), (closes,), (starts,), (label,), (None,), (False,), ("choice",), ("t",))


def build(mode: str) -> SystemOne:
    torch.manual_seed(0)
    lora = mode in ("lora+letters", "lora+pointer")
    backbone: nn.Module = Causal()
    if lora:
        backbone = peft.get_peft_model(
            backbone, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"])
        )
    else:
        backbone.requires_grad_(False)
    config = HeadConfig(kind="letters", dim=8) if mode.endswith("letters") else HeadConfig(kind="set", dim=8, heads=2)
    head = make_head(HIDDEN, config)
    if isinstance(head, LetterHead):
        head.set_letters(torch.randn(26, HIDDEN))
    return SystemOne(backbone, head)


def trainable(model: SystemOne) -> set[str]:
    return {n.split(".")[0] + (".lora" if "lora_" in n else "") for n, p in model.named_parameters() if p.requires_grad}


def test_the_four_modes_train_different_things() -> None:
    """A base Qwen (zero-shot letters), B LoRA + letters, C pointer head on frozen Qwen, D LoRA + pointer head."""
    assert trainable(build("base+letters")) == set()
    assert trainable(build("lora+letters")) == {"backbone.lora"}
    assert trainable(build("base+pointer")) == {"head"}
    assert trainable(build("lora+pointer")) == {"backbone.lora", "head"}


@pytest.mark.parametrize("mode", ["lora+letters", "base+pointer", "lora+pointer"])
def test_gradients_reach_every_trainable_part_and_the_optimizer_holds_them(mode: str) -> None:
    model = build(mode)
    for p in model.parameters():  # lora_B starts at zero, which zeroes lora_A's gradient; perturb it
        if p.requires_grad:
            nn.init.normal_(p, std=0.1)
    batch = collate([example(3, 1), example(5, 4, offset=3)])
    loss = model(batch["input_ids"], batch["attention_mask"], batch["examples"])["loss"]
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if "lora_" in name or name.startswith(("head.q", "head.k")):
                assert float(p.grad.abs().sum()) > 0, f"{name} got a zero gradient"
        else:
            assert p.grad is None, f"frozen {name} got a gradient"
    decay = {n for n, _ in model.named_parameters() if n.endswith("weight")}
    groups = param_groups(model, 2e-4, 1e-3, 0.01, decay)
    in_optimizer = {id(p) for g in groups for p in g["params"]}
    assert in_optimizer == {id(p) for p in model.parameters() if p.requires_grad}
    head_rate = {g["lr"] for g in groups if any(p is q for p in g["params"] for q in model.head.parameters())}
    assert head_rate in (set(), {1e-3})


def test_param_groups_refuses_to_drop_a_trainable_parameter() -> None:
    model = build("lora+pointer")
    model.extra = nn.Parameter(torch.ones(1))  # trainable but, by name, neither head nor adapter: still grouped
    assert sum(len(g["params"]) for g in param_groups(model, 1e-4, 1e-3, 0.0, set())) == sum(
        1 for p in model.parameters() if p.requires_grad
    )


def test_letter_head_is_the_lm_head_on_the_letter_rows() -> None:
    torch.manual_seed(0)
    model = build("base+letters")
    tower = text_tower(model.backbone)
    ex = example(4, 2)
    hidden = tower(input_ids=torch.tensor([ex.ids])).last_hidden_state
    (scores,) = model.head(hidden, [ex])[0]
    letters: torch.Tensor = model.head.letters  # type: ignore[assignment]
    assert scores.shape == (4,) and torch.allclose(scores, hidden[0, 39] @ letters[:4].T)


@pytest.mark.parametrize("kind", ["set", "pointer"])
@pytest.mark.parametrize("rep", ["end", "marker", "mean", "attn"])
@pytest.mark.parametrize("proj", ["linear", "mlp"])
def test_k_options_give_k_logits_in_mixed_batches(kind: str, rep: str, proj: str) -> None:
    torch.manual_seed(0)
    head = PointerHead(HIDDEN, HeadConfig(kind=kind, dim=8, heads=2, rep=rep, proj=proj))  # type: ignore[arg-type]
    for p in head.parameters():
        nn.init.normal_(p, std=0.2)
    sizes = [2, 3, 4, 5, 7]
    batch = [example(k, k - 1, offset=k) for k in sizes]
    hidden = torch.randn(len(batch), 40, HIDDEN, requires_grad=True)
    out = head(hidden, batch)
    assert [s.shape for (s,) in out] == [torch.Size([k]) for k in sizes]  # padded slots never come out
    loss = torch.stack([question_loss(s, e.labels[0], None) for (s,), e in zip(out, batch, strict=True)]).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    alone = [head(hidden[i : i + 1], [e])[0][0] for i, e in enumerate(batch)]  # batching changes nothing
    assert all(torch.allclose(a, b, atol=1e-5) for a, (b,) in zip(alone, out, strict=True))


@pytest.mark.parametrize("rep", ["end", "marker", "mean", "attn"])
def test_an_options_score_follows_its_own_tokens(rep: str) -> None:
    torch.manual_seed(0)
    head = PointerHead(HIDDEN, HeadConfig(kind="pointer", dim=8, rep=rep))  # type: ignore[arg-type]
    for p in head.parameters():
        nn.init.normal_(p, std=0.3)
    ex = example(3, 0)
    hidden = torch.randn(1, 40, HIDDEN)
    changed = hidden.clone()
    changed[0, 5:8] = torch.randn(3, HIDDEN)  # option 1's tokens (5, 6, 7) now read differently
    before, after = head(hidden, [ex])[0][0], head(changed, [ex])[0][0]
    assert not torch.isclose(before[1], after[1])  # its own score moves
    assert torch.allclose(before[[0, 2]], after[[0, 2]])  # pointer: the others don't (they never read option 1)


def _trainer(model: SystemOne, out: Any, data: list[Example], steps: int, save_steps: int = 0) -> Any:  # noqa: ANN401
    from transformers import TrainingArguments

    from den.trainer import PointerTrainer

    return PointerTrainer(
        lengths=[len(e.ids) for e in data],
        head_lr=5e-3,
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=2,
            max_steps=steps,
            learning_rate=5e-3,
            lr_scheduler_type="cosine",
            warmup_steps=0.2,
            save_strategy="steps" if save_steps else "no",
            save_steps=save_steps or 500,
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
            seed=0,
            logging_steps=1,
            disable_tqdm=True,
        ),
        train_dataset=data,
        data_collator=collate,
    )


def test_resume_continues_exactly_where_it_stopped(tmp_path: Any) -> None:  # noqa: ANN401
    """A run checkpointed at step 3 and resumed in a fresh process must end with the same adapter and head as the
    same run trained straight through: trainable weights, optimizer, scheduler, RNG and data position all restored."""
    from safetensors.torch import load_file

    from den.trainer import TRAINABLE, trainable_state

    data = [example(k, k - 1, offset=i) for i, k in enumerate([2, 3, 4, 5, 3, 2, 4, 5])]
    straight = build("lora+pointer")
    _trainer(straight, tmp_path / "straight", data, steps=6).train()

    interrupted = build("lora+pointer")
    _trainer(interrupted, tmp_path / "run", data, steps=6, save_steps=3).train()
    checkpoint = tmp_path / "run" / "checkpoint-3"
    saved = load_file(checkpoint / TRAINABLE)
    assert set(saved) == {n for n, p in interrupted.named_parameters() if p.requires_grad}  # nothing frozen saved
    assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()

    resumed = build("lora+pointer")  # a fresh model: everything trainable comes from the checkpoint
    _trainer(resumed, tmp_path / "run", data, steps=6, save_steps=3).train(resume_from_checkpoint=str(checkpoint))
    a, b = trainable_state(straight), trainable_state(resumed)
    assert set(a) == set(b) and all(torch.allclose(a[n], b[n], atol=1e-6) for n in a)
    assert any(not torch.equal(a[n], trainable_state(build("lora+pointer"))[n]) for n in a)  # it did train


def test_a_checkpoint_of_another_model_is_refused(tmp_path: Any) -> None:  # noqa: ANN401
    from den.trainer import load_trainable, save_trainable

    save_trainable(build("lora+pointer"), tmp_path)
    with pytest.raises(SystemExit, match="another model"):
        load_trainable(build("base+pointer"), tmp_path)


@pytest.mark.parametrize("kind", ["set", "pointer"])
@pytest.mark.parametrize("rep", ["end", "marker", "mean", "attn"])
def test_heads_run_under_bf16_autocast(kind: str, rep: str) -> None:
    """Training runs the head under BF16 autocast. Regression: the attn pooling filled padding with fp32's minimum,
    which overflows bf16 and raised inside masked_fill on the first GPU step. CPU autocast uses the same bf16."""
    torch.manual_seed(0)
    head = PointerHead(HIDDEN, HeadConfig(kind=kind, dim=8, heads=2, rep=rep))  # type: ignore[arg-type]
    batch = [example(k, k - 1, offset=k) for k in (2, 3, 5)]
    hidden = torch.randn(len(batch), 40, HIDDEN, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(hidden, batch)
        loss = torch.stack([question_loss(s, e.labels[0], None) for (s,), e in zip(out, batch, strict=True)]).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(loss) and [s.shape[0] for (s,) in out] == [2, 3, 5]
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    assert all(p.dtype == torch.float32 for p in head.parameters())  # fp32 master weights; autocast only computes


class Rows:
    """`train.Shuffled`'s interface over fixed examples."""

    def __init__(self, rows: list[Example]) -> None:
        self.rows, self.lengths = rows, [len(e.ids) for e in rows]
        self.longest = max(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, n: int) -> Example:
        return self.rows[n]


def test_a_run_saves_its_best_checkpoint_loss_log_and_calibration(tmp_path: Any, monkeypatch: Any) -> None:  # noqa: ANN401
    """`den train` end to end on the tiny model. Dev contradicts the training labels, so dev NLL rises as it trains:
    the best checkpoint is an early step, saved to best/ with its own T, while the final weights stay the run's."""
    import json

    from safetensors.torch import load_file

    from den import model as den_model
    from den import train as den_train

    torch.manual_seed(0)
    backbone = peft.get_peft_model(Causal(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"]))
    monkeypatch.setattr(den_model, "load_backbone", lambda *a, **k: backbone)
    monkeypatch.setattr(den_train, "hidden_size", lambda path: HIDDEN)
    data = [example(k, k - 1, offset=i) for i, k in enumerate([2, 3, 4, 5, 3, 2, 4, 5])]
    dev = [example(k, 0, offset=i) for i, k in enumerate([2, 3, 4, 5, 3, 2, 4, 5])]
    out = tmp_path / "run"
    args = den_train.parser().parse_args(
        ["--engine", "peft", "--lora", "4", "--head-dim", "8", "--head-heads", "2", "--max-steps", "6", "--batch",
         "2", "--accum", "1", "--lr", "5e-3", "--eval-steps", "1", "--eval-max", "8", "--out", str(out)]
    )  # fmt: skip
    den_train.train(args, tmp_path, Rows(data), dev, dev, [])  # type: ignore[arg-type]

    run = json.loads((out / "run.json").read_text())
    probes = [h for h in run["dev_history"] if "full" not in h]
    assert run["steps"] == 6 and [h["step"] for h in probes] == [1, 2, 3, 4, 5, 6]
    best = run["best"]
    assert best["step"] == min(probes, key=lambda h: h["nll"])["step"] < 6 and best["path"] == "best"
    assert best["on"] == "dev sample" and best["questions"] == 8 and "after" in best["calibration_report"]
    assert sorted(p.name for p in (out / "best").iterdir()) == [
        "README.md", "adapter_config.json", "adapter_model.safetensors", "head.json", "head.safetensors"
    ]  # fmt: skip
    assert json.loads((out / "best" / "head.json").read_text())["step"] == best["step"]
    final, early = load_file(out / "head.safetensors"), load_file(out / "best" / "head.safetensors")
    assert any(not torch.equal(final[k], early[k]) for k in final)  # the final weights came back after best/
    assert run["train_history"][-1]["step"] == 6 and "train_loss" in run["train_history"][-1]
    split = run["calibration_report"]["calibration_split"]
    assert split["after"]["nll"] <= split["before"]["nll"] + 1e-9  # T fitted on it can only lower its NLL
