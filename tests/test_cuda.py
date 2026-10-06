"""CUDA checks, run on the GPU box: `make test-cuda` (`pytest -m cuda`). Skipped where torch sees no CUDA device.

The toy tests check devices, dtypes, autocast and padding on CUDA in seconds; the `model`-marked one loads the real
Qwen3.5-4B through Unsloth, exactly as training does, and runs one forward/backward step of LoRA + pointer head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from den.model import HeadConfig, PointerHead, collate, param_groups, question_loss

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]
MODEL = Path("models/qwen3.5-4b")


def test_variable_k_padding_on_cuda() -> None:
    from test_wiring import example  # the same synthetic rows as the CPU wiring tests

    torch.manual_seed(0)
    head = PointerHead(16, HeadConfig(dim=8, heads=2)).cuda()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.2)
    batch = [example(k, k - 1, offset=k) for k in (2, 3, 4, 5, 7)]
    hidden = torch.randn(len(batch), 40, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = head(hidden, batch)
        loss = torch.stack([question_loss(s, e.labels[0], None) for (s,), e in zip(out, batch, strict=True)]).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert [s.shape[0] for (s,) in out] == [2, 3, 4, 5, 7] and all(s.device.type == "cuda" for (s,) in out)
    assert torch.isfinite(loss) and hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_toy_lora_and_head_train_on_cuda() -> None:
    from test_wiring import build, example

    model = build("lora+pointer").cuda()
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.1)
    batch = collate([example(3, 1), example(5, 4, offset=3)])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), batch["examples"])["loss"]
    loss.backward()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert all(p.device.type == "cuda" and p.dtype == torch.float32 for p in trainable.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert any(float(p.grad.abs().sum()) > 0 for n, p in trainable.items() if "lora_B" in n and p.grad is not None)
    groups = param_groups(model, 2e-4, 1e-3, 0.0, set())
    assert {id(p) for g in groups for p in g["params"]} == {id(p) for p in trainable.values()}


@pytest.mark.model
@pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="qwen3.5-4b not downloaded")
def test_unsloth_qwen_lora_pointer_step() -> None:
    """The real training path on the GPU: Unsloth loads Qwen3.5-4B in bf16, LoRA on all 248 text projections in fp32,
    the head on CUDA; one forward/backward over rows with 2, 3 and 5 options gives finite logits and loss, nonzero
    LoRA-B and head gradients, no gradient on the frozen base, and an optimizer step that moves both."""
    from tokenizers import Tokenizer

    from den.api import parse
    from den.model import LoraConfig, SystemOne, adapted, load_backbone, text_tower
    from den.prompt import encode

    backbone = load_backbone(MODEL, LoraConfig(), 1024, 0, engine="unsloth")
    modules = adapted(backbone)
    assert len(modules) == 248 and not any(".visual." in n for n in modules)
    assert type(text_tower(backbone)).__name__ == "Qwen3_5TextModel"
    model = SystemOne(backbone, PointerHead(2560, HeadConfig())).cuda()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert {p.dtype for p in trainable.values()} == {torch.float32}
    assert all(not p.requires_grad for n, p in backbone.named_parameters() if "lora_" not in n)
    assert any(p.dtype == torch.bfloat16 for n, p in backbone.named_parameters() if "lora_" not in n)

    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    rows = []
    for k, label in ((2, "b"), (3, "c"), (5, "a")):
        criteria = {c: None for c in "abcde"[:k]}
        raw = {"state": "The parcel arrived late and damaged.", "questions": {"q": {"type": "choice",
               "instructions": "Pick one", "criteria": criteria, "label": label}}}  # fmt: skip
        example = encode(parse(raw, f"r{k}"), tokenizer)  # type: ignore[arg-type]
        assert example is not None
        rows.append(example)
    batch = collate(rows)
    ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = model.scores(ids, mask, rows)
        loss = model(ids, mask, rows)["loss"]
    assert [len(s) for (s,) in scores] == [2, 3, 5] and all(torch.isfinite(s).all() for (s,) in scores)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert any(float(p.grad.abs().sum()) > 0 for n, p in trainable.items() if "lora_B" in n and p.grad is not None)
    assert any(
        float(p.grad.abs().sum()) > 0 for n, p in trainable.items() if n.startswith("head.") and p.grad is not None
    )
    assert all(p.grad is None for n, p in backbone.named_parameters() if "lora_" not in n)
    before = {n: p.detach().clone() for n, p in trainable.items() if "lora_B" in n or n.startswith("head.q")}
    optimizer = torch.optim.AdamW(param_groups(model, 2e-4, 1e-3, 0.0, set()))
    optimizer.step()
    assert any(not torch.equal(before[n], trainable[n]) for n in before if "lora_B" in n)
    assert any(not torch.equal(before[n], trainable[n]) for n in before if n.startswith("head."))
