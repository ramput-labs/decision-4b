"""The backbone and the head as one module: `forward` returns the loss, so a Hugging Face `Trainer` drives it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .head import question_loss
from .lora import text_tower
from .prompt import Example


class SystemOne(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, ordinal: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.ordinal = ordinal  # weight of the ranked probability score on score questions

    def scores(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, examples: Sequence[Example]
    ) -> list[list[torch.Tensor]]:
        tower = text_tower(self.backbone)
        hidden = tower(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        out: list[list[torch.Tensor]] = self.head(hidden, examples)
        return out

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, examples: Sequence[Example]
    ) -> dict[str, torch.Tensor]:
        losses = [
            question_loss(s, label, target, self.ordinal if ordered else 0.0)
            for example, per in zip(examples, self.scores(input_ids, attention_mask, examples), strict=True)
            for s, label, target, ordered in zip(
                per, example.labels, example.targets, example.ordered or [False] * len(per), strict=True
            )
        ]
        return {"loss": torch.stack(losses).mean()}


def collate(examples: Sequence[Example], pad: int = 0) -> dict[str, Any]:
    """Right-padded ids and mask, plus the examples themselves: the head needs their token positions."""
    width = max(len(e.ids) for e in examples)
    ids = torch.full((len(examples), width), pad, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        mask[i, : len(e.ids)] = 1
    return {"input_ids": ids, "attention_mask": mask, "examples": list(examples)}
