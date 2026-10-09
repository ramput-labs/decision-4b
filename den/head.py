"""The heads that turn hidden states into one logit per option.

Kev's head scores option i as q(question's final token) . k(option i's closing token). Under causal attention an
option never sees the options after it, so the `set` head adds three things, each a no-op at initialisation so
training starts from Kev's scoring:

1. Span pooling: an option's key also attends over its own tokens.
2. Option mixing: permutation-equivariant self-attention over [question, options] (no position encoding).
3. A per-option prior read from the option's mixed vector, so base rates come from text, not position.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional

from .paths import read_json
from .prompt import Example

type HeadKind = Literal["set", "pointer", "letters"]
type OptionRep = Literal["end", "marker", "mean", "attn"]
type Projection = Literal["linear", "mlp"]

HEAD_FORMAT = "den-pointer-head"


@dataclass(frozen=True, slots=True)
class HeadConfig:
    """kind: `set` (ours), `pointer` (Kev's q . k alone) or `letters` (Qwen's own logits for " A".." Z", no weights).
    rep, an option's key: `end` its closing token, `marker` its `-`, `mean` its tokens' mean, `attn` the closing
    token plus attention pooling over its tokens. Default: `attn` for set, `end` for pointer."""

    kind: HeadKind = "set"
    dim: int = 256
    layers: int = 1  # option-mixing blocks (set only)
    heads: int = 4
    dropout: float = 0.0
    rep: OptionRep | None = None
    proj: Projection = "linear"

    @property
    def option_rep(self) -> OptionRep:
        return self.rep or ("end" if self.kind == "pointer" else "attn")

    @property
    def shape(self) -> tuple[object, ...]:
        """What makes two heads interchangeable: everything but dropout."""
        return (self.kind, self.dim, self.layers, self.heads, self.option_rep, self.proj)


def _projection(hidden: int, dim: int, kind: Projection) -> nn.Module:
    if kind == "linear":
        return nn.Linear(hidden, dim)
    return nn.Sequential(nn.Linear(hidden, dim), nn.GELU(), nn.Linear(dim, dim))


def _positions(batch: Sequence[Example]) -> tuple[list[int], list[int], list[list[tuple[int, int]]]]:
    """Flattened over every question in the batch: its row, its final token, and each option's (start, close)."""
    rows, finals, spans = [], [], []
    for b, example in enumerate(batch):
        for final, first, last in zip(example.finals, example.starts, example.closes, strict=True):
            rows.append(b)
            finals.append(final)
            spans.append(list(zip(first, last, strict=True)))
    return rows, finals, spans


def _unflatten(scores: torch.Tensor, batch: Sequence[Example]) -> list[list[torch.Tensor]]:
    """Per example, per question: its own options' scores. Padded option slots are cut off here."""
    out: list[list[torch.Tensor]] = []
    i = 0
    for example in batch:
        out.append([scores[i + j, : len(c)] for j, c in enumerate(example.closes)])
        i += len(example.closes)
    return out


def _zero(*layers: nn.Linear) -> None:
    for layer in layers:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)


class PointerHead(nn.Module):
    """K options in, K logits out, for any K."""

    def __init__(self, hidden: int, config: HeadConfig | None = None) -> None:
        super().__init__()
        config = config or HeadConfig()
        if config.kind == "letters":
            raise ValueError("letters is LetterHead; build heads with make_head")
        dim = config.dim
        self.config = config
        self.hidden = hidden
        self.rep = config.option_rep
        self.norm = nn.RMSNorm(hidden)
        self.q = _projection(hidden, dim, config.proj)
        self.k = _projection(hidden, dim, config.proj)
        self.scale = 1 / math.sqrt(dim)
        self.set = config.kind == "set"
        if self.rep == "attn":
            self.v = nn.Linear(hidden, dim)  # span values, added to the closing token's key
            self.pool = nn.Linear(hidden, 1)  # span attention logits
            _zero(self.v)
        if not self.set:
            return
        blocks = [
            nn.TransformerEncoderLayer(
                dim, config.heads, 2 * dim, config.dropout, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(config.layers)
        ]
        for block in blocks:  # residual branches start at zero: each block begins as the identity
            _zero(block.self_attn.out_proj, block.linear2)
        self.mix = nn.ModuleList(blocks)
        self.prior = nn.Linear(dim, 1)
        _zero(self.prior)

    def forward(self, hidden: torch.Tensor, batch: Sequence[Example]) -> list[list[torch.Tensor]]:
        """hidden (B, T, H) -> per example, per question: a (K,) tensor of option logits."""
        rows, finals, spans = _positions(batch)
        n_options = max(len(s) for s in spans)
        n_tokens = max(c - s + 1 for span in spans for s, c in span)
        index = torch.zeros(len(spans), n_options, n_tokens, dtype=torch.long)  # (Q, K, L): each option's tokens
        tokens = torch.zeros(len(spans), n_options, n_tokens, dtype=torch.bool)  # which of those are real
        for i, span in enumerate(spans):
            for j, (start, close) in enumerate(span):
                index[i, j, : close - start + 1] = torch.arange(start, close + 1)
                tokens[i, j, : close - start + 1] = True
        device = hidden.device
        index, tokens = index.to(device), tokens.to(device)
        options = tokens.any(-1)  # (Q, K): real options; the rest is padding
        row = torch.tensor(rows, device=device)
        count = tokens.sum(-1, keepdim=True)
        closes = index.gather(-1, (count - 1).clamp(min=0)).squeeze(-1)  # (Q, K)

        h = self.norm(hidden.float())  # (B, T, H)
        ask = self.q(h[row, torch.tensor(finals, device=device)])  # (Q, D)
        if self.rep == "end":
            key = self.k(h[row[:, None], closes])  # (Q, K, D)
        elif self.rep == "marker":
            key = self.k(h[row[:, None], index[..., 0]])
        elif self.rep == "mean":
            projected = self.k(h)[row[:, None, None], index]  # (Q, K, L, D)
            key = (projected * tokens[..., None]).sum(2) / count.clamp(min=1)
        else:
            key = self.k(h[row[:, None], closes])
            logits = self.pool(h).squeeze(-1)[row[:, None, None], index]
            # the fill must be finite in the logits' own dtype (fp32's min overflows bf16 under autocast)
            weight = logits.masked_fill(~tokens, torch.finfo(logits.dtype).min)
            key = key + (weight.softmax(-1)[..., None] * self.v(h)[row[:, None, None], index]).sum(2)
        if self.set:
            x = torch.cat([ask[:, None], key], 1)
            pad = torch.cat([torch.zeros_like(options[:, :1]), ~options], 1)
            for block in self.mix:
                x = block(x, src_key_padding_mask=pad)
            ask, key = x[:, 0], x[:, 1:]
        scores = (key @ ask[..., None]).squeeze(-1) * self.scale  # (Q, K)
        if self.set:
            scores = scores + self.prior(key).squeeze(-1)
        return _unflatten(scores, batch)


class LetterHead(nn.Module):
    """The text-generation baseline: option i's logit is h(final) . E[" A" + i], E the tied embedding rows (a fixed
    buffer, so the run is self-contained). No trainable parameters."""

    def __init__(self, hidden: int, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig(kind="letters")
        self.hidden = hidden
        self.register_buffer("letters", torch.zeros(26, hidden))

    def set_letters(self, rows: torch.Tensor) -> None:
        self.letters.copy_(rows.detach().float())  # type: ignore[operator]

    def forward(self, hidden: torch.Tensor, batch: Sequence[Example]) -> list[list[torch.Tensor]]:
        rows, finals, _ = _positions(batch)
        device = hidden.device
        h = hidden[torch.tensor(rows, device=device), torch.tensor(finals, device=device)].float()  # (Q, H)
        letters: torch.Tensor = self.letters  # type: ignore[assignment]
        return _unflatten(h @ letters.to(device).T, batch)


def make_head(hidden: int, config: HeadConfig) -> nn.Module:
    return LetterHead(hidden, config) if config.kind == "letters" else PointerHead(hidden, config)


def question_loss(
    scores: torch.Tensor, label: int, target: Sequence[float] | None, ordinal: float = 0.0
) -> torch.Tensor:
    """Cross-entropy (against the soft target when there is one). `ordinal` > 0 adds Kev's ranked probability score,
    so a near-miss score level costs less than a far one."""
    logp = functional.log_softmax(scores, dim=-1)
    if target is not None:
        return -(torch.tensor(target, device=scores.device) * logp).sum()
    loss = -logp[label]
    if ordinal and len(scores) > 1:
        cdf = logp.exp().cumsum(-1)[:-1]
        observed = (torch.arange(len(scores) - 1, device=scores.device) >= label).to(cdf.dtype)
        loss = loss + ordinal * (cdf - observed).square().mean()
    return loss


def save_head(out: Path, head: nn.Module, **meta: object) -> None:
    """`head.safetensors` (weights) + `head.json` (config and `meta`). No pickle."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, out / "head.safetensors")
    c: HeadConfig = head.config  # type: ignore[assignment]
    config = {"format": HEAD_FORMAT, "kind": c.kind, "dim": c.dim, "layers": c.layers, "heads": c.heads,
              "dropout": c.dropout, "rep": c.option_rep, "proj": c.proj, "hidden_size": head.hidden,
              "prompt": "letters" if c.kind == "letters" else "dash", **meta}  # fmt: skip
    (out / "head.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_head(run: Path, hidden: int | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """The head saved in `run`, and its head.json. Refuses a head built for another hidden size than `hidden`."""
    from safetensors.torch import load_file

    meta = read_json(run / "head.json")
    if meta.get("format") != HEAD_FORMAT:
        raise SystemExit(f"{run}/head.json is missing or not a den head")
    if hidden is not None and meta["hidden_size"] != hidden:
        raise SystemExit(f"{run}: the head expects hidden size {meta['hidden_size']}, the backbone has {hidden}")
    config = HeadConfig(*(meta[k] for k in ("kind", "dim", "layers", "heads", "dropout", "rep", "proj")))
    head = make_head(meta["hidden_size"], config)
    head.load_state_dict(load_file(run / "head.safetensors"))
    return head, meta
