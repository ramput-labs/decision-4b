"""LoRA (Unsloth on CUDA, PEFT elsewhere) on the Qwen3.5 text tower, and a set-aware pointer head over each
question's options.

Kev's head scores option i as q(question final token) . k(option i closing token). Under causal attention option i
never sees options i+1.., so its key cannot depend on the alternatives that follow it. This head keeps that pointer
and adds three things, each a no-op at initialisation so training starts from Kev's scoring:

1. Span pooling: an option's key also attends over its own tokens, not only the newline that closes it.
2. Option mixing: `layers` permutation-equivariant self-attention blocks over [question, option 1..K] (no position
   encoding), so every option is scored against all the others and the scores do not depend on option order.
3. A per-option prior, read from the option's mixed vector, so base rates are learned from option text, not position.

Scores are logits; one temperature fitted on the calibration split (calibrate.py) turns them into probabilities.

LoRA follows Unsloth's Qwen3.5 guide: bf16 weights (QLoRA is not recommended for Qwen3.5), dropout 0 so Unsloth's
fused LoRA kernels apply, gradient checkpointing "unsloth". The targets add Gated DeltaNet's projections to the usual
seven, because 24 of Qwen3.5-4B's 32 layers are linear attention and would otherwise adapt only their MLP. No target
name matches the vision tower (`attn.qkv`, `attn.proj`, `linear_fc*`), and `adapted()` checks that after wrapping.
`rank=0` loads the backbone frozen with no adapter: the head-only baseline.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional

from .prompt import Example

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")  # the 8 full-attention layers of Qwen3.5-4B: 32 modules
MLP = ("gate_proj", "up_proj", "down_proj")  # all 32 layers: 96 modules
DELTANET = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")  # the 24 Gated DeltaNet layers: 120
LORA_PRESETS = {"attention": ATTENTION, "attention-mlp": ATTENTION + MLP, "all": ATTENTION + MLP + DELTANET}
LORA_TARGETS = LORA_PRESETS["all"]  # every linear layer of the text decoder: 248 modules on Qwen3.5-4B


type Engine = Literal["unsloth", "peft"]
type HeadKind = Literal["set", "pointer", "letters"]
type OptionRep = Literal["end", "marker", "mean", "attn"]
type Projection = Literal["linear", "mlp"]


@dataclass(frozen=True, slots=True)
class LoraConfig:
    rank: int = 16  # 0: no adapter, frozen backbone
    alpha: int = 32
    dropout: float = 0.0  # Unsloth's fast LoRA path needs 0
    targets: tuple[str, ...] = LORA_TARGETS


@dataclass(frozen=True, slots=True)
class HeadConfig:
    """The head is independent of the backbone: any kind works on any backbone, given its hidden size.

    kind:
      `set` (default): q(question's final token) . key(option) / sqrt(dim), plus `layers` cross-option attention
        blocks and a per-option prior.
      `pointer`: Kev's head, q . key / sqrt(dim) alone.
      `letters`: no learned head. Qwen's own next-token logits for " A", " B", ... after `Answer:` (the tied
        embedding rows), over the `A) option` prompt: the text-generation baseline, trainable only through LoRA.
    rep, how an option becomes a key: `end` its closing token; `marker` its `-` token; `mean` the mean over its
        tokens; `attn` its closing token plus learned attention pooling over its tokens. Default: `attn` for set,
        `end` for pointer.
    proj: `linear` or `mlp` (two layers, GELU) for the question and option projections."""

    kind: HeadKind = "set"
    dim: int = 256
    layers: int = 1  # option-mixing blocks (set only); 0 keeps span pooling and the prior
    heads: int = 4
    dropout: float = 0.0
    rep: OptionRep | None = None
    proj: Projection = "linear"

    @property
    def option_rep(self) -> OptionRep:
        return self.rep or ("end" if self.kind == "pointer" else "attn")


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
    """Per example, per question: exactly its own options' scores. Padded option slots are cut off here, so they can
    never enter a softmax, a loss or a prediction."""
    out: list[list[torch.Tensor]] = []
    i = 0
    for example in batch:
        out.append([scores[i + j, : len(c)] for j, c in enumerate(example.closes)])
        i += len(example.closes)
    return out


class PointerHead(nn.Module):
    """Scores each of a question's K options with one logit: K options in, K logits out, for any K."""

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
            nn.init.zeros_(self.v.weight)
            nn.init.zeros_(self.v.bias)
        if not self.set:
            return
        blocks = [
            nn.TransformerEncoderLayer(
                dim, config.heads, 2 * dim, config.dropout, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(config.layers)
        ]
        for block in blocks:  # residual branches start at zero: the blocks begin as the identity
            nn.init.zeros_(block.self_attn.out_proj.weight)
            nn.init.zeros_(block.self_attn.out_proj.bias)
            nn.init.zeros_(block.linear2.weight)
            nn.init.zeros_(block.linear2.bias)
        self.mix = nn.ModuleList(blocks)
        self.prior = nn.Linear(dim, 1)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def forward(self, hidden: torch.Tensor, batch: Sequence[Example]) -> list[list[torch.Tensor]]:
        """hidden (B, T, H) -> per example, per question: a (K,) float32 tensor of option logits."""
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
        else:  # attn: the closing token, plus attention pooling over the option's tokens
            key = self.k(h[row[:, None], closes])
            logits = self.pool(h).squeeze(-1)[row[:, None, None], index]  # bf16 under autocast
            # finite and in the logits' own dtype (fp32's min overflows bf16): an all-padding slot gets uniform
            # weights, never NaN
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
    """The text-generation baseline as a head: option i's logit is Qwen's next-token logit for its letter after
    `Answer:`, h(final) . E[" A" + i], with E the tied input/output embedding rows (stored here as a fixed buffer, so
    the run is self-contained). No trainable parameters: with --lora 0 it is zero-shot Qwen; with LoRA, Qwen fine-tuned
    to answer with the letter."""

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
        scores = h @ letters.to(device).T  # (Q, 26); _unflatten keeps each question's first K
        return _unflatten(scores, batch)


def make_head(hidden: int, config: HeadConfig) -> nn.Module:
    return LetterHead(hidden, config) if config.kind == "letters" else PointerHead(hidden, config)


def question_loss(
    scores: torch.Tensor, label: int, target: Sequence[float] | None, ordinal: float = 0.0
) -> torch.Tensor:
    """Cross-entropy, against the soft target when the question has one. `ordinal` > 0 (score questions only) adds
    Kev's ranked probability score: the squared gap between predicted and true cumulative distributions, so a
    near-miss level costs less than a far one."""
    logp = functional.log_softmax(scores, dim=-1)
    if target is not None:
        return -(torch.tensor(target, device=scores.device) * logp).sum()
    loss = -logp[label]
    if ordinal and len(scores) > 1:
        cdf = logp.exp().cumsum(-1)[:-1]
        observed = (torch.arange(len(scores) - 1, device=scores.device) >= label).to(cdf.dtype)
        loss = loss + ordinal * (cdf - observed).square().mean()
    return loss


HEAD_FORMAT = "den-pointer-head"


def save_head(out: Path, head: nn.Module, **meta: object) -> None:
    """The head as Hugging Face-style files: weights in `head.safetensors`, config and metadata in `head.json`. No
    pickle; loads with `load_head` against any backbone of the same hidden size."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, out / "head.safetensors")
    c: HeadConfig = head.config  # type: ignore[assignment]
    config = {"format": HEAD_FORMAT, "version": 2, "kind": c.kind, "dim": c.dim, "layers": c.layers, "heads": c.heads,
              "dropout": c.dropout, "rep": c.option_rep, "proj": c.proj, "hidden_size": head.hidden,
              "prompt": "letters" if c.kind == "letters" else "dash", **meta}  # fmt: skip
    (out / "head.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_head(run: Path, hidden: int | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """The head saved in `run`, and its head.json. Refuses a head built for another hidden size than `hidden`."""
    from safetensors.torch import load_file

    meta = _config_file(run / "head.json")
    if meta.get("format") != HEAD_FORMAT:
        raise SystemExit(f"{run}/head.json is not a den head")
    if hidden is not None and meta["hidden_size"] != hidden:
        raise SystemExit(f"{run}: the head expects hidden size {meta['hidden_size']}, the backbone has {hidden}")
    config = HeadConfig(
        meta["kind"],
        meta["dim"],
        meta["layers"],
        meta["heads"],
        meta["dropout"],
        meta.get("rep"),
        meta.get("proj", "linear"),
    )
    head = make_head(meta["hidden_size"], config)
    head.load_state_dict(load_file(run / "head.safetensors"))
    return head, meta


def load_backbone(path: Path, lora: LoraConfig, max_seq: int, seed: int, engine: Engine = "unsloth") -> Any:  # noqa: ANN401 (untyped peft wrapper)
    """The base in bf16, with fresh LoRA adapters unless `lora.rank` is 0. Tokenization is ours (`prompt.encode` on
    the checkpoint's tokenizer.json), so the engine's tokenizer, a processor for Qwen3.5, isn't kept.

    `unsloth` needs Linux + an NVIDIA GPU (`make setup-gpu`) and is what training uses. `peft` loads the same
    adapters with plain transformers + PEFT, on any device, to check the wiring without a GPU."""
    if engine == "unsloth":
        try:
            unsloth = importlib.import_module("unsloth")  # before transformers and peft, so its patches apply
        except ImportError as e:
            raise SystemExit("unsloth is not installed: it needs Linux with an NVIDIA GPU (`make setup-gpu`)") from e
        model, _ = unsloth.FastLanguageModel.from_pretrained(
            model_name=str(path),
            max_seq_length=max_seq,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        if lora.rank:
            model = unsloth.FastLanguageModel.get_peft_model(
                model,
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=list(lora.targets),
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=seed,
                max_seq_length=max_seq,
            )
    else:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        # the checkpoint's own class, so adapter names and merged weights keep the base's layout and every backbone
        # loads them; a text-only class would save `qwen3_5_text`, which mlx-lm cannot read
        auto = AutoModelForImageTextToText if "vision_config" in _config(path) else AutoModelForCausalLM
        model = auto.from_pretrained(path, dtype=torch.bfloat16)
        if lora.rank:
            peft = importlib.import_module("peft")
            torch.manual_seed(seed)
            config = peft.LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=list(lora.targets),
                bias="none",
            )
            model = peft.get_peft_model(model, config)
    if not lora.rank:
        model.requires_grad_(False)
    elif stray := [name for name in adapted(model) if ".visual." in name]:
        raise SystemExit(f"LoRA reached the vision tower: {stray[:3]}")
    return model


def merged_weights(backbone: nn.Module) -> dict[str, torch.Tensor]:
    """Every adapted linear's weight with its LoRA folded in, W + scaling * B @ A, keyed from `layers.` on
    (e.g. `layers.3.self_attn.q_proj.weight`), so the key doesn't depend on which model class or engine trained it."""
    out: dict[str, torch.Tensor] = {}
    for name, module in backbone.named_modules():
        lora_a, lora_b = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or "default" not in lora_a:
            continue
        base = module.base_layer.weight
        delta = lora_b["default"].weight.float() @ lora_a["default"].weight.float()
        weight = base.float() + module.scaling["default"] * delta
        out[name[name.index("layers.") :] + ".weight"] = weight.to(base.dtype).cpu()
    return out


def save_merged(backbone: nn.Module, base: Path, out: Path) -> int:
    """The LoRA folded into the base's own checkpoint: the base's files copied as they are (config, tokenizer,
    vision tower), with every adapted text weight replaced. The result has the base's layout by construction, so
    every backbone that loads the base loads it, whichever engine or model class did the training. Returns the
    number of weights replaced, which must equal the number of adapted modules."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    merged = merged_weights(backbone)
    shards = sorted(base.glob("*.safetensors"))
    names = [k for shard in shards for k in _keys(shard)]
    prefix = "model.language_model." if any(k.startswith("model.language_model.layers.") for k in names) else "model."
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):  # not the Hub's .gitattributes
            shutil.copy2(f, out / f.name)
    replaced = 0
    for shard in shards:
        tensors = {}
        with safe_open(shard, "pt") as handle:
            for key in _keys(shard):
                tensor = handle.get_tensor(key)
                if key.startswith(prefix) and (new := merged.get(key.removeprefix(prefix))) is not None:
                    if new.shape != tensor.shape:
                        raise SystemExit(
                            f"merge: {key} is {tuple(tensor.shape)} in the base, {tuple(new.shape)} merged"
                        )
                    tensor, replaced = new.to(tensor.dtype), replaced + 1
                tensors[key] = tensor
        save_file(tensors, out / shard.name, metadata={"format": "pt"})
    if replaced != len(merged):
        raise SystemExit(
            f"merge: replaced {replaced} of {len(merged)} adapted weights; the adapter in {out.parent} is fine"
        )
    return replaced


def _keys(shard: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(shard, "pt") as handle:
        keys: list[str] = list(handle.keys())
    return keys


def _config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _config(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return loaded


def _shape(c: HeadConfig) -> tuple[object, ...]:
    """What makes two heads interchangeable: everything but dropout, with the option representation resolved."""
    return (c.kind, c.dim, c.layers, c.heads, c.option_rep, c.proj)


def warm_start(backbone: Any, head: nn.Module, run: Path, base: str, lora: LoraConfig) -> float:  # noqa: ANN401 (untyped peft wrapper)
    """Load a finished run's adapter and pointer head into fresh ones of the same shape, to continue training from it
    (Kev's `--init_from`, used for every stage after the first). Returns the run's temperature, for the record."""
    previous, meta = load_head(run, int(head.hidden))  # type: ignore[arg-type]
    mismatch = [
        f"{name} {theirs!r} != {ours!r}"
        for name, theirs, ours in (
            ("base", meta.get("base"), base),
            ("lora rank", meta.get("lora"), lora.rank),
            ("head", _shape(previous.config), _shape(head.config)),  # type: ignore[arg-type]
        )
        if theirs != ours
    ]
    if mismatch:
        raise SystemExit(f"--init-from {run} does not match this run: {'; '.join(mismatch)}")
    head.load_state_dict(previous.state_dict())
    if lora.rank:
        from safetensors.torch import load_file

        peft = importlib.import_module("peft")
        result = peft.set_peft_model_state_dict(backbone, load_file(run / "adapter_model.safetensors"))
        if result.unexpected_keys or [k for k in result.missing_keys if "lora_" in k]:
            raise SystemExit(f"--init-from {run}: its adapter does not fit this model's LoRA modules")
    return float(meta["temperature"])


def adapted(model: nn.Module) -> list[str]:
    """Names of the modules that carry a LoRA adapter."""
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def text_tower(model: object) -> nn.Module:
    """The text decoder without the LM head (8k tokens x 248k vocab logits would not fit; the head never reads them).

    The same module the serving backends run: Qwen3_5TextModel, never the vision-language wrapper around it, which
    holds the vision tower and builds M-RoPE indices."""
    inner: Any = model.get_base_model() if hasattr(model, "get_base_model") else model
    tower: nn.Module = getattr(inner.base_model, "language_model", inner.base_model)
    return tower


class SystemOne(nn.Module):
    """The backbone and the pointer head as one module. `forward` returns the training loss, so a Hugging Face
    `Trainer` (which Unsloth patches) drives the optimisation; `scores` is the same pass without the loss."""

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


def param_groups(
    model: nn.Module, lr: float, head_lr: float, weight_decay: float, decay: set[str]
) -> list[dict[str, Any]]:
    """AdamW groups: the head (`head.*`) at `head_lr`, everything else trainable (the LoRA adapters) at `lr`; no weight
    decay outside `decay` (norms, biases). Refuses to leave out a trainable parameter, or to take a frozen one."""
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            rate = head_lr if name.startswith("head.") else lr
            groups.setdefault((rate, weight_decay if name in decay else 0.0), []).append(p)
    out: list[dict[str, Any]] = [{"params": ps, "lr": rate, "weight_decay": wd} for (rate, wd), ps in groups.items()]
    grouped = {id(p) for g in out for p in g["params"]}
    if grouped != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("the optimizer's parameters differ from the model's trainable ones")
    return out
