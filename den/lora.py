"""LoRA on the Qwen3.5 text tower: Unsloth on CUDA (training), plain PEFT anywhere (checks).

Following Unsloth's Qwen3.5 guide: bf16 weights (no QLoRA), dropout 0 for the fused kernels, gradient checkpointing
"unsloth". The targets add Gated DeltaNet's projections to the usual seven, since 24 of the 32 layers are linear
attention. No target matches the vision tower, and `load_backbone` checks that after wrapping.
"""

from __future__ import annotations

import importlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .head import load_head
from .paths import read_json

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")  # the 8 full-attention layers: 32 modules
MLP = ("gate_proj", "up_proj", "down_proj")  # all 32 layers: 96 modules
DELTANET = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")  # the 24 DeltaNet layers: 120
LORA_PRESETS = {"attention": ATTENTION, "attention-mlp": ATTENTION + MLP, "all": ATTENTION + MLP + DELTANET}
LORA_TARGETS = LORA_PRESETS["all"]  # 248 modules on Qwen3.5-4B

type Engine = Literal["unsloth", "peft"]


@dataclass(frozen=True, slots=True)
class LoraConfig:
    rank: int = 16  # 0: no adapter, frozen backbone
    alpha: int = 32
    dropout: float = 0.0  # Unsloth's fast path needs 0
    targets: tuple[str, ...] = LORA_TARGETS
    rslora: bool = False  # scale by alpha / sqrt(rank) instead of alpha / rank

    @property
    def scale(self) -> float:
        return self.alpha / (math.sqrt(self.rank) if self.rslora else self.rank) if self.rank else 0.0


def load_backbone(path: Path, lora: LoraConfig, max_seq: int, seed: int, engine: Engine = "unsloth") -> Any:  # noqa: ANN401 (untyped peft wrapper)
    """The base in bf16 with fresh adapters (frozen when `lora.rank` is 0). Tokenization is `prompt.encode`'s, so the
    engine's tokenizer (a processor, for Qwen3.5) is dropped."""
    if engine == "unsloth":
        try:
            unsloth = importlib.import_module("unsloth")  # before transformers and peft, so its patches apply
        except ImportError as e:
            raise SystemExit("unsloth is not installed: it needs Linux with an NVIDIA GPU (`make setup-gpu`)") from e
        model, _ = unsloth.FastLanguageModel.from_pretrained(
            model_name=str(path), max_seq_length=max_seq, load_in_4bit=False, load_in_16bit=True, full_finetuning=False
        )
        if lora.rank:
            model = unsloth.FastLanguageModel.get_peft_model(
                model,
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=list(lora.targets),
                bias="none",
                use_rslora=lora.rslora,
                use_gradient_checkpointing="unsloth",
                random_state=seed,
                max_seq_length=max_seq,
            )
    else:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        # the checkpoint's own class, so adapter names and merged weights keep the base's layout (mlx-lm can't read
        # the `qwen3_5_text` layout a text-only class would save)
        auto = (
            AutoModelForImageTextToText if "vision_config" in read_json(path / "config.json") else AutoModelForCausalLM
        )
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
                use_rslora=lora.rslora,
            )
            model = peft.get_peft_model(model, config)
    if not lora.rank:
        model.requires_grad_(False)
    elif stray := [name for name in adapted(model) if ".visual." in name]:
        raise SystemExit(f"LoRA reached the vision tower: {stray[:3]}")
    return model


def adapted(model: nn.Module) -> list[str]:
    """Names of the modules that carry a LoRA adapter."""
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def text_tower(model: object) -> nn.Module:
    """The text decoder without the LM head: Qwen3_5TextModel, the module the serving backends run, never the
    vision-language wrapper around it."""
    inner: Any = model.get_base_model() if hasattr(model, "get_base_model") else model
    tower: nn.Module = getattr(inner.base_model, "language_model", inner.base_model)
    return tower


def merged_weights(backbone: nn.Module) -> dict[str, torch.Tensor]:
    """Every adapted weight with its LoRA folded in (W + scaling * B @ A), keyed from `layers.` on, so the key doesn't
    depend on the model class or engine."""
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


def _keys(shard: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(shard, "pt") as handle:
        return list(handle.keys())


def save_merged(backbone: nn.Module, base: Path, out: Path) -> int:
    """A copy of the base checkpoint's files with every adapted text weight replaced, so `merged/` has the base's
    layout by construction. Returns the number of weights replaced; refuses unless all were."""
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


def warm_start(backbone: Any, head: nn.Module, run: Path, base: str, lora: LoraConfig) -> float:  # noqa: ANN401 (untyped peft wrapper)
    """Load a finished run's adapter and head into fresh ones of the same shape (Kev's `--init_from`). Returns the
    run's temperature."""
    previous, meta = load_head(run, int(head.hidden))  # type: ignore[arg-type]
    theirs = LoraConfig(meta["lora"], meta["lora_alpha"], rslora=meta["rslora"])
    mismatch = [
        f"{name} {a!r} != {b!r}"
        for name, a, b in (
            ("base", meta["base"], base),
            ("lora rank", theirs.rank, lora.rank),
            ("lora scale (alpha, rsLoRA)", theirs.scale, lora.scale),
            ("head", previous.config.shape, head.config.shape),
        )
        if a != b
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
