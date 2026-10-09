"""The Hugging Face `Trainer` (which Unsloth patches) with three changes: length-grouped batches, the head at its own
learning rate, and checkpoints of only the trainable parameters (the Trainer still saves optimizer, scheduler and RNG
state beside them, so `--resume` continues where a run stopped).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

TRAINABLE = "trainable.safetensors"


def param_groups(
    model: nn.Module, lr: float, head_lr: float, weight_decay: float, decay: set[str]
) -> list[dict[str, Any]]:
    """AdamW groups: the head (`head.*`) at `head_lr`, the adapters at `lr`; weight decay only on `decay`. Refuses to
    leave out a trainable parameter or take a frozen one."""
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            rate = head_lr if name.startswith("head.") else lr
            groups.setdefault((rate, weight_decay if name in decay else 0.0), []).append(p)
    out: list[dict[str, Any]] = [{"params": ps, "lr": rate, "weight_decay": wd} for (rate, wd), ps in groups.items()]
    if {id(p) for g in out for p in g["params"]} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("the optimizer's parameters differ from the model's trainable ones")
    return out


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter that trains, by name, on the CPU."""
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def save_trainable(model: torch.nn.Module, directory: Path) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), directory / TRAINABLE)


def load_trainable(model: torch.nn.Module, directory: Path) -> None:
    """Restore the trainable parameters saved by `save_trainable`; refuses a checkpoint of another model."""
    from safetensors.torch import load_file

    restore(model, load_file(directory / TRAINABLE), str(directory))


def restore(model: torch.nn.Module, saved: dict[str, torch.Tensor], what: str = "the state") -> None:
    """Copy a `trainable_state` back into the model's trainable parameters; refuses one of another model."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(saved) != set(params):
        missing, extra = sorted(set(params) - set(saved))[:3], sorted(set(saved) - set(params))[:3]
        raise SystemExit(f"{what} is a checkpoint of another model (missing {missing}, unexpected {extra})")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(saved[name].to(p.device, p.dtype))


class PointerTrainer(Trainer):
    def __init__(self, *args: Any, lengths: Sequence[int], head_lr: float, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.lengths = list(lengths)
        self.head_lr = head_lr

    def _get_train_sampler(self, train_dataset: object = None) -> LengthGroupedSampler:
        size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        return LengthGroupedSampler(size, lengths=self.lengths)

    def create_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        """Trainer's AdamW grouping (no decay on norms and biases), the head at its own learning rate, and exactly the
        model's trainable parameters (`param_groups` refuses anything else)."""
        if self.optimizer is None:
            this = self.model
            assert this is not None
            decay = set(self.get_decay_parameter_names(this))
            groups = param_groups(this, self.args.learning_rate, self.head_lr, self.args.weight_decay, decay)
            betas = (self.args.adam_beta1, self.args.adam_beta2)
            self.optimizer = torch.optim.AdamW(groups, betas=betas, eps=self.args.adam_epsilon)
        return self.optimizer

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:  # noqa: ANN401
        """A checkpoint's model part: the trainable parameters only."""
        assert self.model is not None
        save_trainable(self.model, Path(output_dir or str(self.args.output_dir)))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:  # noqa: ANN401
        target = model if model is not None else self.model
        assert isinstance(target, torch.nn.Module)
        load_trainable(target, Path(resume_from_checkpoint))
