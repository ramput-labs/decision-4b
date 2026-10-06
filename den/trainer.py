"""The Hugging Face `Trainer` as den uses it (Unsloth patches it on CUDA).

Three things differ from the stock Trainer, nothing else:
- batches come from length groups (`lengths`), so a short record isn't padded to the length of a long document;
- the optimizer is `model.param_groups`: the head at its own learning rate, and exactly the trainable parameters;
- a checkpoint holds only what trains (the LoRA adapters and the head, `trainable.safetensors`), not the frozen
  4B base. The Trainer still writes and restores the optimizer, scheduler, RNG and step state beside it, so
  `--resume` continues a run where it stopped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

from .model import param_groups

TRAINABLE = "trainable.safetensors"


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

    saved = load_file(directory / TRAINABLE)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(saved) != set(params):
        missing, extra = sorted(set(params) - set(saved))[:3], sorted(set(saved) - set(params))[:3]
        raise SystemExit(f"{directory} is a checkpoint of another model (missing {missing}, unexpected {extra})")
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
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:  # noqa: ANN401
        """A checkpoint's model part: the trainable parameters only."""
        assert self.model is not None
        save_trainable(self.model, Path(output_dir or str(self.args.output_dir)))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:  # noqa: ANN401
        target = model if model is not None else self.model
        assert isinstance(target, torch.nn.Module)
        load_trainable(target, Path(resume_from_checkpoint))
