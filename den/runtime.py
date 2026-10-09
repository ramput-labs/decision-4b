"""A trained run, ready to answer /v1/systemone requests in Kev's response shape.

A run directory holds `merged/` (the LoRA folded into bf16 weights) and `head.safetensors` + `head.json`. The
backbone runs on this machine's backend (MLX, CUDA or CPU, `device.load`); the head runs in PyTorch on the CPU.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from .api import Json, Question, Record, RecordError, parse
from .device import load
from .head import load_head
from .metrics import argmax
from .paths import base_weights, read_json
from .prompt import Style, encode, split

DEFAULT_NAME = "den-latest"
UNCAPPED = 1 << 30


class Model:
    """The backbone (merged weights, or the frozen base for a head-only run) and the head with its temperature."""

    def __init__(self, run: Path, backend: str | None = None) -> None:
        meta = read_json(run / "head.json")
        if not meta:
            raise SystemExit(f"{run} has no head.json: not a den run")
        weights = run / "merged"
        if not (weights / "config.json").is_file():
            if meta["lora"]:
                raise SystemExit(f"{weights} is missing: train with --merge, or download the run's merged/ folder")
            weights = base_weights(str(meta["base"]))
        self.backbone = load(weights, backend)  # type: ignore[arg-type]
        self.head, self.meta = load_head(run, self.backbone.hidden_size)
        self.head.eval()
        self.temperature = float(self.meta["temperature"])
        self.temperatures: dict[str, float] = self.meta.get("temperatures") or {}  # per question type, if fitted
        self.style: Style = "letters" if self.meta["kind"] == "letters" else "dash"
        self.tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))

    def predict(self, raw: dict[str, Json], name: str = DEFAULT_NAME) -> dict[str, Any]:
        """The /v1/systemone response for one request, as `den serve` returns it."""
        return respond(self, raw, name)

    def logits(self, record: Record) -> tuple[list[torch.Tensor], int]:
        """Uncalibrated logits for each question's options, and the tokens read. Each question is read with the
        state alone, as in training."""
        out, tokens = [], 0
        for one in split(record):
            example = encode(one, self.tokenizer, max_state=UNCAPPED, style=self.style)
            if example is None:
                raise RecordError(f"{one.questions[0].id}: the letter scorer reads at most 26 options")
            tokens += len(example.ids)
            hidden = torch.from_numpy(self.backbone.hidden_states(example.ids))[None]
            with torch.no_grad():
                out.append(self.head(hidden, [example])[0][0].float())
        return out, tokens

    def calibrate(self, record: Record, logits: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """logits / T, with the question type's own T when one was fitted."""
        return [
            s / self.temperatures.get(q.type, self.temperature) for q, s in zip(record.questions, logits, strict=True)
        ]

    def read(self, record: Record) -> tuple[list[torch.Tensor], int]:
        """Calibrated logits for each question's options, and the tokens read."""
        raw, tokens = self.logits(record)
        return self.calibrate(record, raw), tokens


def request(raw: dict[str, Json]) -> Record:
    """An unlabelled request as a Record: each question gets a placeholder label, which nothing reads."""
    questions = raw.get("questions")
    if not isinstance(questions, dict):
        raise SystemExit("request needs a questions object")
    filled: dict[str, Json] = {}
    for qid, q in questions.items():
        if isinstance(q, dict) and "label" not in q:
            criteria = q.get("criteria")
            placeholder: Json
            match q.get("type"):
                case "choice":
                    placeholder = next(iter(criteria), None) if isinstance(criteria, dict) else None
                case "score":
                    placeholder = 0
                case _:
                    placeholder = False
            q = {**q, "label": placeholder}
        filled[qid] = q
    return parse({**raw, "questions": filled}, "request")


def answer(q: Question, p: list[float]) -> dict[str, Any]:
    """One question's answer: `choice` (the likeliest option), `score` (the expected level) or `noul` (P(true)), with
    confidence = (p_max - 1/K) / (1 - 1/K)."""
    if q.type == "noul":
        return {"type": "noul", "noul": round(p[1], 4)}
    k, top = len(p), argmax(p)
    confidence = round((p[top] - 1 / k) / (1 - 1 / k), 4) if k > 1 else 1.0
    probabilities = {key: round(x, 4) for key, x in zip(q.keys, p, strict=True)}
    if q.type == "choice":
        return {"type": "choice", "choice": q.keys[top], "confidence": confidence, "probabilities": probabilities}
    return {
        "type": "score",
        "score": round(sum(i * x for i, x in enumerate(p)), 4),
        "confidence": confidence,
        "legend": dict(zip(q.keys, q.options, strict=True)),
        "probabilities": probabilities,
    }


def respond(model: Model, raw: dict[str, Json], name: str = DEFAULT_NAME) -> dict[str, Any]:
    """The /v1/systemone response body for one request."""
    started = time.perf_counter()
    record = request(raw)
    scores, tokens = model.read(record)
    answers = {q.id: answer(q, s.softmax(-1).tolist()) for q, s in zip(record.questions, scores, strict=True)}
    return {
        "model": str(raw.get("model") or name),
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "latency_ms": round(1000 * (time.perf_counter() - started)),
    }
