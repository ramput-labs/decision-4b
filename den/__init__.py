"""den: System One decision models. `den.load(run).predict(request)` answers a /v1/systemone request."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluate import Model


def load(run: str, backend: str | None = None) -> Model:
    """A trained run, from a directory or `hf:<org>/<name>[@<revision>]`, ready to `.predict(request)`."""
    from .evaluate import Model, locate

    return Model(locate(run), backend)
