"""den: System One decision models. `den.load(run).predict(request)` answers a /v1/systemone request."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import Model


def load(run: str, backend: str | None = None) -> Model:
    """A trained run, from a directory, `hf:<org>/<name>[@<revision>]` or `release:<version>`."""
    from .paths import locate
    from .runtime import Model

    return Model(locate(run), backend)
