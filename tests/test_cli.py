from __future__ import annotations

import importlib

import pytest

from den import cli


@pytest.mark.parametrize("name", sorted(cli.DELEGATED))
def test_every_delegated_command_resolves(name: str) -> None:
    """A command added to `cli.DELEGATED` must name a real function, and appear in `den --help`."""
    module, _, function = cli.DELEGATED[name][0].partition(":")
    assert callable(getattr(importlib.import_module(f"den.{module}"), function))
    assert f"den {name}" in (cli.__doc__ or "")


def test_delegated_help_reaches_the_commands_own_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["release", "--help"])
    assert "den release" in capsys.readouterr().out
