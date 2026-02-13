from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

import pytest

import pytrickle.runtime_args as runtime_args


def _reload_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    argv: Sequence[str],
    env_value: Optional[str] = None,
) -> object:
    """Reload runtime_args with controlled argv/env state."""
    monkeypatch.setattr(sys, "argv", list(argv))

    if env_value is None:
        monkeypatch.delenv("PYTRICKLE_PORT", raising=False)
    else:
        monkeypatch.setenv("PYTRICKLE_PORT", env_value)

    return importlib.reload(runtime_args)


def test_env_override_applies_without_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_runtime(monkeypatch, argv=["prog"], env_value="8123")

    assert mod.resolve_port(8000) == 8123
    assert mod.port_override_source() == "env"


def test_cli_override_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_runtime(monkeypatch, argv=["prog", "--port", "9005"])

    assert mod.resolve_port(8000) == 8000
    assert mod.port_override_source() is None


def test_cli_override_activates_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_runtime(monkeypatch, argv=["prog", "--foo", "bar", "--port", "9001"])

    mod.enable_cli_port_override()

    assert mod.resolve_port(8000) == 9001
    assert mod.port_override_source() == "cli"


def test_cli_override_supports_equals_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_runtime(monkeypatch, argv=["prog", "--port=9100", "--other"])

    mod.enable_cli_port_override()
    assert mod.resolve_port(8000) == 9100


def test_cli_override_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_runtime(monkeypatch, argv=["prog", "--port", "not-a-number"])

    with pytest.raises(SystemExit):
        mod.enable_cli_port_override()


def test_repo_examples_trigger_auto_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    example_path = repo_root / "examples" / "loading_overlay_example.py"

    mod = _reload_runtime(monkeypatch, argv=["prog", "--port", "7777"])

    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(example_path))

    assert mod.resolve_port(8000) == 7777

