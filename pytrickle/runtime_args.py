from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_PORT_FLAG = "--port"
_PORT_ENV_VAR = "PYTRICKLE_PORT"

_cli_override_enabled = False
_cli_port_override: Optional[int] = None
_env_port_override: Optional[int] = None
_override_source: Optional[str] = None


def _coerce_port(value: str) -> int:
    """Convert *value* to an integer port, raising on invalid input."""
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"PyTrickle: invalid port '{value}'. Provide an integer between 1 and 65535."
        ) from exc

    if not 0 < port < 65536:
        raise SystemExit(
            f"PyTrickle: invalid port '{value}'. Provide an integer between 1 and 65535."
        )
    return port


def _parse_cli_port(args: Sequence[str]) -> Optional[int]:
    """
    Extract the last --port occurrence from *args* (values after '--' are ignored).
    Returns None if flag not present.
    """
    port: Optional[int] = None
    i = 0
    args_list: List[str] = list(args)

    while i < len(args_list):
        token = args_list[i]

        if token == "--":
            break

        if token == _PORT_FLAG:
            if i + 1 >= len(args_list):
                raise SystemExit("PyTrickle: --port flag requires an integer value.")
            port = _coerce_port(args_list[i + 1])
            i += 2
            continue

        if token.startswith(f"{_PORT_FLAG}="):
            port = _coerce_port(token.split("=", 1)[1])
            i += 1
            continue

        i += 1

    return port


def _load_env_override() -> Optional[int]:
    """Read PYTRICKLE_PORT from the environment if present."""
    value = os.getenv(_PORT_ENV_VAR)
    if value is None:
        return None
    port = _coerce_port(value)
    return port


_env_port_override = _load_env_override()
if _env_port_override is not None:
    _override_source = "env"


def enable_cli_port_override(args: Optional[Sequence[str]] = None) -> None:
    """
    Enable parsing of the process argv for --port.

    Args:
        args: Optional iterable of strings to parse instead of sys.argv[1:].
    """
    global _cli_override_enabled
    global _cli_port_override
    global _override_source

    if _cli_override_enabled:
        return

    _cli_override_enabled = True

    parsed_args = args if args is not None else sys.argv[1:]
    port = _parse_cli_port(parsed_args)
    if port is not None:
        _cli_port_override = port
        _override_source = "cli"

def resolve_port(port: Optional[int] = None) -> int:
    """
    Return the effective listening port after applying CLI/env overrides.

    Args:
        port: Port requested by caller.

    Raises:
        ValueError: if no explicit port is provided and no override is enabled.
    """
    _auto_enable_cli_override_for_examples()

    if _cli_override_enabled and _cli_port_override is not None:
        return _cli_port_override

    if _env_port_override is not None:
        return _env_port_override

    if port is None:
        raise ValueError("PyTrickle: port must be provided when no CLI or env override is configured.")

    return port



def port_override_source() -> Optional[str]:
    """Return 'cli', 'env', or None depending on how the port is overridden."""
    return _override_source


def _auto_enable_cli_override_for_examples() -> None:
    """Enable CLI overrides automatically when running bundled examples."""
    if _cli_override_enabled:
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    main_path = getattr(main_module, "__file__", None)
    if not main_path:
        return

    try:
        main_file = Path(main_path).resolve()
    except OSError:
        return

    pkg_dir = Path(__file__).resolve().parent

    candidate_dirs = [
        pkg_dir.parent / "examples",
        pkg_dir / "examples",
    ]

    for candidate in candidate_dirs:
        try:
            candidate_path = candidate.resolve()
        except OSError:
            continue

        if not candidate_path.exists():
            continue

        try:
            main_file.relative_to(candidate_path)
        except ValueError:
            continue

        enable_cli_port_override()
        return

