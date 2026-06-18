"""
File-search tool package.

Provides read-only ``Glob`` (path-pattern match) and ``Grep`` (ripgrep-
backed content search) tools. Sits next to :mod:`..file_edit` and shares
its ``allowed_roots`` / sandbox convention.

To bundle these with the edit tools (``Read`` / ``Write`` / ``Edit`` /
``Delete``), use :class:`grasp_agents.tools.FileToolkit`.

Imports are lazy (PEP 562) so constructing an agent without these tools
does not trigger rg-availability checks at import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .glob import GlobInput, GlobResult, GlobTool
    from .grep import GrepError, GrepInput, GrepResult, GrepTool, rg_available


_LAZY: dict[str, str] = {
    "GlobInput": "glob",
    "GlobResult": "glob",
    "GlobTool": "glob",
    "GrepError": "grep",
    "GrepInput": "grep",
    "GrepResult": "grep",
    "GrepTool": "grep",
    "rg_available": "grep",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for next access
    return attr


__all__ = [
    "GlobInput",
    "GlobResult",
    "GlobTool",
    "GrepError",
    "GrepInput",
    "GrepResult",
    "GrepTool",
    "rg_available",
]
