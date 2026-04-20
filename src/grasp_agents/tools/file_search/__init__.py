"""
File-search tool package.

Provides read-only ``Glob`` (path-pattern match) and ``Grep`` (ripgrep-
backed content search) tools, bundled via :class:`FileSearchToolkit`.
Sits next to :mod:`..file_edit` and shares its ``allowed_roots`` /
sandbox convention.

Usage::

    from grasp_agents.tools.file_search import FileSearchToolkit

    toolkit = FileSearchToolkit(allowed_roots=[Path.cwd()])
    agent.tools = [*toolkit.tools(), ...]

Imports are lazy (PEP 562) so constructing an agent without these tools
does not trigger rg-availability checks at import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .glob import GlobInput, GlobResult, GlobTool
    from .grep import GrepError, GrepInput, GrepResult, GrepTool, rg_available
    from .toolkit import FileSearchToolkit


_LAZY: dict[str, str] = {
    "GlobInput": "glob",
    "GlobResult": "glob",
    "GlobTool": "glob",
    "GrepError": "grep",
    "GrepInput": "grep",
    "GrepResult": "grep",
    "GrepTool": "grep",
    "rg_available": "grep",
    "FileSearchToolkit": "toolkit",
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
    "FileSearchToolkit",
    "GlobInput",
    "GlobResult",
    "GlobTool",
    "GrepError",
    "GrepInput",
    "GrepResult",
    "GrepTool",
    "rg_available",
]
