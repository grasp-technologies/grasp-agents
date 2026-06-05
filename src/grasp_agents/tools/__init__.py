"""
Built-in tool packages for grasp-agents.

- :mod:`.file_edit` — ``Read`` / ``Write`` / ``Edit`` / ``Delete`` file tools.
- :mod:`.file_search` — ``Glob`` / ``Grep`` read-only search tools.
- :class:`FileToolkit` — one factory bundling all of the above.
- :class:`Bash` — run a shell command (fresh process) via ``ctx.exec_backend``.
- :class:`BashSession` — run a command in a persistent shell session.

These are imported lazily (PEP 562) so importing :mod:`grasp_agents.tools`
doesn't pull in the file tools (and their ripgrep-availability checks) or the
exec stack eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bash import (
        Bash,
        BashInput,
        BashOutput,
        BashResult,
        KillBash,
        bash_tools,
    )
    from .bash_session import BashSession
    from .file_toolkit import FileToolkit


_LAZY: dict[str, str] = {
    "Bash": "bash",
    "BashInput": "bash",
    "BashOutput": "bash",
    "BashResult": "bash",
    "KillBash": "bash",
    "bash_tools": "bash",
    "FileToolkit": "file_toolkit",
    "BashSession": "bash_session",
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
    "Bash",
    "BashInput",
    "BashOutput",
    "BashResult",
    "BashSession",
    "FileToolkit",
    "KillBash",
    "bash_tools",
]
