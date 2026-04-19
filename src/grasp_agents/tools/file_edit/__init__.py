"""
File-edit tool package.

Provides ``Read`` / ``Write`` / ``Edit`` primitives and a
:class:`FileEditToolkit` factory that owns per-session state. Not
auto-installed on any agent — consumers explicitly attach the tools
they want.

Usage::

    from grasp_agents.tools.file_edit import FileEditToolkit

    toolkit = FileEditToolkit(allowed_roots=[Path.cwd()])
    agent.tools = [*toolkit.tools(), ...]

Imports are lazy (PEP 562): accessing the full tool set does not trigger
``types.tool`` during ``RunContext`` construction, so ``FileEditStore``
can live in this package without creating an import cycle.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .atomic_write import atomic_write_bytes, atomic_write_text
    from .edit import EditInput, EditResult, EditTool
    from .fuzzy_match import (
        UNICODE_MAP,
        apply_replacements,
        fuzzy_find,
        fuzzy_find_and_replace,
        preserve_quote_style,
    )
    from .paths import (
        PathAccessError,
        check_sensitive_path,
        has_binary_extension,
        is_blocked_device,
        resolve_safe,
    )
    from .read import ReadInput, ReadResult, ReadTool
    from .redact import DefaultSecretRedactor, NullRedactor, SecretRedactor
    from .session_state import FileEditSessionState, ReadRecord
    from .store import FileEditStore, InMemoryFileEditStore
    from .toolkit import FileEditToolkit
    from .write import WriteInput, WriteResult, WriteTool


# Map each public attribute to the submodule that defines it. The list
# is the single source of truth for ``__all__`` + the ``__getattr__``
# lookup; keep them in sync.
_LAZY: dict[str, str] = {
    "atomic_write_bytes": "atomic_write",
    "atomic_write_text": "atomic_write",
    "EditInput": "edit",
    "EditResult": "edit",
    "EditTool": "edit",
    "UNICODE_MAP": "fuzzy_match",
    "apply_replacements": "fuzzy_match",
    "fuzzy_find": "fuzzy_match",
    "fuzzy_find_and_replace": "fuzzy_match",
    "preserve_quote_style": "fuzzy_match",
    "PathAccessError": "paths",
    "check_sensitive_path": "paths",
    "has_binary_extension": "paths",
    "is_blocked_device": "paths",
    "resolve_safe": "paths",
    "ReadInput": "read",
    "ReadResult": "read",
    "ReadTool": "read",
    "DefaultSecretRedactor": "redact",
    "NullRedactor": "redact",
    "SecretRedactor": "redact",
    "FileEditSessionState": "session_state",
    "ReadRecord": "session_state",
    "FileEditStore": "store",
    "InMemoryFileEditStore": "store",
    "FileEditToolkit": "toolkit",
    "WriteInput": "write",
    "WriteResult": "write",
    "WriteTool": "write",
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
    "UNICODE_MAP",
    "DefaultSecretRedactor",
    "EditInput",
    "EditResult",
    "EditTool",
    "FileEditSessionState",
    "FileEditStore",
    "FileEditToolkit",
    "InMemoryFileEditStore",
    "NullRedactor",
    "PathAccessError",
    "ReadInput",
    "ReadRecord",
    "ReadResult",
    "ReadTool",
    "SecretRedactor",
    "WriteInput",
    "WriteResult",
    "WriteTool",
    "apply_replacements",
    "atomic_write_bytes",
    "atomic_write_text",
    "check_sensitive_path",
    "fuzzy_find",
    "fuzzy_find_and_replace",
    "has_binary_extension",
    "is_blocked_device",
    "preserve_quote_style",
    "resolve_safe",
]
