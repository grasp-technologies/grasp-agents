"""
File-edit tool package.

Provides the ``Read`` / ``Write`` / ``Edit`` / ``Delete`` primitives.
Read-before-write bookkeeping lives on the *agent* (each
:class:`AgentLoop` owns its own :class:`FileEditSessionState`); the
:class:`FileBackend` itself is pure I/O. Default backend is
:class:`LocalFileBackend` (host filesystem); :class:`MCPFileBackend`
routes the same calls to an MCP server speaking the file-tool protocol.

To bundle these with the search tools (``Glob`` / ``Grep``), use
:class:`grasp_agents.tools.FileToolkit`.

Imports are lazy (PEP 562) to avoid pulling :mod:`types.tool` into
:class:`RunContext` construction.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_state import (
        get_current_file_edit_state,
        reset_current_file_edit_state,
        set_current_file_edit_state,
    )
    from .atomic_write import atomic_write_bytes, atomic_write_text
    from .backend import FileBackend, FileEntry, FileStat
    from .delete import DeleteInput, DeleteResult, DeleteTool
    from .edit import EditInput, EditResult, EditTool
    from .fuzzy_match import (
        UNICODE_MAP,
        apply_replacements,
        fuzzy_find,
        fuzzy_find_and_replace,
        preserve_quote_style,
    )
    from .local_backend import LocalFileBackend
    from .mcp_backend import MCPFileBackend
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
    from .write import WriteInput, WriteResult, WriteTool


# Map each public attribute to the submodule that defines it. The list
# is the single source of truth for ``__all__`` + the ``__getattr__``
# lookup; keep them in sync.
_LAZY: dict[str, str] = {
    "get_current_file_edit_state": "agent_state",
    "reset_current_file_edit_state": "agent_state",
    "set_current_file_edit_state": "agent_state",
    "atomic_write_bytes": "atomic_write",
    "atomic_write_text": "atomic_write",
    "FileBackend": "backend",
    "FileEntry": "backend",
    "FileStat": "backend",
    "LocalFileBackend": "local_backend",
    "DeleteInput": "delete",
    "DeleteResult": "delete",
    "DeleteTool": "delete",
    "EditInput": "edit",
    "EditResult": "edit",
    "EditTool": "edit",
    "UNICODE_MAP": "fuzzy_match",
    "apply_replacements": "fuzzy_match",
    "fuzzy_find": "fuzzy_match",
    "fuzzy_find_and_replace": "fuzzy_match",
    "preserve_quote_style": "fuzzy_match",
    "MCPFileBackend": "mcp_backend",
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
    "DeleteInput",
    "DeleteResult",
    "DeleteTool",
    "EditInput",
    "EditResult",
    "EditTool",
    "FileBackend",
    "FileEditSessionState",
    "FileEntry",
    "FileStat",
    "LocalFileBackend",
    "MCPFileBackend",
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
    "get_current_file_edit_state",
    "has_binary_extension",
    "is_blocked_device",
    "preserve_quote_style",
    "reset_current_file_edit_state",
    "resolve_safe",
    "set_current_file_edit_state",
]
