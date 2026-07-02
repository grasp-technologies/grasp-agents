"""
File-I/O substrate for the file tools.

The :class:`FileBackend` contract (``base``) plus its standalone implementations
(:class:`LocalFileBackend`, :class:`MCPFileBackend`), path policy (``paths``),
and atomic writes (``atomic_write``). This is the layer the file *tools*
(:mod:`..file_edit` / :mod:`..file_search`) and ``SessionContext`` build on; it does
not depend on the tools themselves.

(The E2B file backend lives in :mod:`grasp_agents.sandbox.e2b` instead — it is
inseparable from the live E2B sandbox handle, so it ships with the exec
environment rather than here.)

Imports are lazy (PEP 562) so importing this package — which ``SessionContext`` does
at construction — does not pull in the MCP client (an optional extra) via
:class:`MCPFileBackend`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .atomic_write import atomic_write_bytes, atomic_write_text
    from .base import FileBackend, FileEntry, FileStat
    from .local import LocalFileBackend
    from .mcp import MCPFileBackend
    from .paths import (
        PathAccessError,
        SensitivePathRules,
        check_sensitive_path,
        has_binary_extension,
        is_blocked_device,
        resolve_safe,
        sensitive_path_rules,
    )


_LAZY: dict[str, str] = {
    "atomic_write_bytes": "atomic_write",
    "atomic_write_text": "atomic_write",
    "FileBackend": "base",
    "FileEntry": "base",
    "FileStat": "base",
    "LocalFileBackend": "local",
    "MCPFileBackend": "mcp",
    "PathAccessError": "paths",
    "SensitivePathRules": "paths",
    "check_sensitive_path": "paths",
    "has_binary_extension": "paths",
    "is_blocked_device": "paths",
    "resolve_safe": "paths",
    "sensitive_path_rules": "paths",
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
    "FileBackend",
    "FileEntry",
    "FileStat",
    "LocalFileBackend",
    "MCPFileBackend",
    "PathAccessError",
    "SensitivePathRules",
    "atomic_write_bytes",
    "atomic_write_text",
    "check_sensitive_path",
    "has_binary_extension",
    "is_blocked_device",
    "resolve_safe",
    "sensitive_path_rules",
]
