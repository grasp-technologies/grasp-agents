"""
File-edit tool package: ``Read`` / ``Write`` / ``Edit`` / ``Delete`` +
``NotebookRead`` / ``NotebookEdit``, plus their per-session bookkeeping
(:class:`FileEditSessionState`), fuzzy-match chain, and secret redaction.

These operate over a :class:`~grasp_agents.file_backend.FileBackend`
(the I/O substrate, now in the sibling :mod:`..file_backend` package) wired onto
:attr:`SessionContext.file_backend`. To bundle them with the search tools (``Glob``
/ ``Grep``), use :class:`grasp_agents.tools.FileToolkit`.

Imports are lazy (PEP 562) to avoid pulling :mod:`tools.base` into
:class:`SessionContext` construction.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .delete import DeleteInput, DeleteResult, DeleteTool
    from .edit import EditInput, EditResult, EditTool
    from .fuzzy_match import (
        UNICODE_MAP,
        apply_replacements,
        fuzzy_find,
        fuzzy_find_and_replace,
        preserve_quote_style,
    )
    from .notebook import (
        NotebookCellView,
        NotebookEditInput,
        NotebookEditResult,
        NotebookEditTool,
        NotebookReadInput,
        NotebookReadResult,
        NotebookReadTool,
    )
    from .read import ReadInput, ReadResult, ReadTool
    from .read_image import ReadImageInput, ReadImageTool
    from .redact import DefaultSecretRedactor, NullRedactor, SecretRedactor
    from .session_state import FileEditSessionState, ReadRecord
    from .write import WriteInput, WriteResult, WriteTool


# Map each public attribute to the submodule that defines it. Single source of
# truth for ``__all__`` + the ``__getattr__`` lookup; keep them in sync.
_LAZY: dict[str, str] = {
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
    "NotebookCellView": "notebook",
    "NotebookEditInput": "notebook",
    "NotebookEditResult": "notebook",
    "NotebookEditTool": "notebook",
    "NotebookReadInput": "notebook",
    "NotebookReadResult": "notebook",
    "NotebookReadTool": "notebook",
    "ReadInput": "read",
    "ReadResult": "read",
    "ReadTool": "read",
    "ReadImageInput": "read_image",
    "ReadImageTool": "read_image",
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
    "FileEditSessionState",
    "NotebookCellView",
    "NotebookEditInput",
    "NotebookEditResult",
    "NotebookEditTool",
    "NotebookReadInput",
    "NotebookReadResult",
    "NotebookReadTool",
    "NullRedactor",
    "ReadImageInput",
    "ReadImageTool",
    "ReadInput",
    "ReadRecord",
    "ReadResult",
    "ReadTool",
    "SecretRedactor",
    "WriteInput",
    "WriteResult",
    "WriteTool",
    "apply_replacements",
    "fuzzy_find",
    "fuzzy_find_and_replace",
    "preserve_quote_style",
]
