"""
``FileToolkit`` — one factory bundling every built-in file tool:
``Read`` / ``Write`` / ``Edit`` / ``Delete`` (from :mod:`.file_edit`) and
``Glob`` / ``Grep`` (from :mod:`.file_search`).

These belong together: an agent doing file-shaped work — memory authoring
especially — both *searches* the tree (Glob/Grep) and *reads/edits* files,
and ``Read`` is shared by both flows. All tools are stateless wrappers —
they consume :attr:`RunContext.file_backend` at run time and the active
:class:`FileEditSessionState` via the :mod:`.file_edit.agent_state`
ContextVar. The toolkit just bundles their per-tool configuration.

Usage::

    backend = LocalFileBackend(allowed_roots=[Path.cwd()])
    ctx = RunContext(state=..., file_backend=backend)
    agent = LLMAgent(..., tools=FileToolkit().tools())

For an agent that may explore but not mutate, use :meth:`read_only_tools`
(``Read`` + ``Glob`` + ``Grep``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .file_edit.delete import DeleteTool
from .file_edit.edit import EditTool
from .file_edit.read import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_READ_CHARS,
    ReadTool,
)
from .file_edit.redact import DefaultSecretRedactor, SecretRedactor
from .file_edit.write import DEFAULT_NEW_FILE_MODE, WriteTool
from .file_search.glob import DEFAULT_HEAD_LIMIT as DEFAULT_GLOB_HEAD_LIMIT
from .file_search.glob import GlobTool
from .file_search.grep import GrepTool

if TYPE_CHECKING:
    from ..types.tool import BaseTool


class FileToolkit:
    """Build a matching set of file tools sharing one configuration."""

    def __init__(
        self,
        *,
        redactor: SecretRedactor | None = None,
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        new_file_mode: int = DEFAULT_NEW_FILE_MODE,
        glob_head_limit: int = DEFAULT_GLOB_HEAD_LIMIT,
        glob_include_hidden: bool = False,
        tool_timeout: float | None = None,
    ) -> None:
        """
        Args:
            redactor: Secret-redaction strategy for ``Read`` output.
                Defaults to :class:`DefaultSecretRedactor`.
            max_read_chars: Character cap on a single ``Read`` window;
                past it the output is truncated with a notice. Default
                ``100_000``.
            max_file_bytes: Whole-file size ceiling for ``Read``; files
                larger than this are refused outright. Default ``10_000_000``.
            new_file_mode: Permissions for files created by ``Write``.
                Default ``0o644``.
            glob_head_limit: Cap on ``Glob`` results before truncation.
                Default 250.
            glob_include_hidden: Let ``Glob`` traverse hidden directories.
                Common build/cache dirs are always skipped regardless.
            tool_timeout: Per-tool async timeout in seconds. ``None``
                disables the timeout.

        """
        self._redactor: SecretRedactor = redactor or DefaultSecretRedactor()
        self._read_tool = ReadTool(
            redactor=self._redactor,
            max_read_chars=max_read_chars,
            max_file_bytes=max_file_bytes,
            timeout=tool_timeout,
        )
        self._write_tool = WriteTool(
            new_file_mode=new_file_mode,
            timeout=tool_timeout,
        )
        self._edit_tool = EditTool(timeout=tool_timeout)
        self._delete_tool = DeleteTool(timeout=tool_timeout)
        self._glob_tool = GlobTool(
            head_limit=glob_head_limit,
            include_hidden=glob_include_hidden,
            timeout=tool_timeout,
        )
        self._grep_tool = GrepTool(timeout=tool_timeout)

    # ---- Tool accessors ----------------------------------------------------

    @property
    def read(self) -> ReadTool:
        return self._read_tool

    @property
    def write(self) -> WriteTool:
        return self._write_tool

    @property
    def edit(self) -> EditTool:
        return self._edit_tool

    @property
    def delete(self) -> DeleteTool:
        return self._delete_tool

    @property
    def glob(self) -> GlobTool:
        return self._glob_tool

    @property
    def grep(self) -> GrepTool:
        return self._grep_tool

    def tools(self) -> list[BaseTool[Any, Any, Any]]:
        """All six file tools, ready to attach to an agent."""
        return [
            self._read_tool,
            self._write_tool,
            self._edit_tool,
            self._delete_tool,
            self._glob_tool,
            self._grep_tool,
        ]

    def read_only_tools(self) -> list[BaseTool[Any, Any, Any]]:
        """The non-mutating subset: ``Read`` + ``Glob`` + ``Grep``."""
        return [self._read_tool, self._glob_tool, self._grep_tool]
