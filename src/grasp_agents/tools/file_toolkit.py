"""
``FileToolkit`` — one factory bundling every built-in file tool:
``Read`` / ``Write`` / ``Edit`` / ``Delete`` (from :mod:`.file_edit`) and
``Glob`` / ``Grep`` (from :mod:`.file_search`).

These belong together: an agent doing file-shaped work — memory authoring
especially — both *searches* the tree (Glob/Grep) and *reads/edits* files,
and ``Read`` is shared by both flows. All tools are stateless wrappers —
they consume :attr:`SessionContext.file_backend` at run time and the active
:class:`FileEditSessionState` from the agent loop's :class:`AgentContext`.
The toolkit just bundles their per-tool configuration.

Usage::

    backend = LocalFileBackend(allowed_roots=[Path.cwd()])
    ctx = SessionContext(state=..., file_backend=backend)
    agent = LLMAgent(..., tools=FileToolkit().tools())

For an agent that may explore but not mutate, use :meth:`read_only_tools`
(``Read`` + ``Glob`` + ``Grep``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .file_edit.delete import DeleteTool
from .file_edit.edit import EditTool
from .file_edit.notebook import NotebookEditTool, NotebookReadTool
from .file_edit.read import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_READ_CHARS,
    ReadTool,
)
from .file_edit.read_image import DEFAULT_MAX_IMAGE_BYTES, ReadImageTool
from .file_edit.redact import DefaultSecretRedactor, SecretRedactor
from .file_edit.write import DEFAULT_NEW_FILE_MODE, WriteTool
from .file_search.glob import DEFAULT_HEAD_LIMIT as DEFAULT_GLOB_HEAD_LIMIT
from .file_search.glob import GlobTool
from .file_search.grep import GrepTool

if TYPE_CHECKING:
    from grasp_agents.tools.base import BaseTool


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
        include_notebook: bool = False,
        include_image: bool = False,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
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
            include_notebook: Add ``NotebookRead`` + ``NotebookEdit`` to
                :meth:`tools` (and ``NotebookRead`` to :meth:`read_only_tools`)
                for cell-structured ``.ipynb`` editing. Off by default — most
                agents don't touch notebooks, and the generic ``Read`` already
                renders a notebook's cells (so a non-notebook agent still sees
                them). Requires the ``nbformat`` package.
            include_image: Add ``ReadImage`` to :meth:`tools` and
                :meth:`read_only_tools`, letting the agent view image files
                (PNG/JPEG/GIF/WebP) as visual content. Off by default — only
                vision-capable agents need it.
            max_image_bytes: Whole-file size ceiling for ``ReadImage``; larger
                images are refused. Default ``5_000_000``.
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
        self._notebook_read_tool = NotebookReadTool(timeout=tool_timeout)
        self._notebook_edit_tool = NotebookEditTool(timeout=tool_timeout)
        self._image_read_tool = ReadImageTool(
            max_image_bytes=max_image_bytes,
            timeout=tool_timeout,
        )
        self._include_notebook = include_notebook
        self._include_image = include_image

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

    @property
    def notebook_read(self) -> NotebookReadTool:
        return self._notebook_read_tool

    @property
    def notebook_edit(self) -> NotebookEditTool:
        return self._notebook_edit_tool

    @property
    def read_image(self) -> ReadImageTool:
        return self._image_read_tool

    def tools(self) -> list[BaseTool[Any, Any, Any]]:
        """
        The file tools, ready to attach to an agent.

        ``Read`` / ``Write`` / ``Edit`` / ``Delete`` / ``Glob`` / ``Grep``,
        plus ``NotebookRead`` + ``NotebookEdit`` when ``include_notebook`` and
        ``ReadImage`` when ``include_image``.
        """
        base: list[BaseTool[Any, Any, Any]] = [
            self._read_tool,
            self._write_tool,
            self._edit_tool,
            self._delete_tool,
            self._glob_tool,
            self._grep_tool,
        ]
        if self._include_notebook:
            base += [self._notebook_read_tool, self._notebook_edit_tool]
        if self._include_image:
            base.append(self._image_read_tool)
        return base

    def read_only_tools(self) -> list[BaseTool[Any, Any, Any]]:
        """
        Non-mutating subset: Read, Glob, Grep (plus NotebookRead / ReadImage
        when their ``include_*`` flag is set).
        """
        base: list[BaseTool[Any, Any, Any]] = [
            self._read_tool,
            self._glob_tool,
            self._grep_tool,
        ]
        if self._include_notebook:
            base.append(self._notebook_read_tool)
        if self._include_image:
            base.append(self._image_read_tool)
        return base
