"""
``FileSearchToolkit`` — factory that bundles read-only search tools
(``Glob`` + ``Grep``) with shared ``allowed_roots``.

Usage::

    toolkit = FileSearchToolkit(allowed_roots=[Path.cwd()])
    agent.tools = [*toolkit.tools(), ...]

Unlike :class:`FileEditToolkit` these tools are stateless — no store,
no session, no read-before-write tracking. The toolkit exists purely to
keep ``allowed_roots`` and per-tool configuration in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .glob import DEFAULT_HEAD_LIMIT as DEFAULT_GLOB_HEAD_LIMIT
from .glob import GlobTool
from .grep import GrepTool

if TYPE_CHECKING:
    from ...types.tool import BaseTool


class FileSearchToolkit:
    """Build and hold a matching set of read-only file-search tools."""

    def __init__(
        self,
        *,
        allowed_roots: list[Path] | None = None,
        glob_head_limit: int = DEFAULT_GLOB_HEAD_LIMIT,
        glob_include_hidden: bool = False,
        tool_timeout: float | None = None,
    ) -> None:
        """
        Create a toolkit.

        Args:
            allowed_roots: Directories the tools may search under.
                Defaults to ``[Path.cwd()]``. Each entry is expanded and
                resolved at tool-call time.
            glob_head_limit: Cap on ``Glob`` results before truncation.
                Default 250.
            glob_include_hidden: Let ``Glob`` traverse into hidden
                directories (``.foo``). Common build/cache dirs
                (``__pycache__``, ``.git``, ``node_modules``, ...) are
                always skipped regardless of this flag.
            tool_timeout: Per-tool async timeout in seconds. ``None``
                disables the timeout.

        """
        self._allowed_roots: list[Path] = (
            list(allowed_roots) if allowed_roots is not None else [Path.cwd()]
        )
        self._tool_timeout = tool_timeout

        self._glob_tool = GlobTool(
            allowed_roots=self._allowed_roots,
            head_limit=glob_head_limit,
            include_hidden=glob_include_hidden,
            timeout=tool_timeout,
        )
        self._grep_tool = GrepTool(
            allowed_roots=self._allowed_roots,
            timeout=tool_timeout,
        )

    @property
    def glob(self) -> GlobTool:
        return self._glob_tool

    @property
    def grep(self) -> GrepTool:
        return self._grep_tool

    def tools(self) -> list[BaseTool[Any, Any, Any]]:
        """Return the configured tools as a list, ready to attach to an agent."""
        return [self._glob_tool, self._grep_tool]
