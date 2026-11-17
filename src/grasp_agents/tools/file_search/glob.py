"""
``Glob`` — fast path-pattern matching routed through ``ctx.file_backend``.

Accepts standard glob patterns (``*.py``, ``**/*.tsx``, ``src/**/test_*.py``)
and returns matching file paths sorted by mtime (newest first) — useful for
"what changed recently" queries.

Guards:

1. ``allowed_roots`` — the search root must be under one of the configured
   roots on the backend; the backend resolves and enforces containment.
2. Read access — matches the backend would refuse to ``Read`` (its
   ``deny_read`` carve-outs, credential files) are dropped, so their names
   do not leak.
3. Result cap — large result sets are truncated to ``head_limit`` (default
   250) and ``truncated=True`` is returned so the model can narrow the
   pattern instead of parsing an unbounded list.
4. Hidden directories (``.git``, ``.venv``, ``node_modules``, ...) are
   skipped by default; opt in with ``include_hidden=True`` on the
   :class:`GlobTool` constructor.
"""

from __future__ import annotations

import stat as stat_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.tools.base import BaseTool, ToolProgressCallback

from ._read_access import readable_paths

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.file_backend.base import FileStat
    from grasp_agents.session_context import SessionContext


def _is_directory(stat: FileStat) -> bool:
    """
    Best-effort isdir check.

    The built-in backends report POSIX file-type bits, so ``S_ISDIR``
    decides. A backend whose ``mode`` carries no type bits gets the benefit
    of the doubt: rooting Glob at a non-directory is the unusual case.
    """
    if stat.mode and stat_module.S_IFMT(stat.mode):
        return stat_module.S_ISDIR(stat.mode)
    return True


# Default cap on the returned ``files`` list. 250 balances "enough to see
# the shape of the repo" against "context bloat". Callers raise via the
# constructor when a larger budget is needed.
DEFAULT_HEAD_LIMIT = 250


class GlobInput(BaseModel):
    """Input schema for the ``Glob`` tool."""

    pattern: str = Field(
        description=(
            "Glob pattern to match file paths against. Supports ``*``, "
            "``?``, ``[abc]``, and ``**`` (recursive). Examples: "
            "``*.py``, ``**/*.tsx``, ``src/**/test_*.py``."
        ),
        min_length=1,
    )
    path: str | None = Field(
        default=None,
        description=(
            "Directory to search within. Must resolve under one of the "
            "backend's allowed roots. Defaults to the first allowed root."
        ),
    )


class GlobResult(BaseModel):
    """Output schema for the ``Glob`` tool."""

    files: list[str]
    num_files: int
    truncated: bool


class GlobTool(BaseTool[GlobInput, GlobResult, Any]):
    """
    Match file paths against a glob pattern, sorted by mtime.

    Stateless: backend + allowed_roots live on
    :attr:`SessionContext.file_backend`.
    """

    name = "Glob"
    description = (
        "Find files by glob pattern, sorted by modification time (newest "
        "first).\n"
        "\n"
        "* Supports ``*``, ``?``, ``[abc]``, and ``**`` (recursive). "
        "Examples: ``*.py`` (top-level Python), ``**/*.tsx`` (all TSX), "
        "``src/**/test_*.py``.\n"
        "* Hidden files and common build/cache dirs are skipped. Results are "
        "capped at the tool's ``head_limit`` — narrow the pattern if "
        "truncated.\n"
        "* Returns the matching paths (newest first), ``num_files``, and "
        "``truncated``."
    )

    def __init__(
        self,
        *,
        head_limit: int = DEFAULT_HEAD_LIMIT,
        include_hidden: bool = False,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._head_limit = head_limit
        self._include_hidden = include_hidden

    async def _run(
        self,
        inp: GlobInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> GlobResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Glob requires ctx.file_backend. Wire a FileBackend on "
                "SessionContext before running the agent."
            )

        backend = ctx.file_backend
        roots = backend.allowed_roots
        if inp.path is not None:
            raw_root = Path(inp.path)
        elif roots:
            raw_root = roots[0]
        else:
            raise ValueError(
                "Glob requires a path or a backend with at least one allowed_root."
            )

        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        try:
            resolved = await backend.validate_path(
                raw_root, must_exist=True, dotfile_overrides=overrides
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        stat = await backend.stat(resolved)
        if not _is_directory(stat):
            raise ValueError(f"Glob search path must be a directory: {resolved}")

        matched, truncated = await backend.find_files(
            resolved,
            inp.pattern,
            include_hidden=self._include_hidden,
            head_limit=self._head_limit,
        )
        readable = await readable_paths(
            backend, (m.path for m in matched), dotfile_overrides=overrides
        )
        matched = [m for m in matched if m.path in readable]

        # Sort newest-first by mtime.
        matched.sort(key=lambda m: m.mtime, reverse=True)
        kept = matched[: self._head_limit]
        is_truncated = truncated or len(matched) > self._head_limit

        return GlobResult(
            files=[str(m.path) for m in kept],
            num_files=len(kept),
            truncated=is_truncated,
        )
