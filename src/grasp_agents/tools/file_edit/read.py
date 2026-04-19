r"""
``Read`` — read a file under the toolkit's allowed roots.

Output is line-numbered in ``cat -n`` format (``<line-number>\t<content>``),
with 1-indexed ``offset`` (starting line) and ``limit`` (max lines) for
paginated reads.

Guards, applied in order:

1. Device-path blocklist — ``/dev/stdin`` & friends hang the reader.
2. Binary-extension block — image/archive/binary bytes pollute context.
3. ``allowed_roots`` + ``Path.resolve(strict=True)`` — sandbox enforcement,
   rejects symlink escapes.
4. Char cap — reject reads whose formatted output would exceed
   ``max_read_chars`` (default 100 KiB); guidance steers the model toward
   ``offset`` / ``limit``.
5. Secret redaction (if configured) — final pass before the content returns.

After a successful read the file is registered in the session state's
``read_file_state`` map, enabling read-before-write enforcement in
:class:`WriteTool` and staleness detection in :class:`EditTool`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .paths import (
    PathAccessError,
    has_binary_extension,
    is_blocked_device,
    resolve_safe,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ...run_context import RunContext
    from .redact import SecretRedactor
    from .session_state import FileEditSessionState
    from .store import FileEditStore

# Default char cap on the formatted read output. 100 KiB is ~25-35K
# tokens across typical tokenisers — a sensible hazard limit for the
# context window. Configurable via the toolkit.
DEFAULT_MAX_READ_CHARS = 100_000
DEFAULT_READ_LIMIT = 500


class ReadInput(BaseModel):
    """Input schema for the ``Read`` tool."""

    path: str = Field(
        description=(
            "Absolute, relative, or ~-expanded path to the file to read. "
            "Must resolve under one of the toolkit's allowed roots."
        )
    )
    offset: int | None = Field(
        default=None,
        description=(
            "1-indexed line number to start reading from. "
            "Defaults to 1 (start of file)."
        ),
        ge=1,
    )
    limit: int | None = Field(
        default=None,
        description=(
            f"Maximum number of lines to return. Defaults to {DEFAULT_READ_LIMIT}."
        ),
        ge=1,
    )


class ReadResult(BaseModel):
    """Output schema for the ``Read`` tool."""

    path: str
    content: str
    total_lines: int


def _read_file_sync(
    resolved: Path,
    offset: int | None,
    limit: int | None,
    max_read_chars: int,
) -> tuple[str, int, float]:
    """
    Synchronous read + format. Called via ``asyncio.to_thread``.

    Returns ``(formatted_content, total_lines, mtime)``. Raises
    :class:`ValueError` if the formatted content exceeds
    ``max_read_chars`` (converted to a ToolError at the boundary).
    """
    mtime = resolved.stat().st_mtime

    # ``errors="replace"`` lets us read files with stray invalid bytes
    # without raising. The binary-extension guard already filters the
    # obvious binary cases; this is for the mixed / accidental cases.
    with resolved.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total_lines = len(lines)

    start = (offset or 1) - 1
    effective_limit = limit if limit is not None else DEFAULT_READ_LIMIT
    end = start + effective_limit
    window = lines[start:end]

    formatted_parts: list[str] = []
    for i, line in enumerate(window, start=start + 1):
        formatted_parts.append(f"{i:>6}\t{line.rstrip(chr(10))}")
    formatted = "\n".join(formatted_parts)

    if len(formatted) > max_read_chars:
        raise ValueError(
            f"Read produced {len(formatted):,} characters, exceeding the "
            f"safety limit ({max_read_chars:,}). Use `offset` and `limit` "
            f"to narrow the range. File has {total_lines} lines total."
        )

    return formatted, total_lines, mtime


class ReadTool(BaseTool[ReadInput, ReadResult, Any]):
    """
    Read a file with pagination.

    Attach via :class:`FileEditToolkit`; do not instantiate directly
    unless you're constructing a custom toolkit.
    """

    name = "Read"
    description = (
        "Read a text file with pagination and line numbers. Use this "
        "instead of `cat` / `head` / `tail`. Output is in `cat -n` format "
        "(line number + tab + content). Supports `offset` (1-indexed start "
        "line) and `limit` (max lines). Large reads are rejected — use "
        "`offset` and `limit` to narrow the range. Binary files are "
        "refused; use dedicated tools for images / PDFs."
    )

    def __init__(
        self,
        *,
        store: FileEditStore,
        allowed_roots: list[Path],
        redactor: SecretRedactor,
        default_session_key: str = "default",
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._store = store
        self._default_session_key = default_session_key
        self._allowed_roots = allowed_roots
        self._redactor = redactor
        self._max_read_chars = max_read_chars

    async def _resolve_state(self, ctx: RunContext[Any] | None) -> FileEditSessionState:
        """
        Pick the session state this call should read/write.

        Prefers the store + session key on ``ctx`` (production path —
        multi-session safe) and falls back to the tool's own store with
        ``default_session_key`` (standalone/testing path).
        """
        if ctx is not None and ctx.file_edit_store is not None:
            return await ctx.file_edit_store.get_session_state(ctx.session_key)
        return await self._store.get_session_state(self._default_session_key)

    async def _run(
        self,
        inp: ReadInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> ReadResult:
        del exec_id, progress_callback, session_id

        # 1. Device-path guard — literal path, no resolve.
        if is_blocked_device(inp.path):
            raise ValueError(
                f"Cannot read device path {inp.path!r}: would block or "
                "produce infinite output."
            )

        # 2. Binary-extension guard.
        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot read binary file {inp.path!r}. Use a dedicated "
                "tool for this format."
            )

        # 3. Sandbox check (must_exist=True for Read).
        try:
            resolved = resolve_safe(inp.path, self._allowed_roots, must_exist=True)
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        # 4. Perform the read off the event loop.
        formatted, total_lines, mtime = await asyncio.to_thread(
            _read_file_sync,
            resolved,
            inp.offset,
            inp.limit,
            self._max_read_chars,
        )

        # 5. Redaction pass.
        formatted = self._redactor(formatted)

        # 6. Register the read for read-before-write + staleness checks.
        state = await self._resolve_state(ctx)
        state.record_read(resolved, mtime)

        return ReadResult(
            path=str(resolved),
            content=formatted,
            total_lines=total_lines,
        )
