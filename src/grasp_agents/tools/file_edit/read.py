r"""
``Read`` — read a file via the configured :class:`FileBackend`.

Output is line-numbered in ``cat -n`` format (``<line-number>\t<content>``),
with 1-indexed ``offset`` (starting line) and ``limit`` (max lines) for
paginated reads.

Guards, applied in order:

1. Binary-extension block — image/archive/binary bytes pollute context
   (universal — applies to every backend; cheap suffix check).
2. ``backend.validate_path`` — sandbox + sensitive-path policy. Local-FS
   backends additionally enforce device-path blocks and the credential
   dotfile deny list; MCP backends trust their server.
3. Char cap — reject reads whose formatted output would exceed
   ``max_read_chars`` (default 100 KiB); guidance steers the model toward
   ``offset`` / ``limit``.
4. Secret redaction (if configured) — final pass before content returns.

After a successful read the file is registered in the session state's
``read_file_state`` map, enabling read-before-write enforcement in
:class:`WriteTool` and staleness detection in :class:`EditTool`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .paths import PathAccessError, has_binary_extension

if TYPE_CHECKING:
    from ...run_context import RunContext
    from .backend import FileBackend
    from .redact import SecretRedactor
    from .session_state import FileEditSessionState
    from .store import FileEditStore

# Default char cap on the formatted read output. 100 KiB is ~25-35K
# tokens across common tokenizations — a sensible hazard limit for the
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


def _format_window(
    text: str,
    offset: int | None,
    limit: int | None,
    max_read_chars: int,
) -> tuple[str, int]:
    """
    Slice ``text`` to the requested window and format with ``cat -n``.

    Returns ``(formatted_content, total_lines)``. Raises ``ValueError``
    if the formatted output exceeds ``max_read_chars``.
    """
    lines = text.splitlines(keepends=True)
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

    return formatted, total_lines


class ReadTool(BaseTool[ReadInput, ReadResult, Any]):
    """
    Read a text file with pagination via the configured backend.

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
        allowed_roots: list[str] | list[Any],
        redactor: SecretRedactor,
        backend: FileBackend | None = None,
        include_dotfiles: bool = True,
        default_session_key: str = "default",
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        # Local import — keeps the backend module out of the
        # ``types.tool``-triggered import path used by ``RunContext``.
        from .local_backend import LocalFileBackend  # noqa: PLC0415

        self._store = store
        self._backend = backend or LocalFileBackend()
        self._default_session_key = default_session_key
        self._allowed_roots: list[Path] = [Path(r) for r in allowed_roots]
        self._redactor = redactor
        self._include_dotfiles = include_dotfiles
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
    ) -> ReadResult:
        del exec_id, progress_callback

        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot read binary file {inp.path!r}. Use a dedicated "
                "tool for this format."
            )

        state = await self._resolve_state(ctx)

        try:
            resolved = await self._backend.validate_path(
                Path(inp.path),
                self._allowed_roots,
                must_exist=True,
                dotfile_overrides=state.dotfile_overrides,
                include_dotfiles=self._include_dotfiles,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        content, mtime = await self._backend.read_text(resolved)

        formatted, total_lines = _format_window(
            content, inp.offset, inp.limit, self._max_read_chars
        )

        formatted = self._redactor(formatted)

        state.record_read(resolved, mtime)

        return ReadResult(
            path=str(resolved),
            content=formatted,
            total_lines=total_lines,
        )
