r"""
``Read`` — read a file via ``ctx.file_backend``.

Output is line-numbered in ``cat -n`` format (``<line-number>\t<content>``),
with 1-indexed ``offset`` (starting line) and ``limit`` (max lines) for
paginated reads.

Guards, applied in order:

1. Binary-extension block — image/archive/binary bytes pollute context
   (universal — applies to every backend; cheap suffix check).
2. ``backend.validate_path`` — sandbox + sensitive-path policy. Local-FS
   backends additionally enforce device-path blocks and the credential
   dotfile deny list; MCP backends trust their server.
3. Size gate — reject files larger than ``max_file_bytes`` (default
   10 MB): too large to open at all, even paginated.
4. Char cap — when the formatted window would exceed ``max_read_chars``
   (default 100 KiB) it is truncated at a line boundary and a notice is
   appended steering the model toward ``offset`` / ``limit``. Reads are
   no longer rejected for being too wide — only whole files past the
   size gate are.
5. Secret redaction (if configured) — final pass before content returns.

The backend records the read internally so subsequent writes pass the
read-before-write check.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from ..file_backend.paths import PathAccessError, has_binary_extension

if TYPE_CHECKING:
    from ...agent.agent_context import AgentContext
    from ...run_context import RunContext
    from .redact import SecretRedactor

# Default char cap on the formatted read output. 100 KiB is ~25-35K
# tokens across common tokenizations — a sensible hazard limit for the
# context window. Configurable via the toolkit.
DEFAULT_MAX_READ_CHARS = 100_000
DEFAULT_READ_LIMIT = 500
# Whole-file hard ceiling. Files past this are refused outright (pagination
# can't help: ``read_text`` loads the whole file into memory regardless of
# the requested window). Well above ``max_read_chars`` so ordinary large
# source files can still be read in windows.
DEFAULT_MAX_FILE_BYTES = 10_000_000


class ReadInput(BaseModel):
    """Input schema for the ``Read`` tool."""

    path: str = Field(
        description=(
            "Absolute, relative, or ~-expanded path to the file to read. "
            "Must resolve under one of the backend's allowed roots."
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
    truncated: bool = False


def _format_window(
    text: str,
    offset: int | None,
    limit: int | None,
    max_read_chars: int,
) -> tuple[str, int, int, bool]:
    """
    Slice ``text`` to the requested window and format with ``cat -n``.

    Returns ``(formatted_content, total_lines, last_line, truncated)``.
    ``last_line`` is the 1-indexed number of the last line included.
    When the formatted window would exceed ``max_read_chars`` it is cut at
    a line boundary and ``truncated`` is True — the caller appends a notice
    so the model can continue with ``offset=last_line + 1`` rather than
    receiving an error.
    """
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    start = (offset or 1) - 1
    effective_limit = limit if limit is not None else DEFAULT_READ_LIMIT
    end = min(start + effective_limit, total_lines)
    window = lines[start:end]

    formatted_parts: list[str] = []
    used = 0
    truncated = False
    for i, line in enumerate(window, start=start + 1):
        rendered = f"{i:>6}\t{line.rstrip(chr(10))}"
        # +1 for the newline that joins this line to the previous one.
        added = len(rendered) + (1 if formatted_parts else 0)
        if formatted_parts and used + added > max_read_chars:
            truncated = True
            break
        formatted_parts.append(rendered)
        used += added

    formatted = "\n".join(formatted_parts)

    # A single line longer than the cap: keep the first line but hard-cut
    # it so the returned output never exceeds the budget.
    if len(formatted) > max_read_chars:
        formatted = formatted[:max_read_chars]
        truncated = True

    last_line = start + len(formatted_parts)
    return formatted, total_lines, last_line, truncated


class ReadTool(BaseTool[ReadInput, ReadResult, Any]):
    """
    Read a text file with pagination via ``ctx.file_backend``.

    Stateless: backend, allowed_roots, and read-state bookkeeping all
    live on the :class:`FileBackend` wired onto :attr:`RunContext.file_backend`.
    """

    name = "Read"
    description = (
        "Read a text file with pagination and line numbers, in `cat -n` "
        "format (line number + tab + content). Use it instead of `cat` / "
        "`head` / `tail`.\n"
        "\n"
        "* Page through large files with `offset` (1-indexed start line) and "
        "`limit` (max lines); oversized reads are rejected, so narrow the "
        "range.\n"
        "* Binary files are refused — use the dedicated image / PDF tools.\n"
        "* Returns the numbered text, `total_lines`, and `truncated` (true "
        "when more lines exist than were returned — page with `offset`)."
    )

    def __init__(
        self,
        *,
        redactor: SecretRedactor | None = None,
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        from .redact import DefaultSecretRedactor  # noqa: PLC0415

        self._redactor: SecretRedactor = redactor or DefaultSecretRedactor()
        self._max_read_chars = max_read_chars
        self._max_file_bytes = max_file_bytes

    async def _run(
        self,
        inp: ReadInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> ReadResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Read requires ctx.file_backend. Wire a FileBackend on "
                "RunContext before running the agent."
            )

        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot read binary file {inp.path!r}. Use a dedicated "
                "tool for this format."
            )

        backend = ctx.file_backend
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        try:
            resolved = await backend.validate_path(
                Path(inp.path),
                must_exist=True,
                dotfile_overrides=overrides,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        # A notebook is JSON; show its cell-structured view (never raw JSON)
        # so the model can target cells by id in a subsequent NotebookEdit.
        if resolved.suffix == ".ipynb":
            from .notebook import read_notebook_as_text  # noqa: PLC0415

            rendered, mtime = await read_notebook_as_text(backend, resolved)
            if state is not None:
                state.record_read(resolved, mtime)
            rendered = self._redactor(rendered)
            nb_truncated = len(rendered) > self._max_read_chars
            if nb_truncated:
                rendered = (
                    rendered[: self._max_read_chars]
                    + f"\n\n[Notebook view truncated at {self._max_read_chars} "
                    "chars. Use NotebookRead with a cell_id to inspect one cell.]"
                )
            return ReadResult(
                path=str(resolved),
                content=rendered,
                total_lines=rendered.count("\n") + 1,
                truncated=nb_truncated,
            )

        # Hard size gate: a file too large to open at all. Pagination can't
        # rescue it — ``read_text`` reads the whole file into memory first.
        file_size = (await backend.stat(resolved)).size
        if file_size > self._max_file_bytes:
            raise ValueError(
                f"File is {file_size:,} bytes, exceeding the maximum "
                f"readable size ({self._max_file_bytes:,}). It is too large "
                "to open; use a targeted tool (e.g. grep) to inspect it."
            )

        content, mtime = await backend.read_text(resolved)
        if state is not None:
            state.record_read(resolved, mtime)

        formatted, total_lines, last_line, truncated = _format_window(
            content, inp.offset, inp.limit, self._max_read_chars
        )

        formatted = self._redactor(formatted)
        if truncated:
            formatted += (
                f"\n\n[Read truncated: showing lines {(inp.offset or 1)}-"
                f"{last_line} of {total_lines}. Output hit the "
                f"{self._max_read_chars:,}-char limit — continue with "
                f"offset={last_line + 1}.]"
            )

        return ReadResult(
            path=str(resolved),
            content=formatted,
            total_lines=total_lines,
            truncated=truncated,
        )
