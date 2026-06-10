"""
``Grep`` — regex search over files routed through the configured
:class:`FileBackend`.

The :class:`LocalFileBackend` drives ``rg`` (ripgrep) directly:
``rg`` already honours ``.gitignore``, skips hidden files, does binary
detection, and is orders of magnitude faster than a pure-Python
walk+regex. Callers must have ``rg`` on ``PATH`` — the tool raises a
structured error otherwise, pointing at the install.

Other backends (e.g. :class:`MCPFileBackend`) may expose their own grep
implementation; backends that don't ship one raise
:class:`NotImplementedError` with a clear message.

Three output modes, chosen via ``output_mode``:

- ``files_with_matches`` (default): one file path per matching file.
  Cheapest mode; start here and drill in with ``content`` once you know
  which files are relevant.
- ``count``: ``path:N`` per file showing match counts.
- ``content``: ``path:line:<content>`` per matching line. Supports
  ``-A`` / ``-B`` / ``-C`` context lines and ``-n`` line numbers.

``head_limit`` / ``offset`` slice the output after the backend returns.
The tool reports ``truncated=True`` when ``head_limit`` cut results so
the model can narrow the query instead of paging blindly.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from ..file_backend.base import GrepRawResult
from ..file_backend.paths import PathAccessError

if TYPE_CHECKING:
    from ...agent.agent_context import AgentContext
    from ...run_context import RunContext

DEFAULT_HEAD_LIMIT = 250

# Cap on total bytes we'll read from rg's stdout. A runaway match could
# return hundreds of MB; refuse with guidance to narrow the query.
MAX_STDOUT_BYTES = 50 * 1024 * 1024  # 50 MiB

OutputMode = Literal["files_with_matches", "content", "count"]


class GrepInput(BaseModel):
    """Input schema for the ``Grep`` tool."""

    pattern: str = Field(
        description=(
            "Regular expression to search for. Uses ripgrep's default "
            "Rust regex flavor (Perl-style syntax minus lookbehinds). "
            "Examples: ``def \\w+``, ``TODO|FIXME``, ``import \\S+``. "
            "Escape literal braces (``interface\\{\\}`` to match Go "
            "``interface{}``)."
        ),
        min_length=1,
    )
    path: str | None = Field(
        default=None,
        description=(
            "File or directory to search. Must resolve under one of the "
            "toolkit's allowed roots. Defaults to the first allowed root."
        ),
    )
    glob: str | None = Field(
        default=None,
        description=(
            "Filter files by glob (e.g. ``*.py``, ``**/*.tsx``). Combines "
            "with ``type`` if both are set."
        ),
    )
    type: str | None = Field(
        default=None,
        description=(
            "Filter by file type using rg's named types (``py``, ``js``, "
            "``rust``, ``go``, ``md``, ...). See ``rg --type-list`` for "
            "the full set."
        ),
    )
    output_mode: OutputMode = Field(
        default="files_with_matches",
        description=(
            "``files_with_matches`` (default): one path per matching file. "
            "``content``: matching lines with ``-A`` / ``-B`` / ``-C`` "
            "context. ``count``: per-file match counts."
        ),
    )
    case_insensitive: bool = Field(
        default=False,
        description="Case-insensitive match (rg ``-i``).",
    )
    show_line_numbers: bool = Field(
        default=True,
        description=(
            "Show line numbers in ``content`` mode (rg ``-n``). Ignored in other modes."
        ),
    )
    before_context: int | None = Field(
        default=None,
        ge=0,
        description="Lines of context before each match (rg ``-B``).",
    )
    after_context: int | None = Field(
        default=None,
        ge=0,
        description="Lines of context after each match (rg ``-A``).",
    )
    context: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Lines of context before and after each match (rg ``-C``). "
            "Overrides ``before_context`` / ``after_context`` if set."
        ),
    )
    multiline: bool = Field(
        default=False,
        description=(
            "Let ``.`` match newlines and allow patterns to span lines "
            "(rg ``-U --multiline-dotall``). Costs performance — leave "
            "off unless you need cross-line matches."
        ),
    )
    head_limit: int | None = Field(
        default=DEFAULT_HEAD_LIMIT,
        ge=1,
        description=(
            "Cap on returned entries (lines for ``content``, files "
            "otherwise). Defaults to 250. Pair with ``offset`` for "
            "pagination."
        ),
    )
    offset: int | None = Field(
        default=0,
        ge=0,
        description="Skip the first N entries before applying ``head_limit``.",
    )


class GrepResult(BaseModel):
    """Output schema for the ``Grep`` tool."""

    output: str
    output_mode: OutputMode
    num_matches: int
    num_files_matched: int
    truncated: bool


class GrepError(ValueError):
    """Raised when ``rg`` is missing or fails with a non-zero-non-1 exit."""


# ---------------------------------------------------------------------------
# Local backend implementation — drives rg directly
# ---------------------------------------------------------------------------


async def _run_rg(args: list[str]) -> tuple[bytes, bytes, int]:
    """Run rg with ``args`` and return ``(stdout, stderr, returncode)``."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "rg",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise GrepError(
            "`rg` (ripgrep) not found on PATH. Install it via your "
            "package manager (e.g. `brew install ripgrep`, "
            "`apt install ripgrep`) and retry."
        ) from exc

    stdout, stderr = await proc.communicate()
    if len(stdout) > MAX_STDOUT_BYTES:
        raise GrepError(
            f"rg produced {len(stdout):,} bytes of output "
            f"(> {MAX_STDOUT_BYTES:,}). Narrow the search path, glob, or "
            "pattern."
        )
    return stdout, stderr, proc.returncode or 0


def _build_args(
    *,
    pattern: str,
    resolved_path: Path,
    mode: OutputMode,
    glob: str | None,
    file_type: str | None,
    case_insensitive: bool,
    multiline: bool,
    before_context: int | None,
    after_context: int | None,
    context: int | None,
) -> list[str]:
    """Translate grep params into the rg CLI args for ``mode``."""
    args: list[str] = ["--sort", "path"]

    if case_insensitive:
        args.append("--ignore-case")

    if multiline:
        args.extend(["--multiline", "--multiline-dotall"])

    if glob is not None:
        args.extend(["--glob", glob])

    if file_type is not None:
        args.extend(["--type", file_type])

    if mode == "files_with_matches":
        args.append("--files-with-matches")
    elif mode == "count":
        args.append("--count")
    else:
        args.append("--json")
        if context is not None:
            args.extend(["--context", str(context)])
        else:
            if before_context is not None:
                args.extend(["--before-context", str(before_context)])
            if after_context is not None:
                args.extend(["--after-context", str(after_context)])

    args.extend(["--", pattern, str(resolved_path)])
    return args


def _parse_files_with_matches(stdout: bytes) -> list[Path]:
    text = stdout.decode("utf-8", errors="replace")
    return [Path(line) for line in text.splitlines() if line]


def _parse_count(stdout: bytes) -> list[tuple[Path, int]]:
    r"""Parse ``path:N\n`` output; ``N == 0`` lines are omitted by rg."""
    entries: list[tuple[Path, int]] = []
    text = stdout.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line:
            continue
        path, _, count_s = line.rpartition(":")
        if not path or not count_s.isdigit():
            continue
        entries.append((Path(path), int(count_s)))
    return entries


def _get_text(obj: dict[str, Any], key: str) -> str | None:
    """Extract ``obj[key]["text"]`` as a string, or None if the shape differs."""
    nested = obj.get(key)
    if not isinstance(nested, dict):
        return None
    text = cast("dict[str, Any]", nested).get("text")
    return text if isinstance(text, str) else None


def _parse_json_content(
    stdout: bytes, *, show_line_numbers: bool
) -> tuple[list[str], int, set[str]]:
    r"""
    Parse rg's ``--json`` stream into ``(lines, num_matches, paths)``.
    """
    lines: list[str] = []
    num_matches = 0
    paths: set[str] = set()
    current_path: str | None = None

    for raw_line in stdout.splitlines():
        if not raw_line:
            continue
        try:
            obj: Any = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg = cast("dict[str, Any]", obj)
        msg_type = msg.get("type")
        data_obj = msg.get("data")
        data: dict[str, Any] = (
            cast("dict[str, Any]", data_obj) if isinstance(data_obj, dict) else {}
        )

        if msg_type == "begin":
            path_text = _get_text(data, "path")
            if path_text is not None:
                current_path = path_text
                paths.add(path_text)
            continue

        if msg_type in {"match", "context"}:
            path = _get_text(data, "path") or current_path or ""
            line_no = data.get("line_number")
            text = (_get_text(data, "lines") or "").rstrip("\n")
            sep = ":" if msg_type == "match" else "-"
            if show_line_numbers and isinstance(line_no, int):
                rendered = f"{path}{sep}{line_no}{sep}{text}"
            else:
                rendered = f"{path}{sep}{text}"
            lines.append(rendered)
            if msg_type == "match":
                num_matches += 1

    return lines, num_matches, paths


async def local_backend_grep(
    *,
    root: Path,
    pattern: str,
    glob: str | None,
    file_type: str | None,
    case_insensitive: bool,
    multiline: bool,
    output_mode: OutputMode,
    show_line_numbers: bool,
    before_context: int | None,
    after_context: int | None,
    context: int | None,
) -> GrepRawResult:
    """
    Local-FS grep via rg. Used by :meth:`LocalFileBackend.grep`. Public
    so alternate backends can share the rg invocation if they shell to
    a local rg too.
    """
    args = _build_args(
        pattern=pattern,
        resolved_path=root,
        mode=output_mode,
        glob=glob,
        file_type=file_type,
        case_insensitive=case_insensitive,
        multiline=multiline,
        before_context=before_context,
        after_context=after_context,
        context=context,
    )
    stdout, stderr, rc = await _run_rg(args)
    if rc not in {0, 1}:
        err = stderr.decode("utf-8", errors="replace").strip() or "unknown error"
        raise GrepError(f"rg exited with code {rc}: {err}")

    if output_mode == "files_with_matches":
        files = _parse_files_with_matches(stdout)
        return GrepRawResult(
            files=files,
            num_matches=len(files),
            num_files_matched=len(files),
        )

    if output_mode == "count":
        entries = _parse_count(stdout)
        total_matches = sum(n for _, n in entries)
        return GrepRawResult(
            counts=entries,
            num_matches=total_matches,
            num_files_matched=len(entries),
        )

    # content mode
    lines, num_matches, paths = _parse_json_content(
        stdout, show_line_numbers=show_line_numbers
    )
    return GrepRawResult(
        lines=lines,
        num_matches=num_matches,
        num_files_matched=len(paths),
    )


# ---------------------------------------------------------------------------
# Tool — delegates to ``backend.grep``
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


def _slice(
    entries: list[_T], *, offset: int, head_limit: int | None
) -> tuple[list[_T], bool]:
    """Apply ``offset`` + ``head_limit``; return sliced list + trunc flag."""
    total = len(entries)
    start = min(offset, total)
    if head_limit is None:
        return entries[start:], False
    end = start + head_limit
    return entries[start:end], end < total


class GrepTool(BaseTool[GrepInput, GrepResult, Any]):
    """
    Regex search via ``ctx.file_backend``.

    Stateless: backend + allowed_roots live on
    :attr:`RunContext.file_backend`.
    """

    name = "Grep"
    description = (
        "Search file contents with a regex, backed by ripgrep.\n"
        "\n"
        "* Three output modes: ``files_with_matches`` (default — matching "
        "files), ``content`` (matching lines with optional context), "
        "``count`` (per-file match counts).\n"
        "* Filter files with ``glob`` (``*.py``) or ``type`` (``py``, ``js``, "
        "``rust``). Use ``case_insensitive``, ``context`` / ``before_context`` "
        "/ ``after_context``, and ``multiline`` as needed.\n"
        "* Large result sets are capped by ``head_limit`` (default 250) — page "
        "with ``offset``.\n"
        "* Returns the formatted ``output`` for the chosen mode, "
        "``num_matches``, ``num_files_matched``, and ``truncated``."
    )
    untrusted_output = True

    def __init__(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)

    async def _run(
        self,
        inp: GrepInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> GrepResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Grep requires ctx.file_backend. Wire a FileBackend on "
                "RunContext before running the agent."
            )

        backend = ctx.file_backend
        roots = backend.allowed_roots
        if inp.path is not None:
            raw_path = Path(inp.path)
        elif roots:
            raw_path = roots[0]
        else:
            raise ValueError(
                "Grep requires a path or a backend with at least one allowed_root."
            )

        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        try:
            resolved = await backend.validate_path(
                raw_path, must_exist=True, dotfile_overrides=overrides
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        raw = await backend.grep(
            root=resolved,
            pattern=inp.pattern,
            glob=inp.glob,
            file_type=inp.type,
            case_insensitive=inp.case_insensitive,
            multiline=inp.multiline,
            output_mode=inp.output_mode,
            show_line_numbers=inp.show_line_numbers,
            before_context=inp.before_context,
            after_context=inp.after_context,
            context=inp.context,
        )

        offset = inp.offset or 0

        if inp.output_mode == "files_with_matches":
            sliced_paths, truncated = _slice(
                raw.files, offset=offset, head_limit=inp.head_limit
            )
            return GrepResult(
                output="\n".join(str(p) for p in sliced_paths),
                output_mode=inp.output_mode,
                num_matches=raw.num_matches,
                num_files_matched=raw.num_files_matched,
                truncated=truncated,
            )

        if inp.output_mode == "count":
            rendered = [f"{p}:{n}" for p, n in raw.counts]
            sliced, truncated = _slice(
                rendered, offset=offset, head_limit=inp.head_limit
            )
            return GrepResult(
                output="\n".join(sliced),
                output_mode=inp.output_mode,
                num_matches=raw.num_matches,
                num_files_matched=raw.num_files_matched,
                truncated=truncated,
            )

        sliced, truncated = _slice(raw.lines, offset=offset, head_limit=inp.head_limit)
        return GrepResult(
            output="\n".join(sliced),
            output_mode=inp.output_mode,
            num_matches=raw.num_matches,
            num_files_matched=raw.num_files_matched,
            truncated=truncated,
        )


def rg_available() -> bool:
    """Return True if ``rg`` is on the system PATH — useful for test guards."""
    return shutil.which("rg") is not None
