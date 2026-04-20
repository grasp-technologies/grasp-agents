"""
``Grep`` — regex search over files under the toolkit's allowed roots.

Shells out to ``rg`` (ripgrep): it already honours ``.gitignore``, skips
hidden files, does binary detection, and is orders of magnitude faster
than a pure-Python walk+regex. Callers must have ``rg`` on ``PATH`` —
the tool raises a structured error otherwise, pointing at the install.

Three output modes, chosen via ``output_mode``:

- ``files_with_matches`` (default): one file path per matching file.
  Cheapest mode; start here and drill in with ``content`` once you know
  which files are relevant.
- ``count``: ``path:N`` per file showing match counts.
- ``content``: ``path:line:<content>`` per matching line. Supports
  ``-A`` / ``-B`` / ``-C`` context lines and ``-n`` line numbers.

``head_limit`` / ``offset`` slice the output after rg returns. The tool
reports ``truncated=True`` when ``head_limit`` cut results so the model
can narrow the query instead of paging blindly.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from ..file_edit.paths import PathAccessError, resolve_safe

if TYPE_CHECKING:
    from pathlib import Path

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
            "Show line numbers in ``content`` mode (rg ``-n``). Ignored "
            "in other modes."
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


def _build_args(inp: GrepInput, resolved_path: Path, mode: OutputMode) -> list[str]:
    """Translate ``GrepInput`` into the rg CLI args for ``mode``."""
    # ``--sort path`` disables rg's parallel walking in exchange for
    # deterministic output. Needed for pagination (offset + head_limit)
    # to partition the result set reproducibly, and costs negligibly on
    # any sensible codebase.
    args: list[str] = ["--sort", "path"]

    if inp.case_insensitive:
        args.append("--ignore-case")

    if inp.multiline:
        args.extend(["--multiline", "--multiline-dotall"])

    if inp.glob is not None:
        args.extend(["--glob", inp.glob])

    if inp.type is not None:
        args.extend(["--type", inp.type])

    if mode == "files_with_matches":
        args.append("--files-with-matches")
    elif mode == "count":
        args.append("--count")
    else:
        # content mode — structured JSON output
        args.append("--json")
        # rg --json always emits line numbers; ``show_line_numbers=False``
        # is honoured post-parse by stripping ``:line:`` from the format.
        if inp.context is not None:
            args.extend(["--context", str(inp.context)])
        else:
            if inp.before_context is not None:
                args.extend(["--before-context", str(inp.before_context)])
            if inp.after_context is not None:
                args.extend(["--after-context", str(inp.after_context)])

    # Terminate flag parsing so a pattern starting with ``-`` is not
    # interpreted as a flag.
    args.extend(["--", inp.pattern, str(resolved_path)])

    return args


def _parse_files_with_matches(stdout: bytes) -> list[str]:
    text = stdout.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if line]


def _parse_count(stdout: bytes) -> list[tuple[str, int]]:
    r"""Parse ``path:N\n`` output; ``N == 0`` lines are omitted by rg."""
    entries: list[tuple[str, int]] = []
    text = stdout.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line:
            continue
        # Filenames can contain colons; split from the right once.
        path, _, count_s = line.rpartition(":")
        if not path or not count_s.isdigit():
            # Malformed line — surface as zero rather than crash the
            # parse. Should not happen with rg's own output.
            continue
        entries.append((path, int(count_s)))
    return entries


def _get_text(obj: dict[str, Any], key: str) -> str | None:
    """Extract ``obj[key]["text"]`` as a string, or None if the shape differs."""
    nested = obj.get(key)
    if not isinstance(nested, dict):
        return None
    # Narrow the dict's generic parameters; the leaf ``isinstance(text, str)``
    # below remains the only runtime shape check we rely on.
    text = cast("dict[str, Any]", nested).get("text")
    return text if isinstance(text, str) else None


def _parse_json_content(
    stdout: bytes, *, show_line_numbers: bool
) -> tuple[list[str], int, set[str]]:
    r"""
    Parse rg's ``--json`` stream into ``(lines, num_matches, paths)``.

    Each rendered line is ``path:line:content`` or ``path:content`` per
    ``show_line_numbers``. Context lines use ``-`` as the separator so
    the model can tell matches apart from context (rg's own convention).
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
            # Strip the trailing newline rg embeds in ``lines.text``.
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


def _slice(
    entries: list[str], *, offset: int, head_limit: int | None
) -> tuple[list[str], bool]:
    """
    Apply ``offset`` + ``head_limit``; return sliced list and truncation flag.

    ``truncated`` is True when there are entries past the returned window
    — the model should interpret it as "narrow the query or paginate
    with ``offset``".
    """
    total = len(entries)
    start = min(offset, total)
    if head_limit is None:
        return entries[start:], False
    end = start + head_limit
    return entries[start:end], end < total


class GrepTool(BaseTool[GrepInput, GrepResult, Any]):
    """
    Regex search via ripgrep.

    Attach via :class:`FileSearchToolkit` or instantiate directly with
    ``allowed_roots``.
    """

    name = "Grep"
    description = (
        "Search file contents with a regex, backed by ripgrep. Three "
        "output modes: ``files_with_matches`` (default — list matching "
        "files), ``content`` (matching lines with optional context), "
        "``count`` (per-file match counts). Filter files with ``glob`` "
        "(``*.py``) or ``type`` (``py``, ``js``, ``rust``). Use "
        "``case_insensitive``, ``context`` / ``before_context`` / "
        "``after_context``, and ``multiline`` as needed. Large result "
        "sets are capped by ``head_limit`` (default 250)."
    )

    def __init__(
        self,
        *,
        allowed_roots: list[Path],
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._allowed_roots = allowed_roots

    async def _run(
        self,
        inp: GrepInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> GrepResult:
        del ctx, exec_id, progress_callback

        raw_path = inp.path if inp.path is not None else str(self._allowed_roots[0])
        try:
            resolved = resolve_safe(raw_path, self._allowed_roots, must_exist=True)
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        args = _build_args(inp, resolved, inp.output_mode)
        stdout, stderr, rc = await _run_rg(args)

        # rg exit codes: 0 = matches found, 1 = no matches, 2 = error.
        if rc not in {0, 1}:
            err = stderr.decode("utf-8", errors="replace").strip() or "unknown error"
            raise GrepError(f"rg exited with code {rc}: {err}")

        offset = inp.offset or 0

        if inp.output_mode == "files_with_matches":
            files = _parse_files_with_matches(stdout)
            sliced, truncated = _slice(files, offset=offset, head_limit=inp.head_limit)
            return GrepResult(
                output="\n".join(sliced),
                output_mode=inp.output_mode,
                num_matches=len(files),
                num_files_matched=len(files),
                truncated=truncated,
            )

        if inp.output_mode == "count":
            entries = _parse_count(stdout)
            total_matches = sum(n for _, n in entries)
            rendered = [f"{p}:{n}" for p, n in entries]
            sliced, truncated = _slice(
                rendered, offset=offset, head_limit=inp.head_limit
            )
            return GrepResult(
                output="\n".join(sliced),
                output_mode=inp.output_mode,
                num_matches=total_matches,
                num_files_matched=len(entries),
                truncated=truncated,
            )

        # content mode
        lines, num_matches, paths = _parse_json_content(
            stdout, show_line_numbers=inp.show_line_numbers
        )
        sliced, truncated = _slice(lines, offset=offset, head_limit=inp.head_limit)
        return GrepResult(
            output="\n".join(sliced),
            output_mode=inp.output_mode,
            num_matches=num_matches,
            num_files_matched=len(paths),
            truncated=truncated,
        )


def rg_available() -> bool:
    """Return True if ``rg`` is on the system PATH — useful for test guards."""
    return shutil.which("rg") is not None
