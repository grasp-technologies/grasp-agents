"""
Backend protocol + shared dataclasses for the file-edit + file-search tools.

A :class:`FileBackend` is the I/O substrate the :class:`ReadTool`,
:class:`WriteTool`, :class:`EditTool`, :class:`GlobTool`, and
:class:`GrepTool` route through. The default implementation
:class:`LocalFileBackend` (in :mod:`.local_backend`) operates on the
host filesystem; alternative implementations
(e.g. :class:`MCPFileBackend` in :mod:`.mcp_backend`) route the same
calls to a remote MCP server.

All paths flowing across this protocol are :class:`pathlib.Path`. Each
backend is free to translate Path into its own address form (POSIX
string, ``file://`` URI, etc.) internally; consumers of the protocol
work in Path exclusively.

Time is wall-clock seconds since epoch (float) for parity with
:func:`os.stat().st_mtime` — MCP backends convert ms→s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class FileStat:
    """Backend-uniform stat result."""

    mtime: float  # seconds since epoch (matches os.stat().st_mtime)
    mode: int = 0o644
    size: int = 0


@dataclass(frozen=True)
class FileEntry:
    """One entry in a directory listing."""

    name: str  # file/dir basename (no path)
    path: Path  # absolute path in the backend's address space
    is_dir: bool
    mtime: float = 0.0


GrepOutputMode = Literal["files_with_matches", "content", "count"]


@dataclass(frozen=True)
class GrepRawResult:
    """
    Backend-uniform grep output. The :class:`GrepTool` slices + renders
    these for the agent.

    Fields by mode:

    * ``files_with_matches``: ``files`` holds matching file paths.
    * ``count``: ``counts`` holds ``(path, n)`` tuples.
    * ``content``: ``lines`` holds rendered ``path:line:content`` (or
      ``path-line-context``) lines.
    """

    files: list[Path] = field(default_factory=list["Path"])
    counts: list[tuple[Path, int]] = field(default_factory=list[tuple["Path", int]])
    lines: list[str] = field(default_factory=list[str])
    num_matches: int = 0
    num_files_matched: int = 0


@runtime_checkable
class FileBackend(Protocol):
    """
    File I/O + search contract for the file tools.

    Implementations own:

    1. Path safety (sandbox containment + sensitive-path policy).
    2. Raw I/O (stat, read, write, delete, list).
    3. Search (find_files, grep) — optional; backends that don't ship a
       fast path raise :class:`NotImplementedError` so tools can surface
       a clear error.
    """

    @property
    def name(self) -> str: ...

    async def validate_path(
        self,
        path: Path,
        allowed_roots: list[Path],
        *,
        must_exist: bool,
        dotfile_overrides: Iterable[Path] | None = None,
        include_dotfiles: bool = True,
    ) -> Path:
        """
        Resolve ``path`` to an absolute, canonical Path and enforce the
        backend's safety policy. Returns the resolved Path.

        Raises:
            PathAccessError: On policy violations (out of allowed roots,
                sensitive, blocked device, etc.) or — when
                ``must_exist=True`` — missing target.

        """
        ...

    async def stat(self, path: Path) -> FileStat: ...

    async def exists(self, path: Path) -> bool: ...

    async def parent_exists(self, path: Path) -> bool: ...

    async def read_text(self, path: Path) -> tuple[str, float]:
        """Return ``(content, mtime)``. ``errors='replace'`` for utf-8."""
        ...

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        """Return ``(data, mtime)``. Used by :class:`EditTool`."""
        ...

    async def write_bytes(
        self, path: Path, data: bytes, *, mode: int, overwrite: bool = True
    ) -> float:
        """Atomically write ``data``. Returns the post-write mtime."""
        ...

    async def delete(self, path: Path) -> None: ...

    async def list_dir(
        self, path: Path, *, recursive: bool = False
    ) -> list[FileEntry]: ...

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        """
        Glob-pattern walk of ``root``. Returns ``(matched, truncated)``;
        ``matched`` is sorted newest-first by mtime and capped to
        ``head_limit``.
        """
        ...

    async def grep(
        self,
        root: Path,
        pattern: str,
        *,
        glob: str | None = None,
        file_type: str | None = None,
        case_insensitive: bool = False,
        multiline: bool = False,
        output_mode: GrepOutputMode = "files_with_matches",
        show_line_numbers: bool = True,
        before_context: int | None = None,
        after_context: int | None = None,
        context: int | None = None,
    ) -> GrepRawResult:
        """
        Regex search over file contents under ``root``. Returns the raw
        result; the tool slices + paginates per ``head_limit`` /
        ``offset``. Default raises :class:`NotImplementedError`.
        """
        ...


def normalize_backend_kwargs(*, file_type: str | None, **rest: Any) -> dict[str, Any]:
    """
    Translate the public ``type`` GrepInput field to the backend kwarg
    ``file_type``. Keeps the agent-facing schema natural while avoiding
    the Python ``type`` builtin shadow inside backends.
    """
    return {"file_type": file_type, **rest}


__all__ = [
    "FileBackend",
    "FileEntry",
    "FileStat",
    "GrepOutputMode",
    "GrepRawResult",
    "normalize_backend_kwargs",
]
