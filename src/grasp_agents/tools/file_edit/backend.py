"""
Backend protocol + shared dataclasses for the file-edit + file-search tools.

A :class:`FileBackend` is the I/O substrate the :class:`ReadTool`,
:class:`WriteTool`, :class:`EditTool`, :class:`DeleteTool`,
:class:`GlobTool`, and :class:`GrepTool` route through. The default
:class:`LocalFileBackend` (in :mod:`.local_backend`) operates on the host
filesystem; alternative implementations (e.g. :class:`MCPFileBackend` in
:mod:`.mcp_backend`) route the same calls to a remote MCP server.

A backend owns:

* its own :class:`allowed_roots` — the address space it accepts paths in;
* path-safety policy (sandbox containment + sensitive-path deny list).

Read-before-write bookkeeping lives on the *agent*, not the backend —
each :class:`AgentLoop` carries its own :class:`FileEditSessionState`
and tools consult the active state via :mod:`.agent_state` before
deciding whether a write is allowed.

Hosts wire one backend instance onto :attr:`RunContext.file_backend`
and the tools route every call through it. There is no static
``allowed_roots`` plumbing on the tools or toolkits.

All paths flowing across this protocol are :class:`pathlib.Path`. Each
backend is free to translate Path into its own address form (POSIX
string, ``file://`` URI, etc.) internally; consumers of the protocol
work in Path exclusively.

Time is wall-clock seconds since epoch (float) for parity with
:func:`os.stat().st_mtime` — MCP backends convert ms→s.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from .paths import AccessMode


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


class FileBackend(ABC):
    """
    File I/O + search contract for the file tools.

    Implementations own:

    1. Path safety (sandbox containment + sensitive-path policy).
    2. Raw I/O (stat, read, write, delete, list).
    3. Search (find_files, grep) — backends that don't ship a fast path
       raise :class:`NotImplementedError` so tools can surface a clear
       error.

    Read-before-write enforcement lives on the *agent*: tools consult
    the active :class:`FileEditSessionState` via the call's ``AgentContext``
    and record reads/writes there directly. Backends are pure I/O.
    """

    name: str  # a class attr or a property both satisfy this

    @property
    @abstractmethod
    def allowed_roots(self) -> list[Path]: ...

    @abstractmethod
    def add_allowed_root(self, root: Path) -> None:
        """
        Widen the backend's address space to include ``root``.

        Idempotent — a no-op when ``root`` is already an allowed root or
        nested under one. Called by the :class:`RunContext` validator to
        admit a configured memory directory automatically, so memory
        authoring through the file tools works without the host repeating
        the memdir in ``allowed_roots``.
        """
        ...

    @abstractmethod
    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        """
        Resolve ``path`` to an absolute, canonical Path and enforce the
        backend's safety policy. Returns the resolved Path.

        ``access`` (``"read"`` / ``"write"``) selects which policy carve-outs
        apply — ``deny_read`` / ``allow_read`` for reads, ``deny_write`` for
        writes. Mutating tools pass ``"write"``.

        ``dotfile_overrides`` (if provided) is a set of resolved paths
        the caller has explicitly whitelisted for this run, bypassing
        the local-FS credential-dotfile deny list. Remote backends
        ignore it (sandbox policy lives server-side).

        Raises:
            PathAccessError: On policy violations (out of allowed roots,
                sensitive, blocked device, etc.) or — when
                ``must_exist=True`` — missing target.

        """
        ...

    @abstractmethod
    async def stat(self, path: Path) -> FileStat: ...

    @abstractmethod
    async def exists(self, path: Path) -> bool: ...

    @abstractmethod
    async def parent_exists(self, path: Path) -> bool: ...

    @abstractmethod
    async def read_text(self, path: Path) -> tuple[str, float]:
        """
        Return ``(content, mtime)``. ``errors='replace'`` for utf-8.

        Read-before-write bookkeeping happens in the caller — tools
        consult :func:`get_current_file_edit_state` and record the read
        themselves.
        """
        ...

    @abstractmethod
    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        """Return ``(data, mtime)``. Used by :class:`EditTool`."""
        ...

    @abstractmethod
    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
    ) -> float:
        """
        Atomically write ``data``. Returns the post-write mtime.

        The caller refreshes its :class:`FileEditSessionState` read
        record with the returned mtime so a following :class:`EditTool`
        doesn't trip on its own write.
        """
        ...

    @abstractmethod
    async def delete(self, path: Path) -> None:
        """Remove ``path``. Callers clear their read record separately."""
        ...

    @abstractmethod
    async def mkdir(self, path: Path) -> None:
        """
        Create directory ``path`` and any missing parents. Idempotent — no
        error if it already exists. Used to bootstrap store layouts (e.g.
        the memdir before its index file is first written). Backends with a
        flat or implicit namespace (e.g. MCP resources) may no-op.
        """
        ...

    @abstractmethod
    async def list_dir(
        self, path: Path, *, recursive: bool = False
    ) -> list[FileEntry]: ...

    @abstractmethod
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

    @abstractmethod
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
