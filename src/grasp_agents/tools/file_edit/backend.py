"""
Backends for the file-edit + file-search tools.

A :class:`FileBackend` is the I/O substrate the :class:`ReadTool`,
:class:`WriteTool`, :class:`EditTool`, :class:`GlobTool`, and
:class:`GrepTool` route through. The default :class:`LocalFileBackend`
operates on the host filesystem; alternative implementations (e.g.
:class:`MCPFileBackend`) route the same calls to a remote MCP server
speaking the file-tool protocol.

Tools depend only on the protocol — `is_blocked_device`,
`check_sensitive_path`, and `resolve_safe` are local-FS concerns and
live inside :class:`LocalFileBackend.validate_path`; the MCP backend
trusts its server's containment policy.

Path identity is a UTF-8 string (resolved-absolute in the backend's own
address space). Time is wall-clock seconds since epoch (float) for
parity with :func:`os.stat().st_mtime` — MCP backends convert ms→s.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .atomic_write import atomic_write_bytes
from .paths import (
    PathAccessError,
    check_sensitive_path,
    is_blocked_device,
    resolve_safe,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class FileStat:
    """Backend-uniform stat result."""

    mtime: float  # seconds since epoch (matches os.stat().st_mtime)
    mode: int = 0o644
    size: int = 0


@dataclass(frozen=True)
class FileEntry:
    """One entry in a directory listing."""

    name: str  # file/dir name (no path)
    path: str  # absolute path in the backend's address space
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

    files: list[str] = field(default_factory=list[str])
    counts: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
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
    3. Search (find_files, grep) — optional; default raises
       :class:`NotImplementedError` so tools can surface a clear error
       when the backend doesn't ship a fast path.
    """

    @property
    def name(self) -> str: ...

    async def validate_path(
        self,
        path: str,
        allowed_roots: list[str],
        *,
        must_exist: bool,
        dotfile_overrides: Iterable[str] | None = None,
        include_dotfiles: bool = True,
    ) -> str:
        """
        Resolve ``path`` to an absolute, canonical form and enforce
        backend-appropriate safety policy. Returns the resolved path.

        Raises:
            PathAccessError: On policy violations (out of allowed roots,
                sensitive, blocked device, etc.) or — when
                ``must_exist=True`` — missing target.

        """
        ...

    async def stat(self, path: str) -> FileStat: ...

    async def exists(self, path: str) -> bool: ...

    async def parent_exists(self, path: str) -> bool: ...

    async def read_text(self, path: str) -> tuple[str, float]:
        """Return ``(content, mtime)``. ``errors='replace'`` for utf-8."""
        ...

    async def read_bytes(self, path: str) -> tuple[bytes, float]:
        """Return ``(data, mtime)``. Used by :class:`EditTool`."""
        ...

    async def write_bytes(
        self, path: str, data: bytes, *, mode: int, overwrite: bool = True
    ) -> float:
        """Atomically write ``data``. Returns the post-write mtime."""
        ...

    async def delete(self, path: str) -> None: ...

    async def list_dir(
        self, path: str, *, recursive: bool = False
    ) -> list[FileEntry]: ...

    async def find_files(
        self,
        root: str,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        """
        Glob-pattern walk of ``root``. Returns ``(matched, truncated)``;
        ``matched`` is sorted newest-first by mtime and capped to
        ``head_limit``. Default raises
        :class:`NotImplementedError`.
        """
        ...

    async def grep(
        self,
        root: str,
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


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


# Directory names always skipped on local Glob traversal — common
# build/cache families that bloat results without informing the model.
_ALWAYS_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


def _glob_matches(pattern: str, rel_path: str) -> bool:
    """Match ``rel_path`` against ``pattern`` with ``**`` semantics."""
    norm = rel_path.replace(os.sep, "/")
    if "**" not in pattern:
        return fnmatch.fnmatchcase(norm, pattern)
    return _match_segments(pattern.split("/"), norm.split("/"))


def _match_segments(pattern_parts: list[str], path_parts: list[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    head, *tail = pattern_parts
    if head == "**":
        if not tail:
            return True
        for i in range(len(path_parts) + 1):
            if _match_segments(tail, path_parts[i:]):
                return True
        return False
    if not path_parts:
        return False
    if fnmatch.fnmatchcase(path_parts[0], head):
        return _match_segments(tail, path_parts[1:])
    return False


class LocalFileBackend:
    """
    Default :class:`FileBackend` operating on the host filesystem.

    All path-safety guards (sandbox roots, sensitive-path deny list,
    device-path block) live here; the tools just call
    :meth:`validate_path` before reading or writing.
    """

    name: str = "local"

    async def validate_path(
        self,
        path: str,
        allowed_roots: list[str],
        *,
        must_exist: bool,
        dotfile_overrides: Iterable[str] | None = None,
        include_dotfiles: bool = True,
    ) -> str:
        if is_blocked_device(path):
            raise PathAccessError(
                f"Cannot access device path {path!r}: blocks or produces "
                "infinite output."
            )

        roots = [Path(r) for r in allowed_roots]
        resolved = resolve_safe(path, roots, must_exist=must_exist)

        override_paths: set[Path] | None = None
        if dotfile_overrides:
            override_paths = {Path(p) for p in dotfile_overrides}

        err = check_sensitive_path(
            resolved,
            include_dotfiles=include_dotfiles,
            session_overrides=override_paths,
        )
        if err is not None:
            raise PathAccessError(err)

        return str(resolved)

    async def stat(self, path: str) -> FileStat:
        st = await asyncio.to_thread(os.stat, path)
        # Keep the full mode (type bits + permissions). The 0o7777 mask
        # was applied here previously which stripped S_IFDIR / S_IFREG —
        # callers that need just the permissions can mask themselves.
        return FileStat(mtime=st.st_mtime, mode=st.st_mode, size=st.st_size)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)

    async def parent_exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).parent.exists)

    async def read_text(self, path: str) -> tuple[str, float]:
        def _read() -> tuple[str, float]:
            p = Path(path)
            mtime = p.stat().st_mtime
            return p.read_text(encoding="utf-8", errors="replace"), mtime

        return await asyncio.to_thread(_read)

    async def read_bytes(self, path: str) -> tuple[bytes, float]:
        def _read() -> tuple[bytes, float]:
            p = Path(path)
            mtime = p.stat().st_mtime
            return p.read_bytes(), mtime

        return await asyncio.to_thread(_read)

    async def write_bytes(
        self, path: str, data: bytes, *, mode: int, overwrite: bool = True
    ) -> float:
        def _write() -> float:
            atomic_write_bytes(
                Path(path), data, mode=mode, overwrite=overwrite
            )
            return Path(path).stat().st_mtime

        return await asyncio.to_thread(_write)

    async def delete(self, path: str) -> None:
        await asyncio.to_thread(Path(path).unlink)

    async def list_dir(
        self, path: str, *, recursive: bool = False
    ) -> list[FileEntry]:
        def _walk() -> list[FileEntry]:
            root = Path(path)
            if not root.is_dir():
                return []
            entries: list[FileEntry] = []
            paths_iter = root.rglob("*") if recursive else root.iterdir()
            for p in paths_iter:
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append(
                    FileEntry(
                        name=p.name,
                        path=str(p),
                        is_dir=p.is_dir(),
                        mtime=st.st_mtime,
                    )
                )
            return entries

        return await asyncio.to_thread(_walk)

    async def find_files(
        self,
        root: str,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        def _walk() -> tuple[list[FileEntry], bool]:
            matched: list[FileEntry] = []
            collect_budget = head_limit + 1
            truncated = False
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _ALWAYS_SKIP_DIRS
                    and (include_hidden or not d.startswith("."))
                ]
                for fname in filenames:
                    if not include_hidden and fname.startswith("."):
                        continue
                    abs_path = os.path.join(dirpath, fname)  # noqa: PTH118
                    rel_path = os.path.relpath(abs_path, root)
                    if _glob_matches(pattern, rel_path):
                        try:
                            mtime = os.stat(abs_path).st_mtime  # noqa: PTH116
                        except OSError:
                            continue
                        matched.append(
                            FileEntry(
                                name=fname,
                                path=abs_path,
                                is_dir=False,
                                mtime=mtime,
                            )
                        )
                        if len(matched) >= collect_budget:
                            truncated = True
                            return matched, truncated
            return matched, truncated

        return await asyncio.to_thread(_walk)

    async def grep(
        self,
        root: str,
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
        # Defer the heavy rg-driving helper to keep this module slim and
        # avoid a circular import from ``..file_search``.
        from ..file_search.grep import (  # noqa: PLC0415
            local_backend_grep,
        )

        return await local_backend_grep(
            root=root,
            pattern=pattern,
            glob=glob,
            file_type=file_type,
            case_insensitive=case_insensitive,
            multiline=multiline,
            output_mode=output_mode,
            show_line_numbers=show_line_numbers,
            before_context=before_context,
            after_context=after_context,
            context=context,
        )


# ---------------------------------------------------------------------------
# Helpers reused by alternate backends
# ---------------------------------------------------------------------------


def glob_filter_entries(
    entries: Iterable[FileEntry],
    root: str,
    pattern: str,
    *,
    include_hidden: bool,
    head_limit: int,
) -> tuple[list[FileEntry], bool]:
    """
    Filter a ``list_dir(recursive=True)`` result by glob pattern.

    Shared between any backend that falls back to ``list_dir`` instead
    of providing its own fast path. Sorts newest-first by mtime and
    caps at ``head_limit``.
    """
    norm_root = root.rstrip("/") + "/"
    matched: list[FileEntry] = []
    truncated = False
    collect_budget = head_limit + 1
    for entry in entries:
        if entry.is_dir:
            continue
        if not entry.path.startswith(norm_root):
            continue
        rel = entry.path[len(norm_root) :]
        if not include_hidden and any(
            part.startswith(".") for part in rel.split("/")
        ):
            continue
        if not _glob_matches(pattern, rel):
            continue
        matched.append(entry)
        if len(matched) >= collect_budget:
            truncated = True
            break
    matched.sort(key=lambda e: e.mtime, reverse=True)
    if len(matched) > head_limit:
        matched = matched[:head_limit]
        truncated = True
    return matched, truncated


def normalize_backend_kwargs(*, file_type: str | None, **rest: Any) -> dict[str, Any]:
    """
    Translate the public ``type`` GrepInput field to the backend kwarg
    ``file_type``. Keeps the agent-facing schema natural while avoiding
    the Python ``type`` builtin shadow inside backends.
    """
    return {"file_type": file_type, **rest}
