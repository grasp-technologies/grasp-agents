"""
:class:`LocalFileBackend` — default :class:`FileBackend` on the host
filesystem.

Holds the path-safety guards (sandbox roots, sensitive-path deny list,
device-path block) — the tools call :meth:`validate_path` before any
I/O. Search delegates to ``rg`` for grep (see :mod:`..file_search.grep`)
and ``os.walk`` for find_files.

Read-before-write bookkeeping lives on the *agent* (each
:class:`AgentLoop` owns its own :class:`FileEditSessionState`); the
backend itself is pure I/O.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .atomic_write import atomic_write_bytes
from .backend import FileBackend, FileEntry, FileStat, GrepOutputMode, GrepRawResult
from .paths import (
    PathAccessError,
    check_access_path,
    check_sensitive_path,
    is_blocked_device,
    resolve_safe,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .paths import AccessMode


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


class LocalFileBackend(FileBackend):
    """
    Default :class:`FileBackend` operating on the host filesystem.

    Args:
        allowed_roots: Directories the backend will accept paths under.
            Defaults to ``[Path.cwd()]``.
        include_dotfiles: If True (default), the sensitive-path deny list
            adds common credential-dotfile patterns (``.env``, ``~/.ssh``,
            etc.) on top of the system-path baseline.

    """

    name: str = "local"

    def __init__(
        self,
        *,
        allowed_roots: list[Path | str] | None = None,
        include_dotfiles: bool = True,
        deny_read: list[Path | str] | None = None,
        allow_read: list[Path | str] | None = None,
        deny_write: list[Path | str] | None = None,
    ) -> None:
        if allowed_roots is None:
            roots: list[Path] = [Path.cwd()]
        else:
            roots = [Path(r) for r in allowed_roots]
        self._allowed_roots: list[Path] = roots
        self._include_dotfiles = include_dotfiles
        self._deny_read = self._resolve_carveouts(deny_read)
        self._allow_read = self._resolve_carveouts(allow_read)
        self._deny_write = self._resolve_carveouts(deny_write)

    @staticmethod
    def _resolve_carveouts(paths: list[Path | str] | None) -> tuple[Path, ...]:
        if not paths:
            return ()
        return tuple(Path(p).expanduser().resolve() for p in paths)

    @property
    def allowed_roots(self) -> list[Path]:
        return list(self._allowed_roots)

    def add_allowed_root(self, root: Path) -> None:
        resolved = Path(root).expanduser()
        if any(resolved == r or r in resolved.parents for r in self._allowed_roots):
            return
        self._allowed_roots.append(resolved)

    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        if is_blocked_device(path):
            raise PathAccessError(
                f"Cannot access device path {path}: blocks or produces infinite output."
            )

        resolved = resolve_safe(path, self._allowed_roots, must_exist=must_exist)

        err = check_sensitive_path(
            resolved,
            include_dotfiles=self._include_dotfiles,
            session_overrides=dotfile_overrides,
        )
        if err is not None:
            raise PathAccessError(err)

        access_err = check_access_path(
            resolved,
            access=access,
            deny_read=self._deny_read,
            allow_read=self._allow_read,
            deny_write=self._deny_write,
        )
        if access_err is not None:
            raise PathAccessError(access_err)

        return resolved

    async def stat(self, path: Path) -> FileStat:
        st = await asyncio.to_thread(path.stat)
        # Keep the full mode (type bits + permissions). The 0o7777 mask
        # was applied here previously which stripped S_IFDIR / S_IFREG —
        # callers that need just the permissions can mask themselves.
        return FileStat(mtime=st.st_mtime, mode=st.st_mode, size=st.st_size)

    async def exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    async def parent_exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.parent.exists)

    async def read_text(self, path: Path) -> tuple[str, float]:
        def _read() -> tuple[str, float]:
            resolved = path.resolve(strict=True)
            mtime = resolved.stat().st_mtime
            text = resolved.read_text(encoding="utf-8", errors="replace")
            return text, mtime

        return await asyncio.to_thread(_read)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        def _read() -> tuple[bytes, float]:
            resolved = path.resolve(strict=True)
            mtime = resolved.stat().st_mtime
            return resolved.read_bytes(), mtime

        return await asyncio.to_thread(_read)

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
    ) -> float:
        def _write() -> float:
            atomic_write_bytes(path, data, mode=mode, overwrite=overwrite)
            # ``resolve`` after the write so a brand-new file's parent
            # symlinks (if any) follow the same canonicalization as
            # reads.
            resolved = path.resolve(strict=True)
            return resolved.stat().st_mtime

        return await asyncio.to_thread(_write)

    async def delete(self, path: Path) -> None:
        await asyncio.to_thread(path.unlink)

    async def mkdir(self, path: Path) -> None:
        resolved = await self.validate_path(path, must_exist=False, access="write")
        await asyncio.to_thread(lambda: resolved.mkdir(parents=True, exist_ok=True))

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        def _walk() -> list[FileEntry]:
            if not path.is_dir():
                return []
            entries: list[FileEntry] = []
            paths_iter = path.rglob("*") if recursive else path.iterdir()
            for p in paths_iter:
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append(
                    FileEntry(
                        name=p.name,
                        path=p,
                        is_dir=p.is_dir(),
                        mtime=st.st_mtime,
                    )
                )
            return entries

        return await asyncio.to_thread(_walk)

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        def _walk() -> tuple[list[FileEntry], bool]:
            matched: list[FileEntry] = []
            collect_budget = head_limit + 1
            truncated = False
            root_str = str(root)
            for dirpath, dirnames, filenames in os.walk(root_str):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _ALWAYS_SKIP_DIRS
                    and (include_hidden or not d.startswith("."))
                ]
                for fname in filenames:
                    if not include_hidden and fname.startswith("."):
                        continue
                    abs_path = Path(dirpath) / fname
                    rel_path = str(abs_path.relative_to(root))
                    if _glob_matches(pattern, rel_path):
                        try:
                            mtime = abs_path.stat().st_mtime
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


def glob_filter_entries(
    entries: Iterable[FileEntry],
    root: Path,
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
    matched: list[FileEntry] = []
    truncated = False
    collect_budget = head_limit + 1
    for entry in entries:
        if entry.is_dir:
            continue
        try:
            rel = entry.path.relative_to(root)
        except ValueError:
            continue
        rel_str = str(rel)
        if not include_hidden and any(part.startswith(".") for part in rel.parts):
            continue
        if not _glob_matches(pattern, rel_str):
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
