"""
:class:`LocalFileBackend` — default :class:`FileBackend` on the host
filesystem.

Holds the path-safety guards (sandbox roots, sensitive-path deny list,
device-path block) — the tools call :meth:`validate_path` before any
I/O. Search delegates to ``rg`` for grep (see :mod:`..file_search.grep`)
and ``os.walk`` for find_files.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .atomic_write import atomic_write_bytes
from .backend import FileEntry, FileStat, GrepOutputMode, GrepRawResult
from .paths import (
    PathAccessError,
    check_sensitive_path,
    is_blocked_device,
    resolve_safe,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        path: Path,
        allowed_roots: list[Path],
        *,
        must_exist: bool,
        dotfile_overrides: Iterable[Path] | None = None,
        include_dotfiles: bool = True,
    ) -> Path:
        if is_blocked_device(path):
            raise PathAccessError(
                f"Cannot access device path {path}: blocks or produces "
                "infinite output."
            )

        resolved = resolve_safe(path, allowed_roots, must_exist=must_exist)

        override_paths: set[Path] | None = (
            set(dotfile_overrides) if dotfile_overrides else None
        )

        err = check_sensitive_path(
            resolved,
            include_dotfiles=include_dotfiles,
            session_overrides=override_paths,
        )
        if err is not None:
            raise PathAccessError(err)

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
            mtime = path.stat().st_mtime
            return path.read_text(encoding="utf-8", errors="replace"), mtime

        return await asyncio.to_thread(_read)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        def _read() -> tuple[bytes, float]:
            mtime = path.stat().st_mtime
            return path.read_bytes(), mtime

        return await asyncio.to_thread(_read)

    async def write_bytes(
        self, path: Path, data: bytes, *, mode: int, overwrite: bool = True
    ) -> float:
        def _write() -> float:
            atomic_write_bytes(path, data, mode=mode, overwrite=overwrite)
            return path.stat().st_mtime

        return await asyncio.to_thread(_write)

    async def delete(self, path: Path) -> None:
        await asyncio.to_thread(path.unlink)

    async def list_dir(
        self, path: Path, *, recursive: bool = False
    ) -> list[FileEntry]:
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
        if not include_hidden and any(
            part.startswith(".") for part in rel.parts
        ):
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
