"""
:class:`InMemoryFileBackend` — a :class:`FileBackend` held in a dict.

Useful as a per-session scratch mount in a :class:`CompositeFileBackend`
(spilled tool output, task logs) and as a test double. Nothing survives the
process. Directories are implicit: a directory exists when a file sits
under it, when it was created with :meth:`mkdir`, or when it is a root.
"""

from __future__ import annotations

import asyncio
import stat as stat_module
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .base import (
    FileBackend,
    FileEntry,
    FileStat,
    GrepOutputMode,
    GrepRawResult,
    StaleFileError,
)
from .local import glob_filter_entries
from .paths import PathAccessError, normalize_posix
from .text_grep import DEFAULT_GREP_TIME_BUDGET, grep_texts_async

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .paths import AccessMode

_DIR_MODE = stat_module.S_IFDIR | 0o755


@dataclass
class _File:
    data: bytes
    mtime: float
    mode: int


class InMemoryFileBackend(FileBackend):
    """
    A :class:`FileBackend` whose files live in process memory.

    Args:
        allowed_roots: Absolute POSIX directories the backend accepts paths
            under. Each root exists as an empty directory from the start.
        files: Optional seed content, path → text or bytes.
        grep_time_budget: Seconds one :meth:`grep` may search before it
            returns what it found so far with a note (see :func:`grep_texts`).

    Writes are atomic and ``expected_mtime`` checks are exact: nothing
    awaits between the check and the write. Every write gets an mtime
    strictly greater than any earlier one in this backend.

    """

    name: str = "memory"

    def __init__(
        self,
        allowed_roots: Sequence[Path | str],
        *,
        files: Mapping[str | Path, str | bytes] | None = None,
        grep_time_budget: float = DEFAULT_GREP_TIME_BUDGET,
    ) -> None:
        self._grep_time_budget = grep_time_budget
        self._allowed_roots: list[Path] = [
            Path(normalize_posix(r)) for r in allowed_roots
        ]
        for root in self._allowed_roots:
            if not PurePosixPath(root).is_absolute():
                raise ValueError(f"InMemoryFileBackend roots must be absolute: {root}")
        self._files: dict[PurePosixPath, _File] = {}
        self._dirs: set[PurePosixPath] = set()
        self._last_mtime = 0.0
        for raw_path, content in (files or {}).items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            self._files[normalize_posix(raw_path)] = _File(
                data, self._next_mtime(), 0o644
            )

    # ---- address space ----------------------------------------------------

    @property
    def allowed_roots(self) -> list[Path]:
        return list(self._allowed_roots)

    def add_allowed_root(self, root: Path) -> None:
        resolved = Path(normalize_posix(root))
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
        del access, dotfile_overrides
        if not self._allowed_roots:
            raise PathAccessError("No allowed_roots configured for in-memory backend.")
        raw = PurePosixPath(path)
        if not raw.is_absolute():
            raw = PurePosixPath(self._allowed_roots[0]) / raw
        candidate = normalize_posix(raw)
        if not any(
            candidate == PurePosixPath(r) or PurePosixPath(r) in candidate.parents
            for r in self._allowed_roots
        ):
            roots = ", ".join(str(r) for r in self._allowed_roots)
            raise PathAccessError(f"Path {path} is outside allowed roots [{roots}]")
        resolved = Path(candidate)
        if must_exist and not await self.exists(resolved):
            raise PathAccessError(f"Path does not exist: {resolved}")
        return resolved

    # ---- metadata -----------------------------------------------------------

    async def stat(self, path: Path) -> FileStat:
        key = normalize_posix(path)
        entry = self._files.get(key)
        if entry is not None:
            return FileStat(
                mtime=entry.mtime,
                mode=stat_module.S_IFREG | entry.mode,
                size=len(entry.data),
            )
        if self._is_dir(key):
            return FileStat(mtime=self._dir_mtime(key), mode=_DIR_MODE, size=0)
        raise FileNotFoundError(f"No such file or directory: {path}")

    async def exists(self, path: Path) -> bool:
        key = normalize_posix(path)
        return key in self._files or self._is_dir(key)

    async def parent_exists(self, path: Path) -> bool:
        return self._is_dir(normalize_posix(path).parent)

    # ---- content ------------------------------------------------------------

    async def read_text(self, path: Path) -> tuple[str, float]:
        data, mtime = await self.read_bytes(path)
        return data.decode("utf-8", errors="replace"), mtime

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        entry = self._require_file(normalize_posix(path))
        return entry.data, entry.mtime

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
        expected_mtime: float | None = None,
    ) -> float:
        key = normalize_posix(path)
        if self._is_dir(key):
            raise IsADirectoryError(f"Is a directory: {path}")
        current = self._files.get(key)
        if current is not None and not overwrite:
            raise FileExistsError(f"File exists: {path}")
        self._check_expected(path, current, expected_mtime)
        mtime = self._next_mtime()
        self._files[key] = _File(bytes(data), mtime, mode & 0o7777)
        return mtime

    async def append_bytes(self, path: Path, data: bytes, *, mode: int) -> float:
        key = normalize_posix(path)
        current = self._files.get(key)
        existing = current.data if current is not None else b""
        file_mode = current.mode if current is not None else mode & 0o7777
        mtime = self._next_mtime()
        self._files[key] = _File(existing + data, mtime, file_mode)
        return mtime

    async def delete(self, path: Path, *, expected_mtime: float | None = None) -> None:
        key = normalize_posix(path)
        current = self._require_file(key)
        self._check_expected(path, current, expected_mtime)
        del self._files[key]

    async def mkdir(self, path: Path) -> None:
        key = normalize_posix(path)
        if key in self._files:
            raise FileExistsError(f"A file exists at {path}")
        self._dirs.add(key)

    # ---- listing and search -------------------------------------------------

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        base = normalize_posix(path)
        if not self._is_dir(base):
            return []
        seen: dict[PurePosixPath, FileEntry] = {}
        for key in self._all_dirs() | set(self._files):
            if key == base or base not in key.parents:
                continue
            rel = key.relative_to(base)
            depth = len(rel.parts)
            if depth > 1 and not recursive:
                child = base / rel.parts[0]
                seen.setdefault(child, self._entry(child))
                continue
            seen[key] = self._entry(key)
        return sorted(seen.values(), key=lambda e: e.path.parts)

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        entries = await self.list_dir(root, recursive=True)
        return await asyncio.to_thread(
            glob_filter_entries,
            entries,
            Path(normalize_posix(root)),
            pattern,
            include_hidden=include_hidden,
            head_limit=head_limit,
        )

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
        base = normalize_posix(root)
        snapshot = [
            (key, entry.data)
            for key, entry in self._files.items()
            if key == base or base in key.parents
        ]
        return await grep_texts_async(
            (
                (Path(key), data.decode("utf-8", errors="replace"))
                for key, data in snapshot
            ),
            pattern,
            root=Path(base),
            glob=glob,
            file_type=file_type,
            case_insensitive=case_insensitive,
            multiline=multiline,
            output_mode=output_mode,
            show_line_numbers=show_line_numbers,
            before_context=before_context,
            after_context=after_context,
            context=context,
            time_budget=self._grep_time_budget,
        )

    # ---- internals ----------------------------------------------------------

    def _next_mtime(self) -> float:
        now = time.time()
        self._last_mtime = max(now, self._last_mtime + 1e-6)
        return self._last_mtime

    def _check_expected(
        self, path: Path, current: _File | None, expected_mtime: float | None
    ) -> None:
        if expected_mtime is None:
            return
        if current is None:
            raise StaleFileError(f"{path} no longer exists")
        if current.mtime != expected_mtime:
            raise StaleFileError(
                f"{path} changed (expected mtime {expected_mtime!r}, "
                f"current {current.mtime!r})"
            )

    def _require_file(self, key: PurePosixPath) -> _File:
        entry = self._files.get(key)
        if entry is not None:
            return entry
        if self._is_dir(key):
            raise IsADirectoryError(f"Is a directory: {key}")
        raise FileNotFoundError(f"No such file: {key}")

    def _roots(self) -> set[PurePosixPath]:
        return {PurePosixPath(r) for r in self._allowed_roots}

    def _all_dirs(self) -> set[PurePosixPath]:
        dirs = set(self._dirs)
        for key in self._files:
            dirs.update(key.parents)
        for d in list(self._dirs):
            dirs.update(d.parents)
        roots = self._roots()
        return {
            d for d in dirs if d in roots or any(r in d.parents for r in roots)
        } | roots

    def _is_dir(self, key: PurePosixPath) -> bool:
        if key in self._roots() or key in self._dirs:
            return True
        return any(key in f.parents for f in self._files) or any(
            key in d.parents for d in self._dirs
        )

    def _dir_mtime(self, key: PurePosixPath) -> float:
        mtimes = [e.mtime for f, e in self._files.items() if key in f.parents]
        return max(mtimes, default=0.0)

    def _entry(self, key: PurePosixPath) -> FileEntry:
        file = self._files.get(key)
        if file is not None:
            return FileEntry(
                name=key.name, path=Path(key), is_dir=False, mtime=file.mtime
            )
        return FileEntry(
            name=key.name, path=Path(key), is_dir=True, mtime=self._dir_mtime(key)
        )


__all__ = ["InMemoryFileBackend"]
