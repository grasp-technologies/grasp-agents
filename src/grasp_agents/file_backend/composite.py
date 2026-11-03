"""
:class:`CompositeFileBackend` — one file tree served by several backends.

Each :class:`Mount` attaches a child :class:`FileBackend` at an absolute
path. The composite routes every call to the mount whose path contains the
target, so the file tools see a single tree, e.g.::

    /user/memory      -> a database-backed backend
    /user/notes       -> the same, read-only for this agent
    /scratch          -> InMemoryFileBackend (spills, task logs)
    /home/user/work   -> the E2B sandbox's own file backend

Rules:

* Paths are never translated. A child receives the same absolute path the
  tools use, so each child must accept its mount path (it must equal or sit
  under one of the child's ``allowed_roots``). For a mounted sandbox backend
  this keeps the file tools and ``Bash`` agreeing on every path. Mount local
  directories at their resolved (symlink-free) paths.
* Mounts never nest. Strict ancestors of mount paths (``/`` and
  ``/user`` above) are virtual directories: they can be listed, globbed
  and grepped, which fans out to the mounts below, but never written.
* A read-only mount refuses writes, both in :meth:`validate_path` with
  ``access="write"`` and in the mutating calls themselves.
* ``allowed_roots`` lists the ``scratch`` mount first. Framework writes
  that use ``allowed_roots[0]`` (spilled tool output, background task logs)
  and path-less ``Glob`` / ``Grep`` calls therefore land on scratch rather
  than in, say, a user's memory.
"""

from __future__ import annotations

import asyncio
import logging
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .base import FileBackend, FileEntry, FileStat, GrepOutputMode, GrepRawResult
from .local import glob_filter_entries
from .paths import PathAccessError, normalize_posix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .paths import AccessMode

logger = logging.getLogger(__name__)

_VIRTUAL_DIR_MODE = stat_module.S_IFDIR | 0o755


@dataclass(frozen=True)
class Mount:
    """
    A child backend attached at ``path`` in a :class:`CompositeFileBackend`.

    ``description`` tells the agent what lives there; the mounts
    system-prompt section renders it.
    """

    path: Path
    backend: FileBackend
    read_only: bool = False
    description: str = ""


class CompositeFileBackend(FileBackend):
    """
    Route file calls to child backends by mount path.

    Args:
        mounts: The mounts, in the order the prompt lists them. Paths must be
            absolute, lexically normalized, unique and not nested.
        scratch: The mount path that receives framework writes (spilled tool
            output, task logs) and serves as the default search root. It
            must be one of the mount paths and writable. Defaults to the
            first writable mount.

    Raises:
        ValueError: A mount or the scratch path breaks the rules above, or
            no mount is writable.

    """

    name: str = "composite"

    def __init__(self, mounts: Sequence[Mount], *, scratch: Path | None = None) -> None:
        if not mounts:
            raise ValueError("CompositeFileBackend needs at least one mount.")
        paths: list[PurePosixPath] = []
        for mount in mounts:
            raw = PurePosixPath(mount.path)
            if not raw.is_absolute():
                raise ValueError(f"Mount path must be absolute: {mount.path}")
            if raw != normalize_posix(raw):
                raise ValueError(f"Mount path must be normalized: {mount.path}")
            if raw in paths:
                raise ValueError(f"duplicate mount path: {mount.path}")
            if not any(
                raw == PurePosixPath(r) or PurePosixPath(r) in raw.parents
                for r in mount.backend.allowed_roots
            ):
                roots = ", ".join(str(r) for r in mount.backend.allowed_roots)
                raise ValueError(
                    f"Backend {mount.backend.name!r} does not serve mount path "
                    f"{mount.path} (its roots: [{roots}])."
                )
            paths.append(raw)
        for outer in paths:
            for inner in paths:
                if outer in inner.parents:
                    raise ValueError(
                        f"nested mounts are not supported: {inner} is inside {outer}"
                    )

        self._mounts: tuple[Mount, ...] = tuple(mounts)
        if scratch is None:
            writable = next((m for m in self._mounts if not m.read_only), None)
            if writable is None:
                raise ValueError(
                    "CompositeFileBackend needs a writable mount for scratch "
                    "(spilled tool output, task logs); every mount is read-only."
                )
            self._scratch = writable
        else:
            match = self._mount_at(normalize_posix(scratch))
            if match is None:
                raise ValueError(f"scratch {scratch} is not one of the mount paths.")
            if match.read_only:
                raise ValueError(f"scratch mount {scratch} must be writable.")
            self._scratch = match

    # ---- mount table ------------------------------------------------------

    @property
    def mounts(self) -> tuple[Mount, ...]:
        """The mounts in declared order."""
        return self._mounts

    @property
    def scratch(self) -> Mount:
        """The mount that receives framework writes."""
        return self._scratch

    def mount_for(self, path: Path) -> Mount | None:
        """The mount containing ``path`` (lexically normalized), or None."""
        return self._mount_containing(normalize_posix(path))

    @property
    def allowed_roots(self) -> list[Path]:
        others = [m.path for m in self._mounts if m is not self._scratch]
        return [self._scratch.path, *others]

    def add_allowed_root(self, root: Path) -> None:
        if self.mount_for(Path(root)) is not None:
            return
        raise PathAccessError(
            f"{root} is covered by no mount, so this composite cannot serve it. "
            f"Mounts: {self._mount_list()}"
        )

    def includes(self, backend: FileBackend) -> bool:
        return backend is self or any(m.backend.includes(backend) for m in self._mounts)

    # ---- validation ---------------------------------------------------------

    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        candidate = self._absolute(path)
        mount = self._mount_containing(candidate)
        if mount is None:
            if self._is_virtual_dir(candidate):
                if access == "write":
                    raise PathAccessError(
                        f"{candidate} is not inside a mount; it only groups the "
                        f"mounts below it. Mounts: {self._mount_list()}"
                    )
                return Path(str(candidate))
            raise PathAccessError(
                f"Path {path} is outside every mount. Mounts: {self._mount_list()}"
            )
        if access == "write" and mount.read_only:
            raise PathAccessError(
                f"{mount.path} is mounted read-only; {candidate} cannot be changed."
            )
        resolved = await mount.backend.validate_path(
            Path(str(candidate)),
            must_exist=must_exist,
            access=access,
            dotfile_overrides=dotfile_overrides,
        )
        if self._mount_containing(normalize_posix(resolved)) is not mount:
            raise PathAccessError(
                f"{path} resolves to {resolved}, outside its mount {mount.path}."
            )
        return resolved

    # ---- metadata -------------------------------------------------------------

    async def stat(self, path: Path) -> FileStat:
        target = normalize_posix(path)
        mount = self._mount_containing(target)
        if mount is None:
            if self._is_virtual_dir(target):
                return FileStat(mtime=0.0, mode=_VIRTUAL_DIR_MODE, size=0)
            raise FileNotFoundError(f"No such file or directory: {path}")
        return await mount.backend.stat(Path(str(target)))

    async def exists(self, path: Path) -> bool:
        target = normalize_posix(path)
        mount = self._mount_containing(target)
        if mount is None:
            return self._is_virtual_dir(target)
        return await mount.backend.exists(Path(str(target)))

    async def parent_exists(self, path: Path) -> bool:
        target = normalize_posix(path)
        if self._is_virtual_dir(target.parent):
            return True
        mount = self._mount_containing(target.parent)
        if mount is None:
            return False
        return await mount.backend.parent_exists(Path(str(target)))

    # ---- content --------------------------------------------------------------

    async def read_text(self, path: Path) -> tuple[str, float]:
        mount, target = self._route_file(path)
        return await mount.backend.read_text(target)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        mount, target = self._route_file(path)
        return await mount.backend.read_bytes(target)

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
        expected_mtime: float | None = None,
    ) -> float:
        mount, target = self._route_write(path)
        return await mount.backend.write_bytes(
            target, data, mode=mode, overwrite=overwrite, expected_mtime=expected_mtime
        )

    async def append_bytes(self, path: Path, data: bytes, *, mode: int) -> float:
        mount, target = self._route_write(path)
        return await mount.backend.append_bytes(target, data, mode=mode)

    async def delete(self, path: Path, *, expected_mtime: float | None = None) -> None:
        mount, target = self._route_write(path)
        await mount.backend.delete(target, expected_mtime=expected_mtime)

    async def mkdir(self, path: Path) -> None:
        # A virtual directory always exists; a mount's own root may not exist
        # in its backend yet (a new local memdir), so it is passed down.
        if self._is_virtual_dir(normalize_posix(path)):
            return
        mount, routed = self._route_write(path)
        await mount.backend.mkdir(routed)

    # ---- listing and search ---------------------------------------------------

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        target = normalize_posix(path)
        mount = self._mount_containing(target)
        if mount is not None:
            return await mount.backend.list_dir(Path(str(target)), recursive=recursive)
        if not self._is_virtual_dir(target):
            return []

        entries: dict[PurePosixPath, FileEntry] = {}
        for below in self._mounts_below(target):
            rel = PurePosixPath(below.path).relative_to(target)
            depth = len(rel.parts) if recursive else 1
            for k in range(1, depth + 1):
                d = target.joinpath(*rel.parts[:k])
                entries.setdefault(
                    d, FileEntry(name=d.name, path=Path(str(d)), is_dir=True, mtime=0.0)
                )
            if not recursive:
                continue
            try:
                listed = await below.backend.list_dir(below.path, recursive=True)
            except (NotImplementedError, OSError):
                logger.warning("Could not list mount %s", below.path, exc_info=True)
                continue
            for entry in listed:
                entries[PurePosixPath(entry.path)] = entry
        return sorted(entries.values(), key=lambda e: e.path.parts)

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        target = normalize_posix(root)
        mount = self._mount_containing(target)
        if mount is not None:
            return await mount.backend.find_files(
                Path(str(target)),
                pattern,
                include_hidden=include_hidden,
                head_limit=head_limit,
            )
        entries = await self.list_dir(Path(str(target)), recursive=True)
        return await asyncio.to_thread(
            glob_filter_entries,
            entries,
            Path(str(target)),
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
        target = normalize_posix(root)

        async def run(backend: FileBackend, at: Path) -> GrepRawResult:
            return await backend.grep(
                at,
                pattern,
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

        mount = self._mount_containing(target)
        if mount is not None:
            return await run(mount.backend, Path(str(target)))

        below = sorted(self._mounts_below(target), key=lambda m: m.path.parts)
        outcomes = await asyncio.gather(
            *(run(m.backend, m.path) for m in below), return_exceptions=True
        )
        results: list[GrepRawResult] = []
        notes: list[str] = []
        failures: list[BaseException] = []
        for m, outcome in zip(below, outcomes, strict=True):
            if isinstance(outcome, GrepRawResult):
                results.append(outcome)
                notes.extend(outcome.notes)
            elif isinstance(outcome, (NotImplementedError, OSError)):
                failures.append(outcome)
                notes.append(f"{m.path} was not searched: {outcome}")
            else:
                raise outcome
        if failures and not results:
            raise failures[0]
        return GrepRawResult(
            files=[f for r in results for f in r.files],
            counts=[c for r in results for c in r.counts],
            lines=[line for r in results for line in r.lines],
            num_matches=sum(r.num_matches for r in results),
            num_files_matched=sum(r.num_files_matched for r in results),
            notes=notes,
        )

    # ---- internals ------------------------------------------------------------

    def _absolute(self, path: Path) -> PurePosixPath:
        raw = PurePosixPath(path)
        if not raw.is_absolute():
            raw = PurePosixPath(self._scratch.path) / raw
        return normalize_posix(raw)

    def _mount_at(self, target: PurePosixPath) -> Mount | None:
        return next((m for m in self._mounts if PurePosixPath(m.path) == target), None)

    def _mount_containing(self, target: PurePosixPath) -> Mount | None:
        for mount in self._mounts:
            mp = PurePosixPath(mount.path)
            if target == mp or mp in target.parents:
                return mount
        return None

    def _mounts_below(self, target: PurePosixPath) -> list[Mount]:
        return [m for m in self._mounts if target in PurePosixPath(m.path).parents]

    def _is_virtual_dir(self, target: PurePosixPath) -> bool:
        return bool(self._mounts_below(target))

    def _route_file(self, path: Path) -> tuple[Mount, Path]:
        target = normalize_posix(path)
        mount = self._mount_containing(target)
        if mount is None:
            if self._is_virtual_dir(target):
                raise IsADirectoryError(f"Is a directory: {path}")
            raise FileNotFoundError(f"No such file: {path}")
        return mount, Path(str(target))

    def _route_write(self, path: Path) -> tuple[Mount, Path]:
        target = normalize_posix(path)
        mount = self._mount_containing(target)
        if mount is None:
            raise PathAccessError(
                f"{path} is not inside a mount. Mounts: {self._mount_list()}"
            )
        if mount.read_only:
            raise PathAccessError(
                f"{mount.path} is mounted read-only; {path} cannot be changed."
            )
        return mount, Path(str(target))

    def _mount_list(self) -> str:
        return ", ".join(
            f"{m.path}{' (read-only)' if m.read_only else ''}" for m in self._mounts
        )


__all__ = ["CompositeFileBackend", "Mount"]
