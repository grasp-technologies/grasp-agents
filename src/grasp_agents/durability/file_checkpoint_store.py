"""
Filesystem-backed :class:`CheckpointStore`.

Each key maps to one JSON file under ``root``. Keys are ``/``-separated;
each segment becomes a directory component, with ``.json`` appended to
the final segment. For example:

- ``"agent/session-abc"``           → ``<root>/agent/session-abc.json``
- ``"workflow/session/subproc"``    → ``<root>/workflow/session/subproc.json``
- ``"task/session-abc/task-42"``    → ``<root>/task/session-abc/task-42.json``

Writes are atomic: the payload lands in a uniquely-named tempfile in the
same directory, then :func:`os.replace` swaps it into place. Concurrent
writes to the same key serialize via a per-key :class:`asyncio.Lock`.

No TTL / GC at this layer — retention policies are the caller's concern.

Crash safety: ``os.replace`` is atomic rename on POSIX (and on Windows
for same-volume moves), so post-crash you see either the old file or the
new file, never a partial write. Orphaned ``*.tmp`` files may be left
behind on a hard crash; they are harmless and can be swept separately.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint_store import CheckpointStore

if TYPE_CHECKING:
    from os import PathLike


_INVALID_SEGMENTS: frozenset[str] = frozenset({"", ".", ".."})


class FileCheckpointStore(CheckpointStore):
    """
    Filesystem-backed :class:`~.checkpoint_store.CheckpointStore`.

    See module docstring for layout and safety semantics.
    """

    def __init__(self, root: str | PathLike[str]) -> None:
        # Resolve early so containment checks compare against a stable,
        # canonical path (no symlink surprises at save time).
        self._root = Path(root).resolve()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    async def save(self, key: str, data: bytes) -> None:
        path = self._key_to_path(key)
        lock = await self._get_lock(key)
        async with lock:
            await asyncio.to_thread(_atomic_write, path, data)

    async def load(self, key: str) -> bytes | None:
        path = self._key_to_path(key)
        return await asyncio.to_thread(_read_if_exists, path)

    async def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        lock = await self._get_lock(key)
        async with lock:
            await asyncio.to_thread(_unlink_if_exists, path)

    async def list_keys(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(_list_keys, self._root, prefix)

    # --- Internals ---

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
        return lock

    def _key_to_path(self, key: str) -> Path:
        """
        Map a checkpoint key to its on-disk file path.

        Rejects malformed / escape-prone keys up front; then verifies the
        resolved path stays under ``self._root`` as defense in depth.
        """
        if not key:
            raise ValueError("Checkpoint key must be non-empty")
        if key.startswith("/") or "\\" in key or "\x00" in key:
            raise ValueError(f"Invalid checkpoint key: {key!r}")

        segments = key.split("/")
        for seg in segments:
            if seg in _INVALID_SEGMENTS:
                raise ValueError(f"Invalid segment {seg!r} in key {key!r}")

        file_seg = segments[-1] + ".json"
        if len(segments) == 1:
            target = self._root / file_seg
        else:
            target = self._root.joinpath(*segments[:-1], file_seg)

        # Defense in depth: even with the syntactic checks above, make
        # sure the resolved target does not climb out of the root via a
        # pre-existing symlink in the store tree.
        try:
            resolved = target.resolve(strict=False)
        except OSError as exc:
            raise ValueError(
                f"Cannot resolve store path for key {key!r}: {exc}"
            ) from exc
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(
                f"Key {key!r} resolves outside store root {self._root}"
            ) from exc
        return target


def _atomic_write(path: Path, data: bytes) -> None:
    """Blocking write — run via :func:`asyncio.to_thread`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile yields a unique name in the target dir, so the
    # rename is atomic (same filesystem) and concurrent writers that
    # bypass the per-key lock (e.g. separate processes) don't clobber
    # each other's tmp payload.
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        Path(tmp_path).replace(path)
    except BaseException:
        # Never leave a stale tmp behind if the replace failed.
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass
        raise


def _read_if_exists(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _list_keys(root: Path, prefix: str) -> list[str]:
    if not root.exists():
        return []
    keys: list[str] = []
    for path in root.rglob("*.json"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        key_parts = (*rel.parent.parts, rel.stem)
        key = "/".join(key_parts)
        if key.startswith(prefix):
            keys.append(key)
    return keys
