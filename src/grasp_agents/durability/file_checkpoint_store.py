"""
Filesystem-backed :class:`CheckpointStore`.

Each key maps to one JSON file under ``root``: segments become directory
components and ``.json`` is appended to the leaf. A checkpoint at
``<x>`` and metadata at ``<x>/lifecycle`` coexist on disk because the
parent's file (``<x>.json``) and the directory (``<x>/``) carrying its
children share a name without colliding.

Writes are atomic (``tempfile.mkstemp`` + ``os.replace``). Concurrent
writes to the same key serialize via a per-key ``asyncio.Lock``. No
TTL / GC — retention is the caller's concern.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from grasp_agents.file_backend.atomic_write import atomic_write_bytes

from .checkpoint_store import (
    CheckpointStore,
    decode_message_log,
    encode_messages,
)
from .store_keys import is_under

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from grasp_agents.types.items import InputItem


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
        lock = await self._get_lock(key)
        async with lock:
            await asyncio.to_thread(_unlink_if_exists, self._key_to_path(key))
            await asyncio.to_thread(
                _unlink_if_exists, self._key_to_path(key, suffix=".jsonl")
            )
            # Version-suffixed message logs (``<leaf>.v<N>.jsonl``).
            await asyncio.to_thread(
                _unlink_log_versions, self._key_to_path(key, suffix="")
            )
        # Evict the per-key lock so long-lived stores serving many sessions
        # don't grow it unboundedly. A task racing this delete on the same
        # key may briefly hold the old lock object — delete-vs-write races
        # on one key are a caller-level conflict either way.
        async with self._locks_guard:
            self._locks.pop(key, None)

    async def list_keys(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(_list_keys, self._root, prefix)

    # --- Append-only message log ---
    #
    # The transcript lives in a sibling ``.jsonl`` at the head key's path
    # (``.v<N>.jsonl`` for versions > 0 — see the base class on why
    # rewrites go to a fresh version). Appends go straight to the file's
    # end (no temp+rename), so a torn final record is possible —
    # ``decode_message_log`` discards it. ``list_keys`` globs ``*.json``
    # only, so logs never surface as checkpoint keys.

    def _log_suffix(self, version: int) -> str:
        return ".jsonl" if version == 0 else f".v{version}.jsonl"

    async def append_messages(
        self, key: str, messages: Sequence[InputItem], *, version: int = 0
    ) -> None:
        if not messages:
            return
        path = self._key_to_path(key, suffix=self._log_suffix(version))
        blob = encode_messages(messages)
        lock = await self._get_lock(key)
        async with lock:
            await asyncio.to_thread(_append_bytes, path, blob)

    async def read_messages(self, key: str, *, version: int = 0) -> list[InputItem]:
        path = self._key_to_path(key, suffix=self._log_suffix(version))
        return await asyncio.to_thread(_read_message_log, path)

    async def rewrite_messages(
        self, key: str, messages: Sequence[InputItem], *, version: int = 0
    ) -> None:
        path = self._key_to_path(key, suffix=self._log_suffix(version))
        lock = await self._get_lock(key)
        async with lock:
            if not messages:
                await asyncio.to_thread(_unlink_if_exists, path)
            else:
                await asyncio.to_thread(_atomic_write, path, encode_messages(messages))

    # --- Internals ---

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
        return lock

    def _key_to_path(self, key: str, *, suffix: str = ".json") -> Path:
        """
        Map a checkpoint key to its on-disk file path.

        ``suffix`` selects the file the key maps to — ``.json`` for the
        checkpoint head, ``.jsonl`` for its sibling message log. Rejects
        malformed / escape-prone keys up front; then verifies the resolved
        path stays under ``self._root`` as defense in depth.
        """
        if not key:
            raise ValueError("Checkpoint key must be non-empty")
        if key.startswith("/") or "\\" in key or "\x00" in key:
            raise ValueError(f"Invalid checkpoint key: {key!r}")

        segments = key.split("/")
        for seg in segments:
            if seg in _INVALID_SEGMENTS:
                raise ValueError(f"Invalid segment {seg!r} in key {key!r}")

        file_seg = segments[-1] + suffix
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
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, data)


def _read_if_exists(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _append_bytes(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(blob)
        # fsync to match the head's durability: the head is saved (fsynced)
        # right after this append, so a power loss must not be able to keep
        # the head while losing the log tail it acknowledges.
        f.flush()
        os.fsync(f.fileno())


def _read_message_log(path: Path) -> list[InputItem]:
    try:
        blob = path.read_bytes()
    except FileNotFoundError:
        return []
    return decode_message_log(blob)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _unlink_log_versions(base_path: Path) -> None:
    """Remove every version-suffixed log (``<leaf>.v<N>.jsonl``)."""
    parent = base_path.parent
    if not parent.is_dir():
        return
    for path in parent.glob(f"{base_path.name}.v*.jsonl"):
        _unlink_if_exists(path)


def _list_keys(root: Path, prefix: str) -> list[str]:
    if not root.exists():
        return []
    dir_part = prefix.rpartition("/")[0]
    start = root
    if dir_part:
        segments = dir_part.split("/")
        if not any(seg in _INVALID_SEGMENTS for seg in segments):
            start = root.joinpath(*segments)
    if not start.is_dir():
        return []
    keys: list[str] = []
    for path in start.rglob("*.json"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        key_parts = (*rel.parent.parts, rel.stem)
        key = "/".join(key_parts)
        if is_under(key, prefix):
            keys.append(key)
    return keys
