"""
:class:`FileMemoryProvider` — filesystem-backed memdir + path resolver.

Walks ``root`` once on first :meth:`load`, returns a frozen snapshot
for the rest of the session, and re-walks on :meth:`refresh`.

:func:`default_memdir_path` resolves the project-local memdir path:
``$GRASP_MEMORY_DIR`` if set, else
``~/.grasp/projects/<sanitized-cwd>/memory/``. The cwd sanitization
NFC-normalizes and replaces separators / unsafe chars with underscores
so the per-project memdir tree is filesystem-safe.
"""

from __future__ import annotations

import asyncio
import os
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .loader import scan_memdir
from .provider import MemoryProvider, MemorySnapshot, build_snapshot
from .types import DEFAULT_STALE_AFTER

if TYPE_CHECKING:
    from datetime import timedelta

    from grasp_agents.run_context import RunContext


GRASP_MEMORY_ENV = "GRASP_MEMORY_DIR"
GRASP_HOME_DIR_NAME = ".grasp"
PROJECTS_DIR_NAME = "projects"
MEMDIR_DIR_NAME = "memory"


class FileMemoryProvider(MemoryProvider):
    """
    Filesystem-backed memdir provider.

    Walks ``root`` once on first :meth:`load`, returns a frozen snapshot for
    the rest of the session, and re-walks on :meth:`refresh`. Default ``root``
    is :func:`default_memdir_path`. Staleness warnings are computed at load
    time and frozen on the snapshot.

    :meth:`fetch_body` re-reads the topic file off disk on every call so
    mid-session edits are immediately visible.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
    ) -> None:
        super().__init__()
        self._root = Path(root).expanduser() if root else default_memdir_path()
        self._stale_after = stale_after
        self._cached: MemorySnapshot | None = None
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    async def load(
        self, *, session_id: str = "", ctx: RunContext[Any] | None = None
    ) -> MemorySnapshot:
        del session_id, ctx
        if self._cached is not None:
            return self._cached
        async with self._lock:
            if self._cached is None:
                self._cached = await asyncio.to_thread(self._load_sync)
            return self._cached

    async def refresh(self) -> None:
        async with self._lock:
            self._cached = None

    async def fetch_body(self, name: str, *, ctx: RunContext[Any] | None = None) -> str:
        from .loader import parse_memory_md  # noqa: PLC0415
        from .types import MemoryNotFoundError  # noqa: PLC0415

        snapshot = await self.load(ctx=ctx)
        entry = snapshot.get(name)
        if entry is None or entry.path is None:
            raise MemoryNotFoundError(f"Topic memory {name!r} is not available.")

        try:
            text = await asyncio.to_thread(entry.path.read_text, encoding="utf-8")
        except OSError as exc:
            raise MemoryNotFoundError(
                f"Topic memory {name!r} could not be read: {exc}"
            ) from exc

        try:
            _, body = parse_memory_md(text, path=entry.path)
        except Exception:
            return entry.body or ""

        return body

    def _load_sync(self) -> MemorySnapshot:
        index, index_mtime, index_truncated, entries = scan_memdir(self._root)
        return build_snapshot(
            root=self._root,
            index=index,
            index_mtime_ms=index_mtime,
            index_truncated=index_truncated,
            entries=entries,
            stale_after=self._stale_after,
        )


def default_memdir_path(cwd: Path | None = None) -> Path:
    """
    Resolve the default memdir path for the current project.

    Resolution order:
    1. ``GRASP_MEMORY_DIR`` environment variable (full path, no expansion).
    2. ``~/.grasp/projects/<sanitized-cwd>/memory/``.
    """
    override = os.environ.get(GRASP_MEMORY_ENV)
    if override:
        return Path(override)
    base = Path.home() / GRASP_HOME_DIR_NAME / PROJECTS_DIR_NAME
    sanitized = _sanitize_path((cwd or Path.cwd()).resolve())

    return base / sanitized / MEMDIR_DIR_NAME


def _sanitize_path(path: Path) -> str:
    """NFC-normalize, replace path separators and unsafe chars with underscores."""
    text = unicodedata.normalize("NFC", str(path))
    text = text.replace("/", "_").replace("\\", "_")
    text = text.replace(":", "_").replace("\x00", "_")
    return text.lstrip("_") or "default"
