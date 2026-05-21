from __future__ import annotations

import asyncio
import logging
import os
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .loader import scan_memdir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..agent.llm_agent_memory import LLMAgentMemory
    from ..run_context import RunContext
    from .types import MemoryEntry

logger = logging.getLogger(__name__)

DEFAULT_STALE_AFTER = timedelta(days=7)


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Frozen-snapshot of a memdir loaded once per session.

    Mirrors the Hermes "load once, never re-fetch mid-session" pattern: the
    snapshot is what the system-prompt section renders against for the whole
    session, even if the underlying files are edited (those edits show up on
    the next session's load).

    ``index`` is the always-loaded ``MEMORY.md`` content (line/byte-capped).
    ``entries`` carries every parsed topic file — currently surfaced only by
    direct lookup, not by the system-prompt section (the relevance selector
    that would inject top-K topic files is deferred).

    Freshness strings are pre-computed at snapshot-creation time so the
    rendered prompt does not drift turn-to-turn ("3 days ago" → "4 days ago"
    would otherwise bust the prompt cache).
    """

    index: str | None = None
    index_mtime_ms: int | None = None
    index_freshness_warning: str | None = None
    entries: tuple[MemoryEntry, ...] = ()
    entry_freshness_warnings: dict[str, str] = field(
        compare=False, default_factory=lambda: dict[str, str]()  # noqa: PLW0108
    )
    root: Path | None = field(compare=False, default=None)

    @property
    def is_empty(self) -> bool:
        return not self.index and not self.entries

    def get(self, name: str) -> MemoryEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None


class MemoryProvider(ABC):
    """
    Cross-session memory loader. One instance per session/agent setup.

    The default contract is read-only: :meth:`load` returns a frozen snapshot
    captured once and reused for the rest of the session. :meth:`write` and
    :meth:`on_pre_compress` are optional opt-ins (default: ``NotImplemented``
    / no-op) so applications can adopt this without wiring authoring or
    compaction up front.
    """

    @abstractmethod
    async def load(
        self, *, session_id: str = "", ctx: RunContext[Any] | None = None
    ) -> MemorySnapshot:
        """Return a frozen memdir snapshot. Implementations should cache."""

    async def write(
        self,
        *,
        entry: MemoryEntry,
        ctx: RunContext[Any] | None = None,
    ) -> None:
        """Persist a memory entry. Default raises ``NotImplementedError``."""
        del entry, ctx
        raise NotImplementedError(
            f"{type(self).__name__} does not implement memory writes; "
            "edit topic files directly or override write()."
        )

    async def on_pre_compress(
        self, *, memory: LLMAgentMemory
    ) -> str:
        """Insights to prepend to a compaction summary prompt. Default empty."""
        del memory
        return ""

    async def refresh(self) -> None:  # noqa: B027
        """Invalidate any cached snapshot. Default no-op."""


class InMemoryMemoryProvider(MemoryProvider):
    """In-memory backend; useful for tests and notebooks."""

    def __init__(
        self,
        *,
        index: str | None = None,
        entries: Sequence[MemoryEntry] = (),
        stale_after: timedelta = DEFAULT_STALE_AFTER,
    ) -> None:
        self._stale_after = stale_after
        self._snapshot = _build_snapshot(
            root=None,
            index=index,
            index_mtime_ms=None,
            entries=list(entries),
            stale_after=stale_after,
        )

    async def load(
        self, *, session_id: str = "", ctx: RunContext[Any] | None = None
    ) -> MemorySnapshot:
        del session_id, ctx
        return self._snapshot


class FileMemoryProvider(MemoryProvider):
    """
    Filesystem-backed memdir provider.

    Walks ``root`` once on first :meth:`load`, returns a frozen snapshot for
    the rest of the session, and re-walks on :meth:`refresh`. Default ``root``
    is :func:`default_memdir_path`. Staleness warnings are computed at load
    time and frozen on the snapshot.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
    ) -> None:
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

    def _load_sync(self) -> MemorySnapshot:
        index, index_mtime, entries = scan_memdir(self._root)
        return _build_snapshot(
            root=self._root,
            index=index,
            index_mtime_ms=index_mtime,
            entries=entries,
            stale_after=self._stale_after,
        )


def _build_snapshot(
    *,
    root: Path | None,
    index: str | None,
    index_mtime_ms: int | None,
    entries: list[MemoryEntry],
    stale_after: timedelta,
) -> MemorySnapshot:
    threshold_ms = int(stale_after.total_seconds() * 1000)
    now_ms = _now_ms()

    index_warning: str | None = None
    if index is not None and index_mtime_ms is not None and threshold_ms > 0:
        age_days = _age_days(now_ms, index_mtime_ms)
        if age_days * 86_400_000 > threshold_ms:
            index_warning = _freshness_warning(age_days)

    entry_warnings: dict[str, str] = {}
    if threshold_ms > 0:
        for entry in entries:
            age_days = _age_days(now_ms, entry.mtime_ms)
            if age_days * 86_400_000 > threshold_ms:
                entry_warnings[entry.name] = _freshness_warning(age_days)

    return MemorySnapshot(
        index=index,
        index_mtime_ms=index_mtime_ms,
        index_freshness_warning=index_warning,
        entries=tuple(entries),
        entry_freshness_warnings=entry_warnings,
        root=root,
    )


def _freshness_warning(age_days: int) -> str:
    return (
        f"<system-reminder>This memory is {age_days} days old — verify before "
        "acting.</system-reminder>"
    )


def _age_days(now_ms: int, mtime_ms: int) -> int:
    if mtime_ms <= 0:
        return 0
    delta = max(0, now_ms - mtime_ms)
    return delta // 86_400_000


def _now_ms() -> int:
    import time  # noqa: PLC0415

    return int(time.time() * 1000)


# ---- Default path resolver --------------------------------------------------

GRASP_MEMORY_ENV = "GRASP_MEMORY_DIR"
GRASP_HOME_DIR_NAME = ".grasp"
PROJECTS_DIR_NAME = "projects"
MEMDIR_DIR_NAME = "memory"


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
