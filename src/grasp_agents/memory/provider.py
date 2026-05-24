from __future__ import annotations

import asyncio
import inspect
import logging
import os
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from grasp_agents.types.selector import Selector

from .loader import scan_memdir
from .types import DEFAULT_STALE_AFTER

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

    from grasp_agents.run_context import RunContext
    from grasp_agents.types.items import InputItem

    from .types import MemoryEntry


MemorySelector: TypeAlias = Selector["MemoryEntry"]
"""Relevance selector for the memory topic catalog. See :class:`Selector`."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Frozen-snapshot of a memdir loaded once per session.

    ``index`` is the always-loaded ``MEMORY.md`` content (line/byte-capped).
    ``entries`` carries every parsed topic file; the relevance selector
    consumed by :data:`memory_relevance_attachment` filters this list per
    turn.

    Freshness strings are pre-computed at snapshot-creation time so the
    rendered prompt does not drift turn-to-turn.
    """

    index: str | None = None
    index_mtime_ms: int | None = None
    index_freshness_warning: str | None = None
    entries: tuple[MemoryEntry, ...] = ()
    entry_freshness_warnings: dict[str, str] = field(
        compare=False,
        default_factory=lambda: dict[str, str](),  # noqa: PLW0108
    )
    root: Path = field(compare=False, default_factory=Path)

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

    The provider is **read-shaped** — it surfaces the memdir snapshot to the
    system-prompt section and per-turn relevance attachment, but does NOT
    expose write / delete tools of its own. Authoring goes through the
    generic file-edit tools rooted at the memdir; that aligns with Claude
    Code's design (``buildMemoryLines`` instructs the agent to use ``Read``
    / ``Write`` / ``Edit`` rather than a specialized memory-tool surface).

    Carries an optional relevance selector (:meth:`set_selector`) consumed
    by per-turn user-message attachments (e.g. the memory_relevance
    attachment) to pick which topic memories are surfaced into the running
    conversation. The system-prompt section does NOT consult the selector
    — it stays cache-stable across turns. Default = identity (return every
    entry).
    """

    def __init__(self) -> None:
        self._selector: MemorySelector | None = None

    @property
    def root(self) -> Path:
        """
        Address of the memdir in the *backend's* address space.

        Local backends return a :class:`pathlib.Path`; MCP-backed providers
        return the root path the server uses (e.g. ``"/memdir"``).
        Consumed by :meth:`make_file_toolkit` and by the memory
        system-prompt section so the agent knows where to author.
        """
        return Path()

    def make_file_toolkit(self, **kwargs: Any) -> Any:
        """
        Construct a :class:`FileEditToolkit` rooted at this provider's
        memdir.

        Convenience helper for the common pattern of wiring the file
        toolkit and the memory provider against the same store. Override
        on subclasses that need to inject a custom :class:`FileBackend`
        (see :class:`MCPMemoryProvider.make_file_toolkit`).
        """
        from grasp_agents.tools.file_edit import FileEditToolkit  # noqa: PLC0415

        return FileEditToolkit(allowed_roots=[self.root], **kwargs)

    @abstractmethod
    async def load(
        self, *, session_id: str = "", ctx: RunContext[Any] | None = None
    ) -> MemorySnapshot:
        """Return a frozen memdir snapshot. Implementations should cache."""

    async def fetch_body(self, name: str, *, ctx: RunContext[Any] | None = None) -> str:
        """
        Return the full body of memory ``name``.

        Default implementation reads from the cached snapshot. Lazy backends
        (e.g. :class:`MCPMemoryProvider`) override this to fetch on demand.
        Raises :class:`MemoryNotFoundError` when the entry doesn't exist or
        carries no body.
        """
        from .types import MemoryNotFoundError  # noqa: PLC0415

        snapshot = await self.load(ctx=ctx)
        entry = snapshot.get(name)
        if entry is None:
            raise MemoryNotFoundError(f"Topic memory {name!r} is not available.")
        if entry.body is None:
            raise MemoryNotFoundError(
                f"Topic memory {name!r} has no body available; "
                f"override fetch_body() on {type(self).__name__}."
            )

        return entry.body

    async def render_index(self, *, ctx: RunContext[Any] | None = None) -> str | None:
        """
        Return the ``MEMORY.md`` index text (already line/byte-capped).

        Default implementation reads from the cached snapshot. Lazy backends
        override this to fetch the index resource on demand.
        """
        snapshot = await self.load(ctx=ctx)

        return snapshot.index

    async def refresh(self) -> None:
        """Invalidate any cached snapshot. Default no-op."""
        return

    # ---- Catalog selector ----------------------------------------------------

    def set_selector(self, fn: MemorySelector | None) -> None:
        """
        Register a relevance selector consulted by the catalog renderer.

        See :class:`Selector` for the call shape. Pass ``None`` to clear.
        """
        self._selector = fn

    @property
    def selector(self) -> MemorySelector | None:
        return self._selector

    async def select_relevant(
        self,
        snapshot: MemorySnapshot,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Run the selector (if any) and return the resulting entries."""
        if self._selector is None:
            return snapshot.entries
        result = self._selector(
            entries=snapshot.entries,
            ctx=ctx,
            exec_id=exec_id,
            messages=messages,
        )
        if inspect.isawaitable(result):
            result = await result

        return tuple(result)


class InMemoryMemoryProvider(MemoryProvider):
    """In-memory backend; useful for tests and notebooks."""

    def __init__(
        self,
        *,
        index: str | None = None,
        entries: Sequence[MemoryEntry] = (),
        stale_after: timedelta = DEFAULT_STALE_AFTER,
    ) -> None:
        super().__init__()
        self._stale_after = stale_after
        self._snapshot = build_snapshot(
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
        index, index_mtime, entries = scan_memdir(self._root)
        return build_snapshot(
            root=self._root,
            index=index,
            index_mtime_ms=index_mtime,
            entries=entries,
            stale_after=self._stale_after,
        )


def build_snapshot(
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
        root=root or Path(),
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
