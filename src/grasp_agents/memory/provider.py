"""
Unified :class:`MemoryProvider` — frontmatter-aware view of a memdir
routed through :attr:`RunContext.file_backend`.

The provider knows the memdir layout (``MEMORY.md`` as index, ``.md``
topic files with YAML frontmatter, freshness from mtime); the
:class:`FileBackend` knows how to fetch bytes. Locally the backend
walks the filesystem; over MCP it walks an :class:`MCPResourceIndex`.
The provider is identical across the two.

For tests / notebooks that don't want any I/O,
:class:`InMemoryMemoryProvider` returns a fixed snapshot.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from grasp_agents.types.selector import Selector

from .types import DEFAULT_STALE_AFTER

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

    from grasp_agents.run_context import RunContext
    from grasp_agents.tools.file_edit.backend import FileBackend
    from grasp_agents.tools.file_edit.session_state import FileEditSessionState
    from grasp_agents.types.items import InputItem

    from .types import MemoryEntry


MemorySelector: TypeAlias = Selector["MemoryEntry"]
"""Relevance selector for the memory topic catalog. See :class:`Selector`."""

logger = logging.getLogger(__name__)

# Stub written when ``auto_create_index`` is on and ``MEMORY.md`` is absent —
# just a heading the agent can append topic pointers under.
DEFAULT_INDEX_CONTENT = "# Memory\n"


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Frozen snapshot of a memdir loaded once per session.

    ``index`` is the always-loaded ``MEMORY.md`` content (line/byte-capped).
    ``entries`` carries every parsed topic file; the relevance selector
    consumed by :data:`relevant_memories_attachment` filters this list per
    turn.

    Freshness strings are pre-computed at snapshot-creation time so the
    rendered prompt does not drift turn-to-turn.
    """

    index: str | None = None
    index_mtime_ms: int | None = None
    index_freshness_warning: str | None = None
    index_truncated: bool = False
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


class MemoryProvider:
    """
    Cross-session memory loader over :attr:`RunContext.file_backend`.

    The provider is **read-shaped** — it surfaces the memdir snapshot to
    the system-prompt section and per-turn relevance attachment, but does
    NOT expose write / delete tools of its own. Authoring goes through
    the generic file-edit tools rooted at the memdir.

    Args:
        root: Address of the memdir in the *backend's* address space.
            Defaults to :func:`default_memdir_path` (host filesystem layout).
            For MCP backends, pass the server-side path (e.g.
            ``Path("/memdir")``).
        stale_after: Age threshold past which a per-entry freshness
            warning is computed at snapshot time.
        auto_create_index: When True (default) and ``MEMORY.md`` is absent,
            bootstrap an empty index (creating the memdir if needed) on first
            load so the agent has a file to append topic pointers to. Best-
            effort: silently falls back to no index if the backend can't
            write (read-only / no write path). Set False for read-only memdirs.

    Carries an optional relevance selector (:meth:`set_selector`)
    consumed by per-turn user-message attachments (e.g. the
    :data:`relevant_memories_attachment`) to pick which topic memories are
    surfaced into the running conversation. The system-prompt section
    does NOT consult the selector — it stays cache-stable across turns.
    Default = identity (return every entry).

    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        backend: FileBackend | None = None,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
        auto_create_index: bool = True,
    ) -> None:
        self._root: Path = (
            Path(root).expanduser() if root is not None else default_memdir_path()
        )
        self._backend: FileBackend | None = backend
        self._stale_after = stale_after
        self._auto_create_index = auto_create_index
        self._cached: MemorySnapshot | None = None
        self._lock = asyncio.Lock()
        self._selector: MemorySelector | None = None

    @property
    def root(self) -> Path:
        return self._root

    def bind_backend(self, backend: FileBackend) -> None:
        """
        Bind a :class:`FileBackend` after construction. Called by the
        :class:`RunContext` validator so users don't have to thread the
        backend explicitly into the provider's ctor.
        """
        self._backend = backend

    async def load(
        self, *, session_state: FileEditSessionState | None = None
    ) -> MemorySnapshot:
        """
        Return a frozen memdir snapshot. Cached after the first call.

        Requires a :class:`FileBackend` to be bound (via ctor or
        :meth:`bind_backend`). The :class:`RunContext` validator binds
        the backend automatically when the memory provider is wired onto
        a context.

        ``session_state`` (the active agent's :class:`FileEditSessionState`,
        from its :class:`AgentContext`) records the index read on a fresh
        load, so the agent may ``Edit`` ``MEMORY.md`` without a redundant
        ``Read``.
        """
        if self._cached is not None:
            return self._cached
        if self._backend is None:
            raise ValueError(
                "MemoryProvider.load requires a FileBackend. Bind one via "
                "the ctor or attach the provider to a RunContext with "
                "file_backend wired — the RunContext validator binds the "
                "backend onto the provider automatically."
            )
        async with self._lock:
            if self._cached is None:
                self._cached = await self._load_via_backend(
                    self._backend, session_state=session_state
                )
            return self._cached

    async def refresh(self) -> None:
        """Invalidate any cached snapshot."""
        async with self._lock:
            self._cached = None

    async def fetch_body(
        self, name: str, *, session_state: FileEditSessionState | None = None
    ) -> str:
        """
        Return the full body of memory ``name``.

        If a path and backend are available, re-read fresh through the
        backend so mid-session edits are visible. Otherwise fall back to
        the body cached on the snapshot (in-memory providers always take
        this path).
        """
        from .loader import parse_memory_md  # noqa: PLC0415
        from .types import MemoryNotFoundError  # noqa: PLC0415

        snapshot = await self.load()
        entry = snapshot.get(name)
        if entry is None:
            raise MemoryNotFoundError(f"Topic memory {name!r} is not available.")

        no_backend = entry.path is None or self._backend is None
        if no_backend:
            if entry.body is None:
                raise MemoryNotFoundError(
                    f"Topic memory {name!r} has no body available."
                )
            return entry.body

        assert entry.path is not None  # narrowed by no_backend above
        assert self._backend is not None

        try:
            text, mtime = await self._backend.read_text(entry.path)
        except (OSError, ValueError) as exc:
            raise MemoryNotFoundError(
                f"Topic memory {name!r} could not be read: {exc}"
            ) from exc

        if session_state is not None:
            session_state.record_read(entry.path, mtime)

        try:
            _, body = parse_memory_md(text, path=entry.path)
        except Exception:
            return entry.body or ""

        return body

    async def render_index(self) -> str | None:
        """Return the cached ``MEMORY.md`` index text (line/byte-capped)."""
        snapshot = await self.load()
        return snapshot.index

    # ---- Catalog selector ----------------------------------------------------

    def set_selector(self, fn: MemorySelector | None) -> None:
        """
        Register a relevance selector consulted by the per-turn attachment.

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
        """
        Run the selector (if any) and return the resulting entries.

        ``ctx`` is passed through to the user-provided selector (which is
        a :class:`Selector` keyed off the active session). The provider
        itself does not consume it.
        """
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

    # ---- Internals -----------------------------------------------------------

    async def _load_via_backend(
        self, backend: FileBackend, *, session_state: FileEditSessionState | None = None
    ) -> MemorySnapshot:
        """
        Walk ``self._root`` via ``backend`` and return a frozen snapshot.

        Filters out hidden dirs / non-``.md`` files; reads each topic
        file via ``backend.read_text`` and parses its frontmatter; the
        index is read separately. Returns an empty snapshot if the root
        doesn't exist on the backend.

        The index read is recorded into the active agent's
        :class:`FileEditSessionState` (it's always in the prompt, so the
        agent may ``Edit`` it without a redundant ``Read``). Topic-file
        reads are NOT recorded here — only their body, when surfaced via
        :meth:`fetch_body`, is.
        """
        from .loader import parse_memory_md, truncate_index  # noqa: PLC0415
        from .types import (  # noqa: PLC0415
            INDEX_FILE_NAME,
            MAX_MEMORY_FILES,
            MemoryEntry,
            MemoryFormatError,
        )

        active_state = session_state

        root = self._root
        index_text: str | None = None
        index_mtime_ms: int | None = None
        index_truncated = False
        index_path = root / INDEX_FILE_NAME
        # Read the index if present. MCP backends have no real
        # directory entries, so a ``backend.exists(root)`` gate would
        # short-circuit empty memdirs; instead we just probe the index
        # file directly and tolerate its absence.
        if await backend.exists(index_path):
            try:
                raw, mtime = await backend.read_text(index_path)
                index_text, index_truncated = truncate_index(raw)
                index_mtime_ms = int(mtime * 1000)
                if active_state is not None:
                    active_state.record_read(index_path, mtime)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "MemoryProvider: failed to read index at %s: %s",
                    index_path,
                    exc,
                )
        elif self._auto_create_index:
            index_text, index_mtime_ms = await self._create_index(
                backend, index_path, active_state
            )

        try:
            entries_listing = await backend.list_dir(root, recursive=True)
        except (OSError, ValueError) as exc:
            logger.warning("MemoryProvider: failed to list memdir %s: %s", root, exc)
            entries_listing = []
        entries: list[MemoryEntry] = []
        for entry in entries_listing:
            if entry.is_dir:
                continue
            if not entry.name.endswith(".md"):
                continue
            if entry.name == INDEX_FILE_NAME:
                continue
            try:
                rel = entry.path.relative_to(root)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                text, mtime = await backend.read_text(entry.path)
            except (OSError, ValueError) as exc:
                logger.warning("MemoryProvider: failed to read %s: %s", entry.path, exc)
                continue
            # NB: no ``record_read`` for topic files here. Only their
            # ``name``/``description`` reach the prompt (via the index);
            # the body is surfaced — and the read recorded — in
            # :meth:`fetch_body` when the per-turn attachment pulls it.
            # Pre-recording every body would let the agent ``Edit`` a
            # file it never actually saw, defeating read-before-write.
            try:
                frontmatter, body = parse_memory_md(text, path=entry.path)
            except MemoryFormatError:
                logger.exception("Failed to load memory at %s", entry.path)
                continue
            entries.append(
                MemoryEntry(
                    frontmatter=frontmatter,
                    body=body,
                    path=entry.path,
                    mtime_ms=int(mtime * 1000),
                )
            )

        entries.sort(key=lambda e: e.mtime_ms, reverse=True)
        if len(entries) > MAX_MEMORY_FILES:
            entries = entries[:MAX_MEMORY_FILES]

        return build_snapshot(
            root=root,
            index=index_text,
            index_mtime_ms=index_mtime_ms,
            entries=entries,
            stale_after=self._stale_after,
            index_truncated=index_truncated,
        )

    async def _create_index(
        self,
        backend: FileBackend,
        index_path: Path,
        active_state: FileEditSessionState | None,
    ) -> tuple[str | None, int | None]:
        """
        Bootstrap an empty ``MEMORY.md`` so the always-loaded index exists.

        Best-effort: if the backend can't create the memdir or write the
        file (read-only, no write path, etc.), log and fall back to no index
        — auto-creation must never break loading. Returns
        ``(index_text, index_mtime_ms)``.
        """
        try:
            await backend.mkdir(self._root)
            resolved = await backend.validate_path(
                index_path, must_exist=False, access="write"
            )
            mtime = await backend.write_bytes(
                resolved,
                DEFAULT_INDEX_CONTENT.encode("utf-8"),
                mode=0o644,
                overwrite=False,
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "MemoryProvider: could not auto-create index at %s: %s",
                index_path,
                exc,
            )
            return None, None

        if active_state is not None:
            active_state.record_read(resolved, mtime)
        logger.info("MemoryProvider: created empty memory index at %s", index_path)
        return DEFAULT_INDEX_CONTENT, int(mtime * 1000)


class InMemoryMemoryProvider(MemoryProvider):
    """In-memory backend; useful for tests and notebooks."""

    def __init__(
        self,
        *,
        index: str | None = None,
        entries: Sequence[MemoryEntry] = (),
        stale_after: timedelta = DEFAULT_STALE_AFTER,
    ) -> None:
        # Bypass MemoryProvider's default-memdir resolution; nothing on
        # disk is touched and the snapshot is provided up-front.
        self._root: Path = Path()
        self._backend = None
        self._stale_after = stale_after
        self._auto_create_index = False
        self._cached: MemorySnapshot | None = None
        self._lock = asyncio.Lock()
        self._selector: MemorySelector | None = None
        self._snapshot = build_snapshot(
            root=None,
            index=index,
            index_mtime_ms=None,
            entries=list(entries),
            stale_after=stale_after,
        )

    async def load(
        self, *, session_state: FileEditSessionState | None = None
    ) -> MemorySnapshot:
        del session_state  # in-memory snapshot: no file read to record
        return self._snapshot


def build_snapshot(
    *,
    root: Path | None,
    index: str | None,
    index_mtime_ms: int | None,
    entries: list[MemoryEntry],
    stale_after: timedelta,
    index_truncated: bool = False,
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
        index_truncated=index_truncated,
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


# Re-export the default-path resolver + memdir-name constants for the
# common case where the host wants the standard memdir layout but the
# provider doesn't manage filesystem I/O itself.
from .default_path import (  # noqa: E402  re-export for back-compat
    GRASP_HOME_DIR_NAME,
    GRASP_MEMORY_ENV,
    MEMDIR_DIR_NAME,
    PROJECTS_DIR_NAME,
    default_memdir_path,
)

__all__ = [
    "DEFAULT_STALE_AFTER",
    "GRASP_HOME_DIR_NAME",
    "GRASP_MEMORY_ENV",
    "MEMDIR_DIR_NAME",
    "PROJECTS_DIR_NAME",
    "InMemoryMemoryProvider",
    "MemoryProvider",
    "MemorySelector",
    "MemorySnapshot",
    "build_snapshot",
    "default_memdir_path",
]
