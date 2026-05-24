"""
MCP-backed memory provider — read-shaped adapter.

Wraps an :class:`MCPClient` and routes :class:`MemoryProvider` reads
through a shared :class:`MCPResourceIndex`:

* ``load`` — discovers topics via ``resources/list`` (one round-trip
  per session, cached).
* ``fetch_body`` — ``resources/read`` against the entry's URI.
* ``render_index`` — pre-fetched alongside ``load`` from the same
  ``resources/list`` payload.

**Authoring** does NOT go through the provider — it uses the generic
file-edit tools rooted at the memdir, with an
:class:`MCPFileBackend` constructed from the same client. The MCP
server is expected to expose:

* Resources for every memdir file (``file://<memdir>/...`` URIs).
* ``write_file`` + ``delete_file`` MCP tools for mutations.

See ``src/grasp_agents/examples/mcp_memory_server.py`` for a reference
implementation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..mcp.resource_index import AnyUrl, MCPResourceIndex
from .loader import strip_frontmatter
from .provider import MemoryProvider, MemorySnapshot
from .types import (
    DEFAULT_STALE_AFTER,
    MEMORY_TYPES,
    MemoryEntry,
    MemoryFrontmatter,
    MemoryNotFoundError,
)

try:
    from mcp.types import TextResourceContents
except ImportError as _err:
    msg = (
        "MCP support requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

if TYPE_CHECKING:
    from datetime import timedelta

    from ..mcp.client import MCPClient
    from ..run_context import RunContext
    from ..tools.file_edit.mcp_backend import MCPFileBackend


logger = logging.getLogger(__name__)


DEFAULT_URI_SCHEME = "file://"
DEFAULT_MEMDIR_PATH = "/memdir"
INDEX_FILE = "MEMORY.md"


class MCPMemoryProvider(MemoryProvider):
    """
    Read-shaped :class:`MemoryProvider` backed by an :class:`MCPClient`.

    Pass a *connected* :class:`MCPClient`; the provider re-uses its
    session. ``root`` is the server-side path holding the memdir. The
    provider lists the server's resources, filters those under
    ``<scheme><root>/``, and reads ``MEMORY.md`` at
    ``<scheme><root>/MEMORY.md``.

    Construction is lightweight — nothing is fetched until the first
    :meth:`load`. The underlying :class:`MCPResourceIndex` can be passed
    in to share with a co-built :class:`MCPFileBackend`, saving a
    duplicate ``resources/list``.

    Server contract: see ``docs/roadmap/13-memory-system.md`` and the
    reference implementation in
    ``src/grasp_agents/examples/mcp_memory_server.py``.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        root: str | None = None,
        uri_scheme: str = DEFAULT_URI_SCHEME,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
        index: MCPResourceIndex | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        # Resolve once so symlink-divergent platforms (macOS' /tmp →
        # /private/tmp is the recurring case) compare equal.
        self._root = Path(root or DEFAULT_MEMDIR_PATH).resolve()
        self._uri_scheme = uri_scheme
        self._index = index or MCPResourceIndex(client, uri_scheme=uri_scheme)
        self._stale_after = stale_after
        self._cached: MemorySnapshot | None = None
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        """Server-side memdir path (used to root an authoring file toolkit)."""
        return self._root

    @property
    def client(self) -> MCPClient:
        return self._client

    @property
    def resource_index(self) -> MCPResourceIndex:
        """Expose the resource index so adapters can share the cache."""
        return self._index

    def make_file_backend(self) -> MCPFileBackend:
        """Construct an :class:`MCPFileBackend` sharing this index."""
        from ..tools.file_edit.mcp_backend import (  # noqa: PLC0415
            MCPFileBackend,
        )

        return MCPFileBackend(
            client=self._client,
            resource_uri_scheme=self._uri_scheme,
            index=self._index,
        )

    def make_file_toolkit(self, **kwargs: Any) -> Any:
        """
        Construct a :class:`FileEditToolkit` rooted at the server-side
        memdir and using this provider's MCP client as the backend.
        """
        from ..tools.file_edit import FileEditToolkit  # noqa: PLC0415

        return FileEditToolkit(
            allowed_roots=[self._root],
            backend=self.make_file_backend(),
            **kwargs,
        )

    # ---- MemoryProvider overrides -------------------------------------------

    async def load(
        self, *, session_id: str = "", ctx: RunContext[Any] | None = None
    ) -> MemorySnapshot:
        del session_id, ctx
        if self._cached is not None:
            return self._cached
        async with self._lock:
            if self._cached is None:
                self._cached = await self._load_uncached()
            return self._cached

    async def refresh(self) -> None:
        async with self._lock:
            self._cached = None
        await self._index.refresh()

    async def fetch_body(self, name: str, *, ctx: RunContext[Any] | None = None) -> str:
        snapshot = await self.load(ctx=ctx)
        entry = snapshot.get(name)
        if entry is None or not entry.uri:
            raise MemoryNotFoundError(f"Topic memory {name!r} is not available.")
        text = await self._read_resource_text(entry.uri)
        if text is None:
            raise MemoryNotFoundError(
                f"Topic memory {name!r} returned no text content."
            )
        # The MCP server returns the full file (including frontmatter).
        # Strip frontmatter so callers see only the body.
        return strip_frontmatter(text)

    # ---- Internals -----------------------------------------------------------

    async def _load_uncached(self) -> MemorySnapshot:
        entries: list[MemoryEntry] = []
        index_mtime_ms: int | None = None
        index_uri = self._index.make_uri(self._root / INDEX_FILE)

        for resource in await self._index.list_under(self._root):
            if resource.uri == index_uri:
                index_mtime_ms = resource.mtime_ms or None
                continue
            # Server-set name (read from frontmatter) is the canonical
            # topic name. The URI's basename is the filename, which may
            # diverge from the frontmatter ``name``; trust the server.
            if not resource.name:
                continue
            raw_type = resource.meta.get("type")
            if raw_type not in MEMORY_TYPES:
                raw_type = None
            try:
                frontmatter = MemoryFrontmatter(
                    name=resource.name,
                    description=resource.description or resource.name,
                    type=raw_type,
                )
            except Exception as exc:
                logger.warning(
                    "MCPMemoryProvider: skipping resource %s with invalid "
                    "frontmatter: %s",
                    resource.uri,
                    exc,
                )
                continue
            entries.append(
                MemoryEntry(
                    frontmatter=frontmatter,
                    body=None,
                    path=None,
                    uri=resource.uri,
                    mtime_ms=resource.mtime_ms,
                )
            )

        # Sort by mtime (newest first), matching FileMemoryProvider.
        entries.sort(key=lambda e: e.mtime_ms, reverse=True)

        # Fetch the index alongside the listing so ``render_index`` (base
        # default) returns it from the snapshot — symmetric with
        # ``FileMemoryProvider`` and avoids one network round-trip per
        # ``build_system_prompt`` call. Failure here is non-fatal: the
        # section just renders without an index sub-block.
        index_text: str | None = None
        try:
            index_text = await self._read_resource_text(index_uri)
        except MemoryNotFoundError:
            logger.debug(
                "MCPMemoryProvider: no index resource at %s", index_uri
            )

        from .provider import build_snapshot  # noqa: PLC0415

        return build_snapshot(
            root=self._root,
            index=index_text,
            index_mtime_ms=index_mtime_ms,
            entries=entries,
            stale_after=self._stale_after,
        )

    async def _read_resource_text(self, uri: str) -> str | None:
        session = self._client.session
        try:
            result = await session.read_resource(AnyUrl(uri))
        except Exception as exc:
            raise MemoryNotFoundError(
                f"MCP resource {uri!r} could not be read: {exc}"
            ) from exc

        for content in result.contents:
            if isinstance(content, TextResourceContents):
                return content.text
        return None
