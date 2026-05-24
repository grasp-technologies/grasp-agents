"""
MCP-backed memory provider — read-shaped adapter.

Wraps an :class:`MCPClient` and routes :class:`MemoryProvider`
**reads** to:

* MCP **resources** for the snapshot — ``resources/list`` to discover
  topics and ``resources/read`` for ``MEMORY.md`` / topic bodies.

**Authoring** (creating / updating / deleting topic files and the
index) does NOT go through the provider — it uses the generic
file-edit tools rooted at the memdir, with an
:class:`MCPFileBackend` constructed from the same client. The MCP
server is expected to expose both surfaces (resources for browsing,
file-protocol tools for editing); see
``src/grasp_agents/examples/mcp_memory_server.py`` for a reference
implementation.

Tool / URI conventions are configurable in case the server doesn't
follow the defaults.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    from mcp import ClientSession

    from grasp_agents.mcp.client import MCPClient
    from grasp_agents.run_context import RunContext
    from grasp_agents.tools.file_edit.mcp_backend import MCPFileBackend

logger = logging.getLogger(__name__)


DEFAULT_URI_SCHEME = "file://"
DEFAULT_MEMDIR_PATH = "/memdir"


class MCPMemoryProvider(MemoryProvider):
    """
    Read-shaped :class:`MemoryProvider` backed by an :class:`MCPClient`.

    Pass a *connected* :class:`MCPClient`; the provider re-uses its
    session. ``memdir_path`` is the server-side path holding the
    memdir. The provider lists the server's resources, filters those
    under ``file://<memdir_path>/`` (the convention of the reference
    server in :mod:`grasp_agents.examples.mcp_memory_server`), and
    reads ``MEMORY.md`` at ``file://<memdir_path>/MEMORY.md``.

    Read flow:

    * :meth:`load` — single ``resources/list`` call, builds a snapshot
      of metadata-only entries (``body=None``, ``uri`` set). Cached for
      the session; :meth:`refresh` invalidates the cache.
    * :meth:`fetch_body` — ``resources/read(entry.uri)``, strips YAML
      frontmatter so callers see just the topic body.
    * :meth:`render_index` — ``resources/read("file://<memdir>/MEMORY.md")``.

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
    ) -> None:
        super().__init__()
        self._client = client
        # Keep the as-given path for ``root`` (so callers see the path
        # they passed in — and the file toolkit roots at exactly the
        # same string). For URI matching, also stash a *resolved*
        # variant so the provider tolerates symlinks the server may
        # follow (macOS' /tmp → /private/tmp is the recurring case).
        self._root = Path(root or DEFAULT_MEMDIR_PATH).resolve()

        self._uri_scheme = uri_scheme

        self._root_uri = f"{uri_scheme}{self._root}/"
        self._index_uri = f"{self._root_uri}MEMORY.md"
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

    def make_file_backend(self) -> MCPFileBackend:
        """Construct an :class:`MCPFileBackend` over the same MCP session."""
        # Local import — keeps the file-edit module out of the optional-MCP
        # import path until the user actually wants the file toolkit.
        from grasp_agents.tools.file_edit.mcp_backend import (  # noqa: PLC0415
            MCPFileBackend,
        )

        return MCPFileBackend(client=self._client)

    def make_file_toolkit(self, **kwargs: Any) -> Any:
        """
        Construct a :class:`FileEditToolkit` rooted at the server-side
        memdir and using this provider's MCP client as the backend.
        """
        from ..tools.file_edit import FileEditToolkit  # noqa: PLC0415

        return FileEditToolkit(
            allowed_roots=[self._root], backend=self.make_file_backend(), **kwargs
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
        return _strip_frontmatter(text)

    # ---- Internals -----------------------------------------------------------

    async def _load_uncached(self) -> MemorySnapshot:
        session = self._session()
        resources_result = await session.list_resources()

        entries: list[MemoryEntry] = []
        index_mtime_ms: int | None = None
        for resource in resources_result.resources:
            uri_str = str(resource.uri)
            if not uri_str.startswith(self._root_uri):
                continue
            meta: dict[str, Any] = resource.meta or {}
            updated_ms_raw = meta.get("updated_ms", 0)
            try:
                updated_ms = int(updated_ms_raw)
            except (TypeError, ValueError):
                updated_ms = 0
            if uri_str == self._index_uri:
                index_mtime_ms = updated_ms or None
                continue
            # Server-set name (read from frontmatter) is the canonical
            # topic name. The URI's basename is the filename, which may
            # diverge from the frontmatter ``name``; trust the server.
            name = (resource.name or "").strip()
            if not name:
                continue
            description = resource.description or name
            raw_type = meta.get("type")
            if raw_type not in MEMORY_TYPES:
                raw_type = None
            try:
                frontmatter = MemoryFrontmatter(
                    name=name, description=description, type=raw_type
                )
            except Exception as exc:
                logger.warning(
                    "MCPMemoryProvider: skipping resource %s with invalid "
                    "frontmatter: %s",
                    uri_str,
                    exc,
                )
                continue
            entries.append(
                MemoryEntry(
                    frontmatter=frontmatter,
                    body=None,
                    path=None,
                    uri=uri_str,
                    mtime_ms=updated_ms,
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
            index_text = await self._read_resource_text(self._index_uri)
        except MemoryNotFoundError:
            logger.debug(
                "MCPMemoryProvider: no index resource at %s",
                self._index_uri,
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
        from pydantic import AnyUrl  # noqa: PLC0415

        session = self._session()
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

    def _session(self) -> ClientSession:
        session: ClientSession | None = self._client._session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if session is None:
            msg = (
                f"MCPClient {self._client.name!r} is not connected; "
                "call connect() or use it as an async context manager first."
            )
            raise RuntimeError(msg)
        return session


_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL
)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``--- ... ---`` YAML block; pass-through if absent."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return text
    return match.group(2).lstrip("\n")
