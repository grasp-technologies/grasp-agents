"""
Cached :class:`MCPResourceIndex` — one ``resources/list`` per session.

:class:`grasp_agents.tools.file_edit.mcp_backend.MCPFileBackend` walks
an MCP server's resources to back the file-edit tools, and the same
listings drive memdir discovery for :class:`MemoryProvider`. The index
keeps a cached :meth:`resources/list` so both surfaces avoid duplicate
round-trips.

This module owns:

* The cached snapshot of ``resources/list`` (lazy + invalidatable).
* URI ↔ Path translation for the configured scheme.
* Convenience queries (lookup by URI, list-under-prefix, stat-by-URI).

The index does **not** perform reads or mutations; callers fetch
content via :meth:`mcp.ClientSession.read_resource` and mutate via
MCP tools. After any write/delete the caller must call
:meth:`refresh` so subsequent metadata queries see the new state.

Resource entries are recorded with their POSIX-form :class:`pathlib.PurePosixPath`
so address comparison is uniform regardless of the host platform —
MCP URIs are POSIX.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

try:
    from mcp.types import PaginatedRequestParams, Resource
    from pydantic import AnyUrl
except ImportError as _err:  # pragma: no cover — same MCP-extra import path as client
    msg = (
        "MCP resource index requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

if TYPE_CHECKING:
    from .client import MCPClient


DEFAULT_URI_SCHEME = "file://"


@dataclass(frozen=True)
class ResourceEntry:
    """One server-side resource as seen by :class:`MCPResourceIndex`."""

    uri: str
    path: PurePosixPath  # POSIX form for cross-platform comparison
    name: str
    description: str | None
    mtime_ms: int  # 0 if the server didn't ship ``meta.mtime_ms``
    size: int  # ``-1`` if unset
    mime_type: str | None
    is_dir: bool  # ``True`` for resources flagged as directories
    meta: dict[str, Any] = field(compare=False, default_factory=dict[str, Any])

    @property
    def mtime_seconds(self) -> float:
        return self.mtime_ms / 1000.0 if self.mtime_ms else 0.0


class MCPResourceIndex:
    """
    Cached view of ``resources/list`` for an :class:`MCPClient`.

    Lazy: nothing is fetched until the first ``load()``. The cache is
    keyed by URI. Call :meth:`refresh` after any server-side mutation
    (write, delete) you control; everything else trusts the cache for
    the session.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        uri_scheme: str = DEFAULT_URI_SCHEME,
    ) -> None:
        self._client = client
        self._uri_scheme = uri_scheme
        self._cached: dict[str, ResourceEntry] | None = None
        self._lock = asyncio.Lock()

    @property
    def uri_scheme(self) -> str:
        return self._uri_scheme

    # ---- URI / path conversion ----------------------------------------------

    def make_uri(self, path: PurePosixPath | str) -> str:
        """Render ``path`` as a resource URI under the configured scheme."""
        return f"{self._uri_scheme}{PurePosixPath(path)}"

    def parse_uri(self, uri: str) -> PurePosixPath | None:
        """Return the POSIX path under ``uri`` if it uses this scheme."""
        if not uri.startswith(self._uri_scheme):
            return None
        return PurePosixPath(uri[len(self._uri_scheme) :])

    # ---- Cache mechanics ----------------------------------------------------

    async def load(self) -> dict[str, ResourceEntry]:
        """Return the cached index, fetching once on first access."""
        if self._cached is not None:
            return self._cached
        async with self._lock:
            if self._cached is None:
                self._cached = await self._fetch_uncached()
            return self._cached

    async def refresh(self) -> None:
        """Drop the cache. Next :meth:`load` re-fetches."""
        async with self._lock:
            self._cached = None

    # ---- Queries ------------------------------------------------------------

    async def get(self, uri: str) -> ResourceEntry | None:
        """Lookup by exact URI."""
        index = await self.load()
        return index.get(uri)

    async def get_by_path(self, path: PurePosixPath | str) -> ResourceEntry | None:
        """Lookup by path under this index's scheme."""
        return await self.get(self.make_uri(path))

    async def list_under(self, root: PurePosixPath | str) -> list[ResourceEntry]:
        """
        Return every entry whose URI sits under ``root`` (POSIX prefix).

        Order is unspecified; callers sort as needed.
        """
        prefix = self.make_uri(root).rstrip("/") + "/"
        index = await self.load()
        return [e for e in index.values() if e.uri.startswith(prefix)]

    async def children_of(
        self, root: PurePosixPath | str, *, recursive: bool = False
    ) -> list[ResourceEntry]:
        """
        Directory-listing flavor of :meth:`list_under`.

        With ``recursive=False`` only the immediate children are returned
        (the same shape an OS ``list_dir`` would produce).
        """
        root_posix = PurePosixPath(root)
        out: list[ResourceEntry] = []
        for entry in await self.list_under(root_posix):
            try:
                rel = entry.path.relative_to(root_posix)
            except ValueError:
                continue
            if not recursive and len(rel.parts) != 1:
                continue
            out.append(entry)
        return out

    # ---- Internals ----------------------------------------------------------

    async def _fetch_uncached(self) -> dict[str, ResourceEntry]:
        session = self._client.session
        index: dict[str, ResourceEntry] = {}

        cursor: str | None = None
        # MCP `resources/list` supports pagination via ``cursor`` — keep
        # iterating until ``nextCursor`` comes back empty.
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await session.list_resources(params=params)
            for resource in result.resources:
                uri = str(resource.uri)
                if not uri.startswith(self._uri_scheme):
                    continue
                index[uri] = _entry_from_resource(uri, resource, self._uri_scheme)
            cursor = getattr(result, "nextCursor", None) or None
            if not cursor:
                break

        return index


def _entry_from_resource(uri: str, resource: Resource, scheme: str) -> ResourceEntry:
    """Build a :class:`ResourceEntry` from an MCP resource block."""
    meta: dict[str, Any] = dict(resource.meta) if resource.meta else {}

    raw_mtime = meta.get("mtime_ms", 0)
    try:
        mtime_ms = int(raw_mtime)
    except (TypeError, ValueError):
        mtime_ms = 0

    raw_size = getattr(resource, "size", None)
    if raw_size is None:
        raw_size = meta.get("size", -1)
    try:
        size = int(raw_size) if raw_size is not None else -1
    except (TypeError, ValueError):
        size = -1

    return ResourceEntry(
        uri=uri,
        path=PurePosixPath(uri[len(scheme) :]),
        name=(resource.name or "").strip(),
        description=resource.description or None,
        mtime_ms=mtime_ms,
        size=size,
        mime_type=getattr(resource, "mimeType", None),
        is_dir=bool(meta.get("is_dir")),
        meta=meta,
    )


# AnyUrl is re-exported here so adapters can resolve the import from one
# place (the resource index module) rather than reaching into pydantic.
__all__ = [
    "DEFAULT_URI_SCHEME",
    "AnyUrl",
    "MCPResourceIndex",
    "ResourceEntry",
]
