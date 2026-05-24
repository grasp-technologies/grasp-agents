"""Tests for MCPMemoryProvider using a fake MCP client + session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import AnyUrl

if TYPE_CHECKING:
    from collections.abc import Mapping

pytest.importorskip("mcp")

from grasp_agents.memory import (  # noqa: E402
    MCPMemoryProvider,
    MemoryNotFoundError,
)


# Default memdir for tests — picks the standard ``file://`` URI scheme.
MEMDIR = "/memdir"


def topic_uri(name: str) -> str:
    return f"file://{MEMDIR}/{name}.md"


def index_uri() -> str:
    return f"file://{MEMDIR}/MEMORY.md"


# ---------- Fake MCP session + client ----------


@dataclass
class _FakeResource:
    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = "text/markdown"  # noqa: N815
    meta: dict[str, Any] | None = None


@dataclass
class _FakeListResourcesResult:
    resources: list[_FakeResource] = field(default_factory=list)


@dataclass
class _FakeReadResult:
    contents: list[Any]


class _FakeSession:
    """Stub MCP ClientSession exposing only the methods used by the provider."""

    def __init__(
        self,
        *,
        resources: list[_FakeResource] | None = None,
        resource_text: dict[str, str] | None = None,
    ) -> None:
        self._resources = resources or []
        self._resource_text = resource_text or {}
        self.list_calls: int = 0
        self.read_calls: list[str] = []
        self.tool_calls: list[tuple[str, Mapping[str, Any]]] = []

    async def list_resources(self) -> _FakeListResourcesResult:  # noqa: RUF029
        self.list_calls += 1
        return _FakeListResourcesResult(resources=list(self._resources))

    async def read_resource(self, uri: AnyUrl) -> _FakeReadResult:
        self.read_calls.append(str(uri))
        text = self._resource_text.get(str(uri))
        if text is None:
            msg = f"resource {uri!s} not found"
            raise RuntimeError(msg)
        from mcp.types import TextResourceContents  # noqa: PLC0415

        return _FakeReadResult(
            contents=[
                TextResourceContents(
                    uri=uri, mimeType="text/markdown", text=text
                )
            ]
        )

    async def call_tool(  # noqa: RUF029
        self, name: str, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.tool_calls.append((name, dict(args)))
        return {"status": "ok"}


class _FakeClient:
    """Stub MCPClient — only ``_session`` and ``name`` are touched."""

    def __init__(self, *, session: _FakeSession | None) -> None:
        self._session = session
        self.name = "fake-mcp"


# ---------- Tests ----------


class TestLoadAndListing:
    @pytest.mark.anyio
    async def test_filters_by_prefix_and_skips_index(self) -> None:
        resources = [
            _FakeResource(
                uri=topic_uri("alpha"),
                name="alpha",
                description="Alpha mem",
                meta={"type": "user", "updated_ms": 2000},
            ),
            _FakeResource(
                uri=topic_uri("beta"),
                name="beta",
                description="Beta mem",
                meta={"type": "project", "updated_ms": 1000},
            ),
            # Index lives at MEMORY.md — should NOT be in entries.
            _FakeResource(uri=index_uri(), name="MEMORY.md"),
            # Unrelated resource — should be filtered out.
            _FakeResource(uri="other://x", name="x"),
        ]
        provider = MCPMemoryProvider(
            client=_FakeClient(session=_FakeSession(resources=resources)),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        snap = await provider.load()
        names = [e.name for e in snap.entries]
        # Sorted newest first by updated_ms.
        assert names == ["alpha", "beta"]
        alpha = snap.entries[0]
        assert alpha.uri == topic_uri("alpha")
        assert alpha.body is None
        assert alpha.path is None
        assert alpha.memory_type == "user"
        assert alpha.mtime_ms == 2000

    @pytest.mark.anyio
    async def test_missing_meta_graceful(self) -> None:
        resources = [
            _FakeResource(
                uri=topic_uri("alpha"),
                name="alpha",
                description="A",
                meta=None,
            ),
        ]
        provider = MCPMemoryProvider(
            client=_FakeClient(session=_FakeSession(resources=resources)),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        snap = await provider.load()
        assert len(snap.entries) == 1
        entry = snap.entries[0]
        assert entry.memory_type is None
        assert entry.mtime_ms == 0

    @pytest.mark.anyio
    async def test_unknown_type_dropped_to_none(self) -> None:
        resources = [
            _FakeResource(
                uri=topic_uri("alpha"),
                name="alpha",
                description="A",
                meta={"type": "not-a-real-type"},
            ),
        ]
        provider = MCPMemoryProvider(
            client=_FakeClient(session=_FakeSession(resources=resources)),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        snap = await provider.load()
        assert snap.entries[0].memory_type is None

    @pytest.mark.anyio
    async def test_invalid_frontmatter_resource_skipped(self) -> None:
        resources = [
            _FakeResource(
                uri=topic_uri("bad"),
                name="Bad Name!",
                description="invalid",
            ),
            _FakeResource(
                uri=topic_uri("good"), name="good", description="ok"
            ),
        ]
        provider = MCPMemoryProvider(
            client=_FakeClient(session=_FakeSession(resources=resources)),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        snap = await provider.load()
        # Only `good` survives — invalid frontmatter is logged and skipped.
        assert [e.name for e in snap.entries] == ["good"]

    @pytest.mark.anyio
    async def test_load_is_cached(self) -> None:
        session = _FakeSession(resources=[])
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        await provider.load()
        await provider.load()
        assert session.list_calls == 1
        await provider.refresh()
        await provider.load()
        assert session.list_calls == 2


class TestFetchBody:
    @pytest.mark.anyio
    async def test_reads_resource_text(self) -> None:
        session = _FakeSession(
            resources=[
                _FakeResource(
                    uri=topic_uri("alpha"),
                    name="alpha",
                    description="A",
                )
            ],
            resource_text={topic_uri("alpha"): "ALPHA BODY HERE"},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        body = await provider.fetch_body("alpha")
        assert body == "ALPHA BODY HERE"
        # The first call resolves via list_resources (cached) then read_resource.
        assert session.read_calls[-1] == topic_uri("alpha")

    @pytest.mark.anyio
    async def test_strips_frontmatter(self) -> None:
        text = (
            "---\nname: alpha\ndescription: A\n---\n"
            "Actual body content here.\n"
        )
        session = _FakeSession(
            resources=[
                _FakeResource(
                    uri=topic_uri("alpha"),
                    name="alpha",
                    description="A",
                )
            ],
            resource_text={topic_uri("alpha"): text},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        body = await provider.fetch_body("alpha")
        assert body.strip() == "Actual body content here."

    @pytest.mark.anyio
    async def test_missing_body_raises(self) -> None:
        session = _FakeSession(
            resources=[
                _FakeResource(uri=topic_uri("alpha"), name="alpha")
            ],
            resource_text={},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        with pytest.raises(MemoryNotFoundError):
            await provider.fetch_body("alpha")


class TestRenderIndex:
    @pytest.mark.anyio
    async def test_reads_index_resource(self) -> None:
        # Index is fetched once during ``load`` and cached on the snapshot
        # so render_index returns it without hitting the network.
        session = _FakeSession(
            resource_text={index_uri(): "# Memory index"},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        text = await provider.render_index()
        assert text == "# Memory index"
        assert session.read_calls == [index_uri()]
        # A second call hits the cache, no new network round-trip.
        text2 = await provider.render_index()
        assert text2 == "# Memory index"
        assert session.read_calls == [index_uri()]

    @pytest.mark.anyio
    async def test_missing_index_returns_none(self) -> None:
        session = _FakeSession(resource_text={})
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        text = await provider.render_index()
        assert text is None

    @pytest.mark.anyio
    async def test_custom_uri_scheme(self) -> None:
        session = _FakeSession(
            resource_text={f"grasp://{MEMDIR}/MEMORY.md": "## Main"},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
            uri_scheme="grasp://",
        )
        text = await provider.render_index()
        assert text == "## Main"

    @pytest.mark.anyio
    async def test_index_meta_drives_freshness(self) -> None:
        # The index resource's ``_meta.updated_ms`` flows into the snapshot
        # so the standard pre-computed staleness warning works against it.
        session = _FakeSession(
            resources=[
                _FakeResource(
                    uri=index_uri(),
                    name="MEMORY.md",
                    meta={"updated_ms": 12345},
                )
            ],
            resource_text={index_uri(): "# idx"},
        )
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        snap = await provider.load()
        assert snap.index == "# idx"
        assert snap.index_mtime_ms == 12345


class TestFileToolkitWiring:
    """
    MCPMemoryProvider is read-shaped — authoring goes through the
    generic file-edit tools rooted at the memdir, backed by
    :class:`MCPFileBackend`. The provider exposes a convenience
    helper for the common wiring.
    """

    def test_root_returns_memdir_path(self) -> None:
        session = _FakeSession(resources=[])
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path="/srv/memdir",
        )
        assert provider.root == "/srv/memdir"

    def test_make_file_toolkit_rooted_at_memdir(self) -> None:
        session = _FakeSession(resources=[])
        provider = MCPMemoryProvider(
            client=_FakeClient(session=session),  # type: ignore[arg-type]
            memdir_path="/srv/memdir",
        )
        toolkit = provider.make_file_toolkit()
        assert toolkit.allowed_roots == ["/srv/memdir"]
        # The toolkit uses an MCP backend talking to the same client.
        assert toolkit.backend.name.startswith("mcp:")


class TestNotConnected:
    @pytest.mark.anyio
    async def test_load_without_session_raises(self) -> None:
        provider = MCPMemoryProvider(
            client=_FakeClient(session=None),  # type: ignore[arg-type]
            memdir_path=MEMDIR,
        )
        with pytest.raises(RuntimeError, match="not connected"):
            await provider.load()
