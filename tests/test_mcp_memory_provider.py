"""
End-to-end wiring test: :class:`MemoryProvider` over :class:`MCPFileBackend`.

The unified :class:`MemoryProvider` walks ``ctx.file_backend.list_dir`` and
``ctx.file_backend.read_text``. With an :class:`MCPFileBackend` against a
fake MCP client, the provider sees the same shape it sees against a real
local filesystem — that's the contract this file exercises.

Deep MCP-resource-index behaviour is tested in
``tests/test_mcp_resource_index.py`` and ``tests/test_mcp_client.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import AnyUrl

if TYPE_CHECKING:
    from collections.abc import Mapping

pytest.importorskip("mcp")

from grasp_agents.memory import MemoryProvider
from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit.mcp_backend import MCPFileBackend

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
    """Stub MCP ClientSession exposing only the methods used by the backend."""

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

    async def list_resources(
        self, *, params: Any = None
    ) -> _FakeListResourcesResult:
        del params
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


class _FakeClient:
    """Stub MCPClient — only ``session`` and ``name`` are touched."""

    def __init__(self, *, session: _FakeSession | None) -> None:
        self._session = session
        self.name = "fake-mcp"

    @property
    def session(self) -> _FakeSession:
        if self._session is None:
            raise RuntimeError("Not connected")
        return self._session


def _topic_text(name: str, *, body: str = "Body text.") -> str:
    return (
        f"---\nname: {name}\ndescription: {name.title()} memory\n---\n"
        f"{body}\n"
    )


def _make_ctx(session: _FakeSession) -> RunContext[Any]:
    """Build a RunContext wired to ``MemoryProvider`` over ``MCPFileBackend``."""
    backend = MCPFileBackend(
        client=_FakeClient(session=session),  # type: ignore[arg-type]
        allowed_roots=[Path(MEMDIR)],
    )
    return RunContext[Any](
        file_backend=backend,
        memory=MemoryProvider(root=MEMDIR),
    )


class TestLoadOverMCP:
    @pytest.mark.anyio
    async def test_filters_to_memdir_and_skips_index(self) -> None:
        session = _FakeSession(
            resources=[
                _FakeResource(
                    uri=topic_uri("alpha"),
                    name="alpha.md",
                    description="Alpha",
                    meta={"mtime_ms": 2000},
                ),
                _FakeResource(
                    uri=topic_uri("beta"),
                    name="beta.md",
                    description="Beta",
                    meta={"mtime_ms": 1000},
                ),
                _FakeResource(
                    uri=index_uri(),
                    name="MEMORY.md",
                    meta={"mtime_ms": 3000},
                ),
                _FakeResource(uri="other://x", name="x.md"),
            ],
            resource_text={
                topic_uri("alpha"): _topic_text("alpha"),
                topic_uri("beta"): _topic_text("beta"),
                index_uri(): "# Memory index",
            },
        )
        ctx = _make_ctx(session)
        assert ctx.memory is not None
        snap = await ctx.memory.load()

        names = [e.name for e in snap.entries]
        # Sorted newest first by mtime.
        assert names == ["alpha", "beta"]
        # Index loaded separately.
        assert snap.index == "# Memory index"

    @pytest.mark.anyio
    async def test_skipped_invalid_frontmatter_logged(self) -> None:
        session = _FakeSession(
            resources=[
                _FakeResource(
                    uri=topic_uri("bad"),
                    name="bad.md",
                    description="invalid",
                ),
                _FakeResource(
                    uri=topic_uri("good"),
                    name="good.md",
                    description="ok",
                ),
            ],
            resource_text={
                # ``bad.md`` has no frontmatter — provider logs + skips.
                topic_uri("bad"): "No frontmatter here.\n",
                topic_uri("good"): _topic_text("good"),
            },
        )
        ctx = _make_ctx(session)
        assert ctx.memory is not None
        snap = await ctx.memory.load()
        assert [e.name for e in snap.entries] == ["good"]


class TestRenderIndexOverMCP:
    @pytest.mark.anyio
    async def test_reads_index_resource(self) -> None:
        session = _FakeSession(
            resources=[_FakeResource(uri=index_uri(), name="MEMORY.md")],
            resource_text={index_uri(): "# idx"},
        )
        ctx = _make_ctx(session)
        assert ctx.memory is not None
        text = await ctx.memory.render_index()
        assert text == "# idx"
