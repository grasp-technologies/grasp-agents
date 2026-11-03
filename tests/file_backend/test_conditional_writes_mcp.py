"""``expected_mtime`` on ``write_bytes`` / ``delete`` in the MCP backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("mcp")

from mcp.types import TextContent

from grasp_agents.file_backend import StaleFileError
from grasp_agents.file_backend.mcp import MCPFileBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grasp_agents.mcp.client import MCPClient

pytestmark = pytest.mark.asyncio


@dataclass
class _Resource:
    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = "text/markdown"  # ruff: ignore[mixed-case-variable-in-class-scope]
    meta: dict[str, Any] | None = None


@dataclass
class _ListResult:
    resources: list[_Resource] = field(default_factory=list[_Resource])


@dataclass
class _ToolResult:
    content: list[Any]
    isError: bool = False  # ruff: ignore[mixed-case-variable-in-class-scope]


class _Session:
    def __init__(self, mtimes_ms: dict[str, int]) -> None:
        self.mtimes_ms = mtimes_ms
        self.tool_calls: list[tuple[str, Mapping[str, Any]]] = []

    async def list_resources(self, *, params: Any = None) -> _ListResult:
        del params
        return _ListResult(
            resources=[
                _Resource(uri=f"file://{p}", name=Path(p).name, meta={"mtime_ms": m})
                for p, m in self.mtimes_ms.items()
            ]
        )

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> _ToolResult:
        self.tool_calls.append((name, args))
        return _ToolResult(content=[TextContent(type="text", text="{}")])


class _Client:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.name = "fake"


async def test_mcp_write_refuses_a_changed_file() -> None:
    session = _Session({"/mem/a.md": 2_000})
    backend = MCPFileBackend(
        client=cast("MCPClient", _Client(session)),
        allowed_roots=[Path("/mem")],
    )

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            Path("/mem/a.md"), b"x", mode=0o644, expected_mtime=1.0
        )
    assert session.tool_calls == []

    await backend.write_bytes(Path("/mem/a.md"), b"x", mode=0o644, expected_mtime=2.0)
    assert [name for name, _ in session.tool_calls] == ["write_file"]


async def test_mcp_delete_refuses_a_missing_file() -> None:
    session = _Session({})
    backend = MCPFileBackend(
        client=cast("MCPClient", _Client(session)),
        allowed_roots=[Path("/mem")],
    )

    with pytest.raises(StaleFileError):
        await backend.delete(Path("/mem/a.md"), expected_mtime=1.0)
    assert session.tool_calls == []
