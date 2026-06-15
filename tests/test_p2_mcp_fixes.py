"""
Regression tests for the P2 MCP fixes
(consolidated audit 2026-06-11, §3 items 29-31).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, TextResourceContents
from mcp.types import Tool as McpToolDef

from grasp_agents.file_backend.mcp import MCPFileBackend
from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.mcp.client import MCPClient, MCPServerStdio
from grasp_agents.mcp.tool import MCPTool

# ---------- Item 29: connect() exception safety ----------


class TestConnectExceptionSafety:
    @pytest.mark.asyncio
    async def test_failed_connect_leaves_client_disconnected(self) -> None:
        client = MCPClient(
            "dead",
            server=MCPServerStdio(
                command=sys.executable, args=["-c", "import sys; sys.exit(1)"]
            ),
        )
        with pytest.raises(BaseException):
            await asyncio.wait_for(client.connect(), timeout=20.0)

        # No half-connected state: a leaked session would wedge retries.
        assert client._session is None
        assert client._exit_stack is None
        with pytest.raises(RuntimeError, match="Not connected"):
            client.tools()

        # A retry starts over (raises again) instead of returning "connected".
        with pytest.raises(BaseException):
            await asyncio.wait_for(client.connect(), timeout=20.0)


# ---------- Item 30: omitted optionals are not sent as null ----------


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        progress_callback: Any = None,
        meta: Any = None,
        read_timeout_seconds: Any = None,
    ) -> CallToolResult:
        del name, progress_callback, meta, read_timeout_seconds
        self.calls.append(arguments)
        return CallToolResult(content=[], isError=False)


class TestOmittedOptionalsNotSent:
    @pytest.mark.asyncio
    async def test_model_dump_excludes_none(self) -> None:
        session = _RecordingSession()
        tool_def = McpToolDef(
            name="t",
            description="d",
            inputSchema={
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "string"},
                },
                "required": ["required_field"],
            },
        )
        tool = MCPTool(session=cast("Any", session), tool_def=tool_def)
        inp = tool.in_type.model_validate({"required_field": "x"})

        await tool._run(inp)

        assert session.calls == [{"required_field": "x"}]


# ---------- Item 31: ".." collapse + same-surface mtimes ----------


@dataclass
class _FakeClient:
    session: Any = None
    name: str = "fake"


class TestMcpPathNormalization:
    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(self) -> None:
        backend = MCPFileBackend(
            cast("Any", _FakeClient()), allowed_roots=[Path("/memdir")]
        )
        with pytest.raises(PathAccessError, match="outside allowed roots"):
            await backend.validate_path(
                Path("/memdir/../etc/passwd"), must_exist=False
            )

    @pytest.mark.asyncio
    async def test_inner_dotdot_collapsed(self) -> None:
        backend = MCPFileBackend(
            cast("Any", _FakeClient()), allowed_roots=[Path("/memdir")]
        )
        resolved = await backend.validate_path(
            Path("/memdir/./notes/../a.md"), must_exist=False
        )
        assert resolved == Path("/memdir/a.md")

    def test_canonical_collapses(self) -> None:
        backend = MCPFileBackend(
            cast("Any", _FakeClient()), allowed_roots=[Path("/memdir")]
        )
        canonical = backend._canonical(Path("/memdir/notes/../a.md"))
        assert canonical == Path("/memdir/a.md")


class _StubIndex:
    def __init__(self, entry: Any) -> None:
        self.entry = entry

    async def get(self, uri: str) -> Any:
        del uri
        return self.entry

    def make_uri(self, path: Any) -> str:
        return f"file://{path}"


class _ReadSession:
    def __init__(self, meta_mtime_ms: int) -> None:
        self._meta_mtime_ms = meta_mtime_ms

    async def read_resource(self, uri: Any) -> Any:
        contents = TextResourceContents.model_validate(
            {
                "uri": str(uri),
                "mimeType": "text/markdown",
                "text": "hello",
                "_meta": {"mtime_ms": self._meta_mtime_ms},
            }
        )
        return SimpleNamespace(contents=[contents])


class TestMcpStalenessSurface:
    @pytest.mark.asyncio
    async def test_read_text_uses_index_mtime(self) -> None:
        entry = SimpleNamespace(mtime_seconds=111.0, size=5)
        backend = MCPFileBackend(
            cast("Any", _FakeClient(session=_ReadSession(meta_mtime_ms=999_000))),
            allowed_roots=[Path("/m")],
            index=cast("Any", _StubIndex(entry)),
        )
        text, mtime = await backend.read_text(Path("/m/a.md"))
        assert text == "hello"
        # The list-index surface (what stat() reports) wins over read-meta.
        assert mtime == 111.0

    @pytest.mark.asyncio
    async def test_read_text_falls_back_to_meta(self) -> None:
        backend = MCPFileBackend(
            cast("Any", _FakeClient(session=_ReadSession(meta_mtime_ms=999_000))),
            allowed_roots=[Path("/m")],
            index=cast("Any", _StubIndex(None)),
        )
        _, mtime = await backend.read_text(Path("/m/a.md"))
        assert mtime == 999.0
