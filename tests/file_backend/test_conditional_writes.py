"""
``expected_mtime`` on ``write_bytes`` / ``delete`` across backends.

The in-memory backend's exact behavior is covered in
``test_memory_backend.py``; these cover the best-effort checks of the local,
E2B and MCP backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from grasp_agents.file_backend import LocalFileBackend, StaleFileError
from tests.sandbox.test_e2b import _WS, _env, _FakeEntry, _FakeFiles, _FakeSandbox

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grasp_agents.mcp.client import MCPClient

pytestmark = pytest.mark.asyncio


# ---- local ------------------------------------------------------------------


async def test_local_write_with_current_mtime_succeeds(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)

    await backend.write_bytes(f, b"v2", mode=0o644, expected_mtime=mtime)

    assert f.read_text() == "v2"


async def test_local_write_refuses_a_changed_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)
    os.utime(f, (mtime + 5, mtime + 5))

    with pytest.raises(StaleFileError):
        await backend.write_bytes(f, b"v2", mode=0o644, expected_mtime=mtime)
    assert f.read_text() == "v1"


async def test_local_write_refuses_a_vanished_file(tmp_path: Path) -> None:
    backend = LocalFileBackend(allowed_roots=[tmp_path])

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            tmp_path / "gone.txt", b"x", mode=0o644, expected_mtime=1.0
        )
    assert not (tmp_path / "gone.txt").exists()


async def test_local_delete_refuses_a_changed_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v1")
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    _, mtime = await backend.read_text(f)
    os.utime(f, (mtime + 5, mtime + 5))

    with pytest.raises(StaleFileError):
        await backend.delete(f, expected_mtime=mtime)
    assert f.exists()

    await backend.delete(f, expected_mtime=mtime + 5)
    assert not f.exists()


# ---- E2B --------------------------------------------------------------------


async def test_e2b_write_refuses_a_changed_file() -> None:
    fake = _FakeSandbox()
    path = f"{_WS}/a.txt"
    fake.files.infos[path] = _FakeEntry(name="a.txt", path=path, mtime_epoch=1500.0)
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.write_bytes(Path(path), b"x", mode=0o644, expected_mtime=1000.0)
    assert path not in fake.files.written

    await backend.write_bytes(Path(path), b"x", mode=0o644, expected_mtime=1500.0)
    assert fake.files.written[path] == b"x"


async def test_e2b_delete_refuses_a_changed_file() -> None:
    fake = _FakeSandbox()
    path = f"{_WS}/a.txt"
    fake.files.infos[path] = _FakeEntry(name="a.txt", path=path, mtime_epoch=1500.0)
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.delete(Path(path), expected_mtime=1000.0)
    assert fake.files.removed == []


class _StrictFiles(_FakeFiles):
    """Raise E2B's not-found error for unknown paths, like the real API."""

    async def get_info(self, path: str) -> Any:
        from e2b import NotFoundException

        if path not in self.infos:
            raise NotFoundException(f"{path} not found")
        return self.infos[path]


async def test_e2b_write_refuses_a_missing_file() -> None:
    fake = _FakeSandbox(files=_StrictFiles())
    backend = _env(fake).file_backend

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            Path(f"{_WS}/missing.txt"), b"x", mode=0o644, expected_mtime=1000.0
        )


# ---- MCP --------------------------------------------------------------------

pytest.importorskip("mcp")

from grasp_agents.file_backend.mcp import MCPFileBackend


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
        from mcp.types import TextContent

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
