"""
Edit, Write and Delete pass the mtime recorded at Read to the backend.

A change that lands after the tool's own staleness check (a concurrent
writer) must still be refused: the backend sees ``expected_mtime`` and raises
:class:`StaleFileError`, which the tools turn into their usual "modified
since you last read it" error.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.file_backend import FileStat, InMemoryFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.file_edit import (
    DeleteInput,
    EditInput,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    ReadResult,
    WriteInput,
)
from grasp_agents.types.events import ToolErrorInfo
from tests._helpers import _make_agent_ctx

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext

pytestmark = pytest.mark.asyncio

TARGET = Path("/mem/a.md")


class _RacingBackend(InMemoryFileBackend):
    """
    Answers one call from the old state, then lets a concurrent writer in.

    ``race_on`` names the call after which the change lands: ``stat`` for
    Write and Delete, ``read_bytes`` for Edit (which reads after its check).
    """

    def __init__(self) -> None:
        super().__init__(["/mem"], files={str(TARGET): "original\n"})
        self.race_on: str | None = None

    async def _race(self, call: str, path: Path) -> None:
        if self.race_on == call and path == TARGET:
            self.race_on = None
            await super().write_bytes(TARGET, b"concurrent\n", mode=0o644)

    async def stat(self, path: Path) -> FileStat:
        result = await super().stat(path)
        await self._race("stat", path)
        return result

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        result = await super().read_bytes(path)
        await self._race("read_bytes", path)
        return result


async def _read_then_arm(
    race_on: str,
) -> tuple[_RacingBackend, SessionContext[Any], AgentContext]:
    backend = _RacingBackend()
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    agent_ctx = _make_agent_ctx(file_edit_state=FileEditSessionState())
    read = await FileToolkit(redactor=NullRedactor()).read.run(
        ReadInput(path=str(TARGET)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(read, ReadResult)
    backend.race_on = race_on
    return backend, ctx, agent_ctx


def _assert_refused(result: Any) -> None:
    assert isinstance(result, ToolErrorInfo), result
    assert "modified since you last read it" in result.error


async def test_edit_refuses_a_concurrent_change() -> None:
    backend, ctx, agent_ctx = await _read_then_arm("read_bytes")

    result = await FileToolkit().edit.run(
        EditInput(path=str(TARGET), old_string="original", new_string="edited"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    _assert_refused(result)
    assert (await backend.read_text(TARGET))[0] == "concurrent\n"


async def test_write_refuses_a_concurrent_change() -> None:
    backend, ctx, agent_ctx = await _read_then_arm("stat")

    result = await FileToolkit().write.run(
        WriteInput(path=str(TARGET), content="replaced\n"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    _assert_refused(result)
    assert (await backend.read_text(TARGET))[0] == "concurrent\n"


async def test_delete_refuses_a_concurrent_change() -> None:
    backend, ctx, agent_ctx = await _read_then_arm("stat")

    result = await FileToolkit().delete.run(
        DeleteInput(path=str(TARGET)), ctx=ctx, agent_ctx=agent_ctx
    )

    _assert_refused(result)
    assert await backend.exists(TARGET)
