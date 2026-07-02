"""
Integration tests for per-agent file-edit routing.

Read-before-write bookkeeping lives on the active agent's
:class:`FileEditSessionState`, surfaced to the tools through the
:class:`AgentContext` passed on each call. Tools share state when they share
the same ``AgentContext``; a different ``AgentContext`` (a parent → child
agent transition) isolates state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.bash_common import ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit import (
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    WriteInput,
    WriteResult,
)
from grasp_agents.tools.notebook_exec import KernelHolder
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


def _agent_ctx(state: FileEditSessionState) -> AgentContext:
    """An ``AgentContext`` wrapping ``state`` — the field the file tools read."""
    transcript = LLMAgentTranscript()
    return AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=state,
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
        session_holder=BashSessionHolder(),
        nb_kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )


async def test_tools_share_state_within_activation(tmp_path: Path) -> None:
    """Read + Write under the same ``AgentContext`` compose."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("original")

    state = FileEditSessionState()
    agent_ctx = _agent_ctx(state)
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    result = await tk.write.run(
        WriteInput(path=str(f), content="updated"), ctx=ctx, agent_ctx=agent_ctx
    )

    assert isinstance(result, WriteResult)
    assert f.read_text() == "updated"
    assert state.read_file_state, "state should hold a read record"


async def test_separate_activations_are_isolated(tmp_path: Path) -> None:
    """A Read under one ``AgentContext`` does not satisfy a Write under another."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("x")

    parent = _agent_ctx(FileEditSessionState())
    child = _agent_ctx(FileEditSessionState())

    # Parent context: Read + Write succeed.
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=parent)
    result_parent = await tk.write.run(
        WriteInput(path=str(f), content="parent-wrote"), ctx=ctx, agent_ctx=parent
    )
    assert isinstance(result_parent, WriteResult)

    # Child context: same backend, but no prior Read in its state.
    result_child = await tk.write.run(
        WriteInput(path=str(f), content="child-wrote"), ctx=ctx, agent_ctx=child
    )
    assert "Must Read" in _error_message(result_child)


async def test_shared_state_mimics_one_agent_two_toolkits(tmp_path: Path) -> None:
    """Two toolkit instances sharing one ``AgentContext`` compose Read → Write."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk_a = FileToolkit(redactor=NullRedactor())
    tk_b = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "shared.txt"
    f.write_text("from the file")

    agent_ctx = _agent_ctx(FileEditSessionState())
    await tk_a.read.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    result = await tk_b.write.run(
        WriteInput(path=str(f), content="child-wrote"), ctx=ctx, agent_ctx=agent_ctx
    )

    assert isinstance(result, WriteResult)
    assert f.read_text() == "child-wrote"


async def test_fresh_state_starts_clean(tmp_path: Path) -> None:
    """A fresh ``AgentContext`` is a new agent — earlier reads don't carry over."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("x")

    state_a = FileEditSessionState()
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=_agent_ctx(state_a))
    assert state_a.read_file_state

    # New agent → new state → no prior reads.
    result = await tk.write.run(
        WriteInput(path=str(f), content="y"),
        ctx=ctx,
        agent_ctx=_agent_ctx(FileEditSessionState()),
    )
    assert "Must Read" in _error_message(result)


async def test_standalone_tool_use_without_state(tmp_path: Path) -> None:
    """With no ``AgentContext``, tools still work but skip read-before-write."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("orig")

    # No agent_ctx — Write proceeds without prior Read.
    result = await tk.write.run(WriteInput(path=str(f), content="updated"), ctx=ctx)
    assert isinstance(result, WriteResult)
    assert f.read_text() == "updated"
