"""
Integration tests for state-on-agent file-edit routing.

Read-before-write bookkeeping lives on the active agent's
:class:`FileEditSessionState`, surfaced to the tools via the
:mod:`agent_state` ContextVar. Tools share state when they share the
same activation; switching the activation (mirroring a parent → child
agent transition) isolates state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.file_edit import (
    FileEditSessionState,
    LocalFileBackend,
    NullRedactor,
    ReadInput,
    WriteInput,
    WriteResult,
    reset_current_file_edit_state,
    set_current_file_edit_state,
)
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


async def test_tools_share_state_within_activation(tmp_path: Path) -> None:
    """
    Read + Write under the same activated state compose: the Read
    record satisfies the Write's read-before-write check.
    """
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("original")

    state = FileEditSessionState()
    token = set_current_file_edit_state(state)
    try:
        await tk.read.run(ReadInput(path=str(f)), ctx=ctx)
        result = await tk.write.run(WriteInput(path=str(f), content="updated"), ctx=ctx)
    finally:
        reset_current_file_edit_state(token)

    assert isinstance(result, WriteResult)
    assert f.read_text() == "updated"
    assert state.read_file_state, "state should hold a read record"


async def test_separate_activations_are_isolated(tmp_path: Path) -> None:
    """
    Different activations (parent vs. child agent) don't see each
    other's reads — a Read under activation A does not satisfy a Write
    under activation B even when both share the backend.
    """
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("x")

    state_parent = FileEditSessionState()
    state_child = FileEditSessionState()

    # Parent activation: Read + Write succeed.
    token = set_current_file_edit_state(state_parent)
    try:
        await tk.read.run(ReadInput(path=str(f)), ctx=ctx)
        result_parent = await tk.write.run(
            WriteInput(path=str(f), content="parent-wrote"), ctx=ctx
        )
    finally:
        reset_current_file_edit_state(token)
    assert isinstance(result_parent, WriteResult)

    # Child activation: same backend, but no prior Read in its state.
    token = set_current_file_edit_state(state_child)
    try:
        result_child = await tk.write.run(
            WriteInput(path=str(f), content="child-wrote"), ctx=ctx
        )
    finally:
        reset_current_file_edit_state(token)
    assert "Must Read" in _error_message(result_child)


async def test_shared_state_mimics_one_agent_two_toolkits(tmp_path: Path) -> None:
    """
    Two *different* toolkit instances drawing the same activated state
    can compose Read → Write — the shape a single agent gets when its
    tools come from multiple sources but share its :class:`FileEditSessionState`.
    """
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    tk_a = FileToolkit(redactor=NullRedactor())
    tk_b = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "shared.txt"
    f.write_text("from the file")

    token = set_current_file_edit_state(FileEditSessionState())
    try:
        await tk_a.read.run(ReadInput(path=str(f)), ctx=ctx)
        result = await tk_b.write.run(
            WriteInput(path=str(f), content="child-wrote"), ctx=ctx
        )
    finally:
        reset_current_file_edit_state(token)

    assert isinstance(result, WriteResult)
    assert f.read_text() == "child-wrote"


async def test_fresh_state_starts_clean(tmp_path: Path) -> None:
    """
    Allocating a fresh :class:`FileEditSessionState` is the equivalent
    of "starting a new agent" — earlier reads don't carry over.
    """
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("x")

    state_a = FileEditSessionState()
    token = set_current_file_edit_state(state_a)
    try:
        await tk.read.run(ReadInput(path=str(f)), ctx=ctx)
    finally:
        reset_current_file_edit_state(token)
    assert state_a.read_file_state

    # New agent → new state → no prior reads.
    state_b = FileEditSessionState()
    token = set_current_file_edit_state(state_b)
    try:
        result = await tk.write.run(WriteInput(path=str(f), content="y"), ctx=ctx)
    finally:
        reset_current_file_edit_state(token)
    assert "Must Read" in _error_message(result)


async def test_standalone_tool_use_without_state(tmp_path: Path) -> None:
    """
    With no active state (no agent in scope), tools still work but
    skip read-before-write enforcement — the power-user escape hatch.
    """
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())

    f = tmp_path / "a.txt"
    f.write_text("orig")

    # No ContextVar activation — Write proceeds without prior Read.
    result = await tk.write.run(WriteInput(path=str(f), content="updated"), ctx=ctx)
    assert isinstance(result, WriteResult)
    assert f.read_text() == "updated"
