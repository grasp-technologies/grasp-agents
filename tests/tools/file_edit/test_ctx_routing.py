"""
Integration tests for store-via-``RunContext`` routing.

These cover the production path where the caller sets
``ctx.file_edit_store`` + ``ctx.session_key`` and tools resolve state
from the context rather than their construction-time store. Ensures:

* tools prefer the ctx store when one is set;
* different session keys keep state isolated on the same store;
* sub-agents (or any code sharing the same RunContext) see the same
  read-before-write records;
* ``reset_session(key)`` on the store clears exactly one slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit import (
    FileEditToolkit,
    InMemoryFileEditStore,
    NullRedactor,
    ReadInput,
    WriteInput,
    WriteResult,
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


async def test_ctx_store_overrides_tool_default_store(tmp_path: Path) -> None:
    """
    When ``ctx.file_edit_store`` is set, tools route through it. A prior
    Read recorded via ctx.store under session_key "alice" satisfies a
    later Write via the same ctx — even though the toolkit's own store
    has no record.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    ctx_store = InMemoryFileEditStore()
    ctx = RunContext[None](file_edit_store=ctx_store, session_key="alice")

    f = tmp_path / "a.txt"
    f.write_text("original")

    await tk.read.run(ReadInput(path=str(f)), ctx=ctx)
    result = await tk.write.run(WriteInput(path=str(f), content="updated"), ctx=ctx)
    assert isinstance(result, WriteResult)
    assert f.read_text() == "updated"

    # State landed in ctx_store, not in the toolkit's default store.
    alice_state = await ctx_store.get_session_state("alice")
    assert alice_state.read_file_state
    tk_state = await tk.store.get_session_state(tk.default_session_key)
    assert not tk_state.read_file_state


async def test_separate_session_keys_are_isolated(tmp_path: Path) -> None:
    """
    Two ctxs sharing the same store but with different ``session_key``
    don't see each other's read records — alice reading does not
    satisfy bob's read-before-write invariant.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    shared = InMemoryFileEditStore()
    ctx_alice = RunContext[None](file_edit_store=shared, session_key="alice")
    ctx_bob = RunContext[None](file_edit_store=shared, session_key="bob")

    f = tmp_path / "a.txt"
    f.write_text("x")

    # Alice reads.
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx_alice)
    # Alice can now write.
    result_alice = await tk.write.run(
        WriteInput(path=str(f), content="alice-wrote"), ctx=ctx_alice
    )
    assert isinstance(result_alice, WriteResult)

    # Bob has never read — his write is refused, even though Alice read
    # the same path on the same store.
    result_bob = await tk.write.run(
        WriteInput(path=str(f), content="bob-wrote"), ctx=ctx_bob
    )
    assert "Must Read" in _error_message(result_bob)


async def test_shared_ctx_mimics_subagent_sharing(tmp_path: Path) -> None:
    """
    Two *different* toolkits built on top of the same ctx store share
    read state — the shape a parent agent + sub-agent-as-tool get when
    both are wired to the same ``RunContext``. The parent's Read
    satisfies the sub-agent's Write even though the sub-agent's
    toolkit has its own (empty) default store.
    """
    shared = InMemoryFileEditStore()

    # "Parent" toolkit — reads the file.
    parent_tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    # "Sub-agent" toolkit — a completely separate toolkit instance,
    # simulating a ProcessorTool-wrapped child agent.
    child_tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())

    f = tmp_path / "shared.txt"
    f.write_text("from the file")

    ctx = RunContext[None](file_edit_store=shared, session_key="shared-sess")

    await parent_tk.read.run(ReadInput(path=str(f)), ctx=ctx)
    # Child toolkit's Write honors the parent's Read because both route
    # through ctx.file_edit_store.
    result = await child_tk.write.run(
        WriteInput(path=str(f), content="child-wrote"), ctx=ctx
    )
    assert isinstance(result, WriteResult)
    assert f.read_text() == "child-wrote"


async def test_store_reset_session_affects_only_that_key(tmp_path: Path) -> None:
    """
    Reset on the store drops exactly one session's state; other
    sessions' read records survive.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    shared = InMemoryFileEditStore()
    ctx_a = RunContext[None](file_edit_store=shared, session_key="A")
    ctx_b = RunContext[None](file_edit_store=shared, session_key="B")

    f = tmp_path / "a.txt"
    f.write_text("x")

    await tk.read.run(ReadInput(path=str(f)), ctx=ctx_a)
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx_b)

    await shared.reset_session("A")

    # A's Write is refused (state cleared); B's still allowed.
    result_a = await tk.write.run(WriteInput(path=str(f), content="a"), ctx=ctx_a)
    assert "Must Read" in _error_message(result_a)

    result_b = await tk.write.run(WriteInput(path=str(f), content="b"), ctx=ctx_b)
    assert isinstance(result_b, WriteResult)


async def test_session_resumption_via_session_key(tmp_path: Path) -> None:
    """
    Setting ``ctx.session_key`` to a previously-used key re-keys into
    the existing slot — a session can be 'resumed' (in memory) by
    reusing its key.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    shared = InMemoryFileEditStore()

    f = tmp_path / "a.txt"
    f.write_text("x")

    # Session 1 reads.
    ctx1 = RunContext[None](file_edit_store=shared, session_key="conv-42")
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx1)

    # A new RunContext with the same session_key — simulating a fresh
    # turn in the same logical conversation — sees the prior Read.
    ctx2 = RunContext[None](file_edit_store=shared, session_key="conv-42")
    result = await tk.write.run(WriteInput(path=str(f), content="y"), ctx=ctx2)
    assert isinstance(result, WriteResult)
