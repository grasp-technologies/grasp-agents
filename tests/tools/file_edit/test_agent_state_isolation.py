"""
State-on-agent isolation tests.

Each :class:`AgentLoop` owns its own :class:`FileEditSessionState`;
``execute_stream`` activates it via the
:mod:`grasp_agents.tools.file_edit.agent_state` ContextVar so the
file-edit tools find it. These tests verify the three load-bearing
invariants of that design:

* the ContextVar default (``None``) keeps standalone tool use safe;
* setting/resetting the var picks up the matching state;
* concurrent ``asyncio.gather`` children inherit the parent's value
  by way of the asyncio context-copy semantics.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from grasp_agents.tools.file_edit.agent_state import (
    get_current_file_edit_state,
    reset_current_file_edit_state,
    set_current_file_edit_state,
)
from grasp_agents.tools.file_edit.session_state import FileEditSessionState

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_default_is_none() -> None:
    """Outside any activation, the var is ``None`` — standalone tool use."""
    assert get_current_file_edit_state() is None


async def test_set_then_reset() -> None:
    """``set_current_file_edit_state`` + ``reset_current_file_edit_state``."""
    state = FileEditSessionState()
    token = set_current_file_edit_state(state)
    try:
        assert get_current_file_edit_state() is state
    finally:
        reset_current_file_edit_state(token)
    assert get_current_file_edit_state() is None


async def test_concurrent_tasks_inherit_parent_state() -> None:
    """
    Asyncio task spawn copies the current context — children see the
    parent's state without any explicit threading. This is what makes
    ``stream_concurrent`` / ``asyncio.gather`` work for tool batches.
    """
    parent_state = FileEditSessionState()
    parent_state.record_read(Path("/tmp/x"), 1.0)
    token = set_current_file_edit_state(parent_state)
    try:

        async def child() -> FileEditSessionState | None:
            return get_current_file_edit_state()

        result_a, result_b = await asyncio.gather(child(), child())
    finally:
        reset_current_file_edit_state(token)

    assert result_a is parent_state
    assert result_b is parent_state


async def test_sibling_activations_dont_leak() -> None:
    """
    Two sequential activations (parent agent → child sub-agent →
    parent resumes) round-trip cleanly: the child cannot see, and
    cannot pollute, the parent's state.
    """
    parent = FileEditSessionState()
    parent.record_read(Path("/tmp/parent-read"), 1.0)

    parent_token = set_current_file_edit_state(parent)
    try:
        assert get_current_file_edit_state() is parent

        child = FileEditSessionState()
        child_token = set_current_file_edit_state(child)
        try:
            active = get_current_file_edit_state()
            assert active is child
            # Parent record is invisible from inside the child's slot.
            assert (
                active is not None
                and active.get_read_record(Path("/tmp/parent-read")) is None
            )
            child.record_read(Path("/tmp/child-read"), 2.0)
        finally:
            reset_current_file_edit_state(child_token)

        # Back at the parent level — the child's record stayed in the child.
        active = get_current_file_edit_state()
        assert active is parent
        assert parent.get_read_record(Path("/tmp/child-read")) is None
        assert parent.get_read_record(Path("/tmp/parent-read")) is not None
    finally:
        reset_current_file_edit_state(parent_token)


async def test_agent_loop_owns_distinct_state_per_instance() -> None:
    """
    Two ``AgentLoop`` instances each get their own ``file_edit_state``
    field — no cross-pollution at construction time.
    """
    # Import lazily so the test module doesn't pay LLMAgent's import
    # cost when run in isolation.
    from pydantic import BaseModel  # noqa: PLC0415

    from grasp_agents.agent.agent_loop import AgentLoop  # noqa: PLC0415
    from grasp_agents.agent.llm_agent_transcript import (  # noqa: PLC0415
        LLMAgentTranscript,
    )
    from grasp_agents.run_context import RunContext  # noqa: PLC0415

    class _StubLLM:
        model_name = "stub"
        litellm_provider = "stub"

    def _make() -> AgentLoop[BaseModel]:
        return AgentLoop[BaseModel](
            agent_name="A",
            llm=_StubLLM(),  # type: ignore[arg-type]
            transcript=LLMAgentTranscript(),
            tools=None,
            ctx=RunContext[BaseModel](),  # type: ignore[call-arg]
            max_turns=1,
        )

    a, b = _make(), _make()
    assert a.file_edit_state is not b.file_edit_state
