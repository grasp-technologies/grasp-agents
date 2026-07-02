"""
Agent-state isolation tests.

Each :class:`AgentLoop` owns its own :class:`FileEditSessionState`, exposed
on its :class:`AgentContext` and passed explicitly to each tool call (no
ContextVar). These tests verify the per-agent separation that gives:
distinct state per loop, and an :class:`AgentContext` that wraps that loop's
own state.
"""

from __future__ import annotations


def _make_loop():
    # Imported lazily so the module doesn't pay LLMAgent's import cost when
    # collected in isolation.
    from pydantic import BaseModel

    from grasp_agents.agent.agent_loop import AgentLoop
    from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
    from grasp_agents.session_context import SessionContext

    class _StubLLM:
        model_name = "stub"
        litellm_provider = "stub"

    return AgentLoop[BaseModel](
        agent_name="A",
        llm=_StubLLM(),  # type: ignore[arg-type]
        transcript=LLMAgentTranscript(),
        tools=None,
        ctx=SessionContext[BaseModel](),  # type: ignore[call-arg]
        max_turns=1,
    )


def test_agent_loop_owns_distinct_state_per_instance() -> None:
    """Two AgentLoop instances each get their own ``file_edit_state``."""
    a, b = _make_loop(), _make_loop()
    assert a.file_edit_state is not b.file_edit_state


def test_agent_ctx_wraps_loop_state() -> None:
    """
    The loop's :class:`AgentContext` exposes that loop's own
    ``file_edit_state`` (what tools read from), and two loops get distinct
    contexts — the per-agent separation that replaces the old ContextVar.
    """
    a, b = _make_loop(), _make_loop()
    assert a.agent_ctx.file_edit_state is a.file_edit_state
    assert a.agent_ctx is not b.agent_ctx
    assert a.agent_ctx.file_edit_state is not b.agent_ctx.file_edit_state
