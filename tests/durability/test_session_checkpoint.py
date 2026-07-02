"""
Tests for the per-session ``SessionCheckpoint`` owned by ``SessionContext``:

- ``ctx.save_checkpoint()`` persists session-scoped state (serialized ``state``,
  ``session_metadata``) into one ``<session_key>/session`` record, written
  alongside every agent checkpoint;
- ``ctx.load_checkpoint()`` restores it exactly once per ctx (idempotent), so a second
  participant sharing the session cannot clobber live state;
- with every session-scoped feature off, no record is written at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    ContextKind,
    InMemoryCheckpointStore,
    SessionCheckpoint,
)
from grasp_agents.session_context import SessionContext
from tests._helpers import MockLLM, _text_response

pytestmark = pytest.mark.asyncio


class _AppState(BaseModel):
    pathway_id: str = ""
    count: int = 0


def _make_agent(
    responses: list[Any],
    *,
    session_key: str,
    store: InMemoryCheckpointStore,
    serialize_state: bool = True,
    session_metadata: dict[str, Any] | None = None,
) -> tuple[LLMAgent[str, str, _AppState], SessionContext[_AppState]]:
    ctx: SessionContext[_AppState] = SessionContext(
        checkpoint_store=store,
        session_key=session_key,
        state=_AppState(),
        serialize_state=serialize_state,
        session_metadata=session_metadata or {},
    )
    agent = LLMAgent[str, str, _AppState](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=responses),
        stream_llm=True,
    )
    return agent, ctx


async def _session_record(
    store: InMemoryCheckpointStore, session_key: str
) -> SessionCheckpoint | None:
    raw = await store.load(f"{session_key}/session")
    if raw is None:
        return None
    return SessionCheckpoint.model_validate_json(raw)


# ---------- save: one record per session, written with agent checkpoints ----


async def test_state_persisted_into_session_record() -> None:
    store = InMemoryCheckpointStore()
    agent, ctx = _make_agent([_text_response("hi")], session_key="s1", store=store)
    ctx.state.pathway_id = "p-1"
    ctx.state.count = 7

    await agent.run("hello")

    record = await _session_record(store, "s1")
    assert record is not None
    assert record.context_kind == ContextKind.PYDANTIC
    assert record.context_data == {"pathway_id": "p-1", "count": 7}


async def test_no_record_when_all_session_features_off() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("hi")], session_key="s2", store=store, serialize_state=False
    )

    await agent.run("hello")

    # The agent's own checkpoint exists; the session record does not.
    assert await store.load("s2/agent/test_agent") is not None
    assert await _session_record(store, "s2") is None


async def test_session_metadata_persisted() -> None:
    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response("hi")],
        session_key="s3",
        store=store,
        serialize_state=False,
        session_metadata={"pathway_id": "pw_123"},
    )

    await agent.run("hello")

    record = await _session_record(store, "s3")
    assert record is not None
    assert record.session_metadata == {"pathway_id": "pw_123"}
    # Metadata alone triggers the write, but state stays unserialized.
    assert record.context_kind is None


# ---------- load: cold-start rehydration, exactly once per ctx ----------


async def test_state_restored_on_cold_start_run() -> None:
    store = InMemoryCheckpointStore()
    agent1, ctx1 = _make_agent([_text_response("hi")], session_key="s4", store=store)
    ctx1.state.pathway_id = "p-42"
    ctx1.state.count = 3
    await agent1.run("hello")

    # Fresh process: new ctx + agent over the same store. The run itself
    # triggers ctx.load_checkpoint() — no explicit resume call needed.
    agent2, ctx2 = _make_agent(
        [_text_response("follow")], session_key="s4", store=store
    )
    assert not ctx2.state.pathway_id  # baseline
    await agent2.run("again")
    assert ctx2.state.pathway_id == "p-42"
    assert ctx2.state.count == 3


async def test_load_is_idempotent_and_does_not_clobber_live_state() -> None:
    store = InMemoryCheckpointStore()
    agent1, ctx1 = _make_agent([_text_response("hi")], session_key="s5", store=store)
    ctx1.state.pathway_id = "persisted"
    await agent1.run("hello")

    agent2, ctx2 = _make_agent(
        [_text_response("a"), _text_response("b")], session_key="s5", store=store
    )
    record = await ctx2.load_checkpoint()
    assert record is not None
    assert ctx2.state.pathway_id == "persisted"

    # Live mutation after the restore...
    ctx2.state.pathway_id = "mutated-live"
    # ...survives both an explicit re-load and further runs on the same ctx
    # (a second participant's run start must not re-restore over it).
    assert await ctx2.load_checkpoint() is None
    await agent2.run("next")
    assert ctx2.state.pathway_id == "mutated-live"


async def test_state_not_restored_when_serialization_off() -> None:
    store = InMemoryCheckpointStore()
    agent1, ctx1 = _make_agent(
        [_text_response("hi")], session_key="s6", store=store, serialize_state=False
    )
    ctx1.state.pathway_id = "p-99"
    await agent1.run("hello")

    agent2, ctx2 = _make_agent(
        [_text_response("follow")],
        session_key="s6",
        store=store,
        serialize_state=False,
    )
    await agent2.run("again")
    assert not ctx2.state.pathway_id


async def test_load_without_store_is_noop() -> None:
    ctx: SessionContext[_AppState] = SessionContext(state=_AppState())
    assert await ctx.load_checkpoint() is None


# ---------- the session record replaces per-agent state persistence ----------


async def test_agent_checkpoint_carries_no_session_state() -> None:
    store = InMemoryCheckpointStore()
    agent, ctx = _make_agent([_text_response("hi")], session_key="s7", store=store)
    ctx.state.pathway_id = "p-1"

    await agent.run("hello")

    raw = await store.load("s7/agent/test_agent")
    assert raw is not None
    head = raw.decode("utf-8")
    assert "context_kind" not in head
    assert "context_data" not in head
    assert "session_metadata" not in head
