"""
Tests for the metadata fields on ``AgentCheckpoint``:

- context_kind / context_data auto-round-trip
- prompt_cache_key round-trip
- pre-persist user input before first LLM call
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    CURRENT_SCHEMA_VERSION,
    AgentCheckpoint,
    ContextKind,
    InMemoryCheckpointStore,
)
from grasp_agents.durability.checkpoints import SCHEMA_VERSION_SUMMARIES
from grasp_agents.run_context import RunContext
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
    load_agent_checkpoint,
)


class _MyState(BaseModel):
    pathway_id: str = ""
    count: int = 0


def _make_agent(
    responses: list[Any],
    *,
    session_key: str,
    store: InMemoryCheckpointStore,
    state_type: type[_MyState] = _MyState,
) -> tuple[LLMAgent[str, str, _MyState], RunContext[_MyState]]:
    agent = LLMAgent[str, str, _MyState](
        name="test_agent",
        llm=MockLLM(responses_queue=responses),
        stream_llm=True,
    )
    ctx: RunContext[_MyState] = RunContext(
        checkpoint_store=store,
        session_key=session_key,
        state=state_type(),
        serialize_state=True,
    )
    agent.on_adopted(ctx=ctx)
    return agent, ctx


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_current_schema_version_has_summary(self) -> None:
        # v10: AgentCheckpoint gained ``folds`` (compaction summaries carried in
        # the head) — additive over v9's step-rollback split.
        assert CURRENT_SCHEMA_VERSION == 10
        assert CURRENT_SCHEMA_VERSION in SCHEMA_VERSION_SUMMARIES

    def test_new_fields_default_to_none(self) -> None:
        snap = AgentCheckpoint(session_key="s", processor_name="a", messages=[])
        assert snap.context_kind is None
        assert snap.context_data is None
        assert snap.folds == []
        assert snap.current.prompt_cache_key is None
        assert snap.current.fs_snapshot_ref is None
        assert snap.current.agent_ctx_state.read_file_state == {}
        assert snap.current.agent_ctx_state.dotfile_overrides == []
        assert snap.current.agent_ctx_state.ipy_exec_context_id is None
        assert snap.current.agent_ctx_state.nb_exec_context_id is None


# ---------------------------------------------------------------------------
# context_kind round-trip
# ---------------------------------------------------------------------------


class TestContextRoundTrip:
    @pytest.mark.asyncio
    async def test_pydantic_state_persisted_on_save(self) -> None:
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent([_text_response("hi")], session_key="s1", store=store)
        ctx.state.pathway_id = "p-1"
        ctx.state.count = 7

        await agent.run("hello")

        data = await store.load("s1/agent/test_agent")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.context_kind == ContextKind.PYDANTIC
        assert snap.context_data == {"pathway_id": "p-1", "count": 7}

    @pytest.mark.asyncio
    async def test_pydantic_state_restored_on_resume(self) -> None:
        store = InMemoryCheckpointStore()
        agent1, ctx1 = _make_agent(
            [_text_response("hi")], session_key="s1", store=store
        )
        ctx1.state.pathway_id = "p-42"
        ctx1.state.count = 3
        await agent1.run("hello")

        # Fresh agent + ctx; state is default-initialized. Load rehydrates.
        agent2, ctx2 = _make_agent(
            [_text_response("follow")], session_key="s1", store=store
        )
        assert not ctx2.state.pathway_id  # baseline
        await agent2.load_checkpoint()
        assert ctx2.state.pathway_id == "p-42"
        assert ctx2.state.count == 3

    @pytest.mark.asyncio
    async def test_none_state_produces_omitted_kind(self) -> None:
        store = InMemoryCheckpointStore()
        agent = LLMAgent[str, str, None](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("hi")]),
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store,
            session_key="s2",
            state=None,
            serialize_state=True,
        )
        agent.on_adopted(ctx=ctx)
        await agent.run("hello")

        snap = AgentCheckpoint.model_validate_json(
            await store.load("s2/agent/test_agent") or b"{}"
        )
        assert snap.context_kind == ContextKind.OMITTED
        assert snap.context_data is None


# ---------------------------------------------------------------------------
# serialize_state opt-in (default off)
# ---------------------------------------------------------------------------


class TestSerializeStateOptIn:
    @pytest.mark.asyncio
    async def test_state_not_persisted_by_default(self) -> None:
        store = InMemoryCheckpointStore()
        agent = LLMAgent[str, str, _MyState](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("hi")]),
            stream_llm=True,
        )
        # No serialize_state -> defaults to False.
        ctx: RunContext[_MyState] = RunContext(
            checkpoint_store=store, session_key="s-off", state=_MyState()
        )
        ctx.state.pathway_id = "p-1"
        agent.on_adopted(ctx=ctx)
        await agent.run("hello")

        snap = AgentCheckpoint.model_validate_json(
            await store.load("s-off/agent/test_agent") or b"{}"
        )
        assert snap.context_kind is None
        assert snap.context_data is None

    @pytest.mark.asyncio
    async def test_state_not_restored_by_default(self) -> None:
        store = InMemoryCheckpointStore()
        agent1 = LLMAgent[str, str, _MyState](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("hi")]),
            stream_llm=True,
        )
        ctx1: RunContext[_MyState] = RunContext(
            checkpoint_store=store, session_key="s-off2", state=_MyState()
        )
        ctx1.state.pathway_id = "p-99"
        agent1.on_adopted(ctx=ctx1)
        await agent1.run("hello")

        # Fresh agent + baseline state; resume must leave state untouched
        agent2 = LLMAgent[str, str, _MyState](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("follow")]),
            stream_llm=True,
        )
        ctx2: RunContext[_MyState] = RunContext(
            checkpoint_store=store, session_key="s-off2", state=_MyState()
        )
        agent2.on_adopted(ctx=ctx2)
        await agent2.load_checkpoint()
        assert not ctx2.state.pathway_id


# ---------------------------------------------------------------------------
# prompt_cache_key
# ---------------------------------------------------------------------------


class TestPromptCacheKey:
    @pytest.mark.asyncio
    async def test_round_trip_through_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()

        agent1, ctx1 = _make_agent(
            [_text_response("hi")], session_key="s1", store=store
        )
        # Simulate a provider writing the cache key post-LLM.
        agent1.prompt_cache_key = "cache-abc"
        await agent1.run("hello")

        # Fresh agent resumes — cache key must restore.
        agent2, ctx2 = _make_agent(
            [_text_response("follow")], session_key="s1", store=store
        )
        assert agent2.prompt_cache_key is None
        await agent2.load_checkpoint()
        assert agent2.prompt_cache_key == "cache-abc"

    @pytest.mark.asyncio
    async def test_defaults_to_none_when_never_set(self) -> None:
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent([_text_response("hi")], session_key="s", store=store)
        await agent.run("hello")

        snap = AgentCheckpoint.model_validate_json(
            await store.load("s/agent/test_agent") or b"{}"
        )
        assert snap.current.prompt_cache_key is None


# ---------------------------------------------------------------------------
# Pre-persist user input
# ---------------------------------------------------------------------------


class TestPrePersistInput:
    @pytest.mark.asyncio
    async def test_checkpoint_exists_before_first_llm_call(self) -> None:
        """
        When the LLM runs, a checkpoint must already be on disk with the
        user input — so a crash during the call loses nothing.
        """
        store = InMemoryCheckpointStore()
        observed_on_llm_call: list[AgentCheckpoint | None] = []

        class _CheckingLLM(MockLLM):
            async def _generate_response_once(
                self,
                input: Any,
                *,
                tools: Any = None,
                output_schema: Any = None,
                tool_choice: Any = None,
                **extra_llm_settings: Any,
            ) -> Any:
                del input, tools, output_schema, tool_choice, extra_llm_settings
                snap = await load_agent_checkpoint(store, "s-pp/agent/test_agent")
                observed_on_llm_call.append(snap)
                # Pop a response the test-local way.
                assert self.responses_queue
                return self.responses_queue.pop(0)

        agent = LLMAgent[str, str, _MyState](
            name="test_agent",
            llm=_CheckingLLM(responses_queue=[_text_response("hi")]),
            stream_llm=True,
        )
        ctx: RunContext[_MyState] = RunContext(
            checkpoint_store=store,
            session_key="s-pp",
            state=_MyState(),
        )
        agent.on_adopted(ctx=ctx)

        await agent.run("hello")

        assert observed_on_llm_call, "LLM was never called"
        first = observed_on_llm_call[0]
        assert first is not None, "no checkpoint saved before first LLM call"
        # The user's message should already be on the persisted transcript.
        assert any(getattr(m, "role", None) == "user" for m in first.messages)
