"""
Integration tests for the B2.c metadata borrows on ``AgentCheckpoint``:

- context_kind / context_data auto-round-trip
- prompt_cache_key round-trip
- pre-persist user input before first LLM call
"""

from __future__ import annotations

import logging
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
from grasp_agents.run_context import RunContext

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
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
    )
    return agent, ctx


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_current_schema_version_is_one(self) -> None:
        assert CURRENT_SCHEMA_VERSION == 1

    def test_new_fields_default_to_none(self) -> None:
        snap = AgentCheckpoint(session_key="s", processor_name="a", messages=[])
        assert snap.context_kind is None
        assert snap.context_data is None
        assert snap.prompt_cache_key is None


# ---------------------------------------------------------------------------
# context_kind round-trip
# ---------------------------------------------------------------------------


class TestContextRoundTrip:
    @pytest.mark.anyio
    async def test_pydantic_state_persisted_on_save(self) -> None:
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent([_text_response("hi")], session_key="s1", store=store)
        ctx.state.pathway_id = "p-1"
        ctx.state.count = 7

        await agent.run("hello", ctx=ctx)

        data = await store.load("s1/agent/test_agent")
        assert data is not None
        snap = AgentCheckpoint.model_validate_json(data)
        assert snap.context_kind == ContextKind.PYDANTIC
        assert snap.context_data == {"pathway_id": "p-1", "count": 7}

    @pytest.mark.anyio
    async def test_pydantic_state_restored_on_resume(self) -> None:
        store = InMemoryCheckpointStore()
        agent1, ctx1 = _make_agent(
            [_text_response("hi")], session_key="s1", store=store
        )
        ctx1.state.pathway_id = "p-42"
        ctx1.state.count = 3
        await agent1.run("hello", ctx=ctx1)

        # Fresh agent + ctx; state is default-initialized. Load rehydrates.
        agent2, ctx2 = _make_agent(
            [_text_response("follow")], session_key="s1", store=store
        )
        assert not ctx2.state.pathway_id  # baseline
        await agent2.load_checkpoint(ctx2)
        assert ctx2.state.pathway_id == "p-42"
        assert ctx2.state.count == 3

    @pytest.mark.anyio
    async def test_none_state_produces_omitted_kind(self) -> None:
        store = InMemoryCheckpointStore()
        agent = LLMAgent[str, str, None](
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("hi")]),
            stream_llm=True,
        )
        ctx: RunContext[None] = RunContext(
            checkpoint_store=store, session_key="s2", state=None
        )
        await agent.run("hello", ctx=ctx)

        snap = AgentCheckpoint.model_validate_json(
            await store.load("s2/agent/test_agent") or b"{}"
        )
        assert snap.context_kind == ContextKind.OMITTED
        assert snap.context_data is None


# ---------------------------------------------------------------------------
# prompt_cache_key
# ---------------------------------------------------------------------------


class TestPromptCacheKey:
    @pytest.mark.anyio
    async def test_round_trip_through_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()

        agent1, ctx1 = _make_agent(
            [_text_response("hi")], session_key="s1", store=store
        )
        # Simulate a provider writing the cache key post-LLM.
        agent1.prompt_cache_key = "cache-abc"
        await agent1.run("hello", ctx=ctx1)

        # Fresh agent resumes — cache key must restore.
        agent2, ctx2 = _make_agent(
            [_text_response("follow")], session_key="s1", store=store
        )
        assert agent2.prompt_cache_key is None
        await agent2.load_checkpoint(ctx2)
        assert agent2.prompt_cache_key == "cache-abc"

    @pytest.mark.anyio
    async def test_defaults_to_none_when_never_set(self) -> None:
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent([_text_response("hi")], session_key="s", store=store)
        await agent.run("hello", ctx=ctx)

        snap = AgentCheckpoint.model_validate_json(await store.load("s/agent/test_agent") or b"{}")
        assert snap.prompt_cache_key is None


# ---------------------------------------------------------------------------
# Pre-persist user input
# ---------------------------------------------------------------------------


class TestPrePersistInput:
    @pytest.mark.anyio
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
                input: Any,  # noqa: A002
                *,
                tools: Any = None,
                output_schema: Any = None,
                tool_choice: Any = None,
                **extra_llm_settings: Any,
            ) -> Any:
                del input, tools, output_schema, tool_choice, extra_llm_settings
                data = await store.load("s-pp/agent/test_agent")
                snap = (
                    AgentCheckpoint.model_validate_json(data)
                    if data is not None
                    else None
                )
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

        await agent.run("hello", ctx=ctx)

        assert observed_on_llm_call, "LLM was never called"
        first = observed_on_llm_call[0]
        assert first is not None, "no checkpoint saved before first LLM call"
        # The user's message should already be on the persisted transcript.
        assert any(getattr(m, "role", None) == "user" for m in first.messages)
