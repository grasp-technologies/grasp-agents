"""
Unit tests for the ``@agent.add_state_builder`` hook.

Contract (``docs/roadmap/03-checkpointing-and-sessions.md`` §2):

1. Fires exactly once per resume (``load_checkpoint`` returned non-None).
2. Does NOT fire on fresh init (``add_memory_builder`` handles that).
3. Has access to the loaded checkpoint and ``RunContext``.
4. Must be async.
5. Fires AFTER conversation messages have been restored into memory, so
   the hook can inspect them if needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
)
from grasp_agents.run_context import RunContext

# Reuse the hand-rolled MockLLM / text-response helpers so every async
# test-support file doesn't drift.
from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
)


class _AppState(BaseModel):
    pathway_id: str = ""
    loaded_from_db: bool = False
    message_count_at_load: int = -1


def _make_agent(
    responses: list[Any],
    *,
    session_key: str,
    store: InMemoryCheckpointStore,
) -> tuple[LLMAgent[str, str, _AppState], RunContext[_AppState]]:
    agent = LLMAgent[str, str, _AppState](
        name="test_agent",
        llm=MockLLM(responses_queue=responses),
        stream_llm=True,
    )
    ctx: RunContext[_AppState] = RunContext(
        checkpoint_store=store,
        session_key=session_key,
        state=_AppState(),
    )
    return agent, ctx


class TestStateBuilder:
    @pytest.mark.anyio
    async def test_does_not_fire_on_fresh_init(self) -> None:
        """With no saved checkpoint, state builder must NOT be called."""
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [_text_response("hello")], session_key="s1", store=store
        )
        calls: list[AgentCheckpoint] = []

        async def rebuild(  # noqa: RUF029
            *,
            checkpoint: AgentCheckpoint,
            ctx: RunContext[_AppState],
            exec_id: str,
        ) -> None:
            del ctx, exec_id
            calls.append(checkpoint)

        agent.add_state_builder(rebuild)

        await agent.run("hi", ctx=ctx)
        assert calls == []

    @pytest.mark.anyio
    async def test_fires_on_resume_with_checkpoint(self) -> None:
        """After a first run persists, a second agent's load triggers builder."""
        store = InMemoryCheckpointStore()

        # Seed: run once to persist a checkpoint.
        agent1, ctx1 = _make_agent(
            [_text_response("hello")], session_key="s1", store=store
        )
        await agent1.run("hi", ctx=ctx1)

        # Fresh agent; register state builder BEFORE load.
        agent2, ctx2 = _make_agent(
            [_text_response("follow")], session_key="s1", store=store
        )
        received: list[AgentCheckpoint] = []

        async def rebuild(  # noqa: RUF029
            *,
            checkpoint: AgentCheckpoint,
            ctx: RunContext[_AppState],
            exec_id: str,
        ) -> None:
            del exec_id
            received.append(checkpoint)
            ctx.state.loaded_from_db = True
            ctx.state.pathway_id = "p-42"
            ctx.state.message_count_at_load = len(agent2.memory.messages)

        agent2.add_state_builder(rebuild)

        cp = await agent2.load_checkpoint(ctx2)
        assert cp is not None
        assert len(received) == 1
        assert received[0].session_key == "s1"
        # Mutation through ctx.state takes effect.
        assert ctx2.state.loaded_from_db is True
        assert ctx2.state.pathway_id == "p-42"
        # Fires AFTER messages are restored — the count must reflect the
        # restored history.
        assert ctx2.state.message_count_at_load > 0

    @pytest.mark.anyio
    async def test_does_not_fire_when_memory_already_populated(self) -> None:
        """
        load_checkpoint short-circuits if memory is non-empty (the agent
        has already been run in-process) — state builder must not fire.
        """
        store = InMemoryCheckpointStore()
        agent, ctx = _make_agent(
            [_text_response("hello"), _text_response("again")],
            session_key="s1",
            store=store,
        )
        # Run once to populate memory in-process.
        await agent.run("hi", ctx=ctx)

        calls = 0

        async def rebuild(  # noqa: RUF029
            *,
            checkpoint: AgentCheckpoint,
            ctx: RunContext[_AppState],
            exec_id: str,
        ) -> None:
            del checkpoint, ctx, exec_id
            nonlocal calls
            calls += 1

        agent.add_state_builder(rebuild)

        result = await agent.load_checkpoint(ctx)
        assert result is None
        assert calls == 0

    @pytest.mark.anyio
    async def test_subclass_override_is_honored(self) -> None:
        """Overriding ``build_state_impl`` on a subclass works identically."""
        store = InMemoryCheckpointStore()
        _seed_agent, seed_ctx = _make_agent(
            [_text_response("seed")], session_key="s2", store=store
        )
        await _seed_agent.run("hi", ctx=seed_ctx)

        class _MyAgent(LLMAgent[str, str, _AppState]):
            build_state_calls: list[AgentCheckpoint]

            def __init__(self, **kw: Any) -> None:
                super().__init__(**kw)
                self.build_state_calls = []

            async def build_state_impl(
                self,
                *,
                checkpoint: AgentCheckpoint,
                ctx: RunContext[_AppState],
                exec_id: str,
            ) -> None:
                del exec_id
                self.build_state_calls.append(checkpoint)
                ctx.state.loaded_from_db = True

        agent = _MyAgent(
            name="test_agent",
            llm=MockLLM(responses_queue=[_text_response("follow")]),
            stream_llm=True,
        )
        ctx: RunContext[_AppState] = RunContext(
            checkpoint_store=store, session_key="s2", state=_AppState()
        )

        cp = await agent.load_checkpoint(ctx)
        assert cp is not None
        assert len(agent.build_state_calls) == 1
        assert ctx.state.loaded_from_db is True

    @pytest.mark.anyio
    async def test_state_persists_into_followup_run(self) -> None:
        """State mutated by the builder survives into the post-resume run."""
        store = InMemoryCheckpointStore()
        agent1, ctx1 = _make_agent(
            [_text_response("hello")], session_key="s3", store=store
        )
        await agent1.run("hi", ctx=ctx1)

        agent2, ctx2 = _make_agent(
            [_text_response("world")], session_key="s3", store=store
        )

        async def rebuild(  # noqa: RUF029
            *,
            checkpoint: AgentCheckpoint,
            ctx: RunContext[_AppState],
            exec_id: str,
        ) -> None:
            del checkpoint, exec_id
            ctx.state.pathway_id = "rebuilt"

        agent2.add_state_builder(rebuild)

        await agent2.load_checkpoint(ctx2)
        assert ctx2.state.pathway_id == "rebuilt"

        await agent2.run("follow up", ctx=ctx2)
        # State still set after a post-resume run completes.
        assert ctx2.state.pathway_id == "rebuilt"
