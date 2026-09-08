"""
Deferred tool effects: a tool's durable side effect is held on the agent
context and run inside the checkpoint that persists the tool call, so it
commits with the transcript (atomically on a transactional store), and is
dropped when the call it belongs to leaves the transcript.
"""

from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from tests._helpers import MockLLM, _make_agent_ctx, _text_response, _tool_call_response


async def _noop() -> None:
    return None


def _recorder(log: list[str], name: str) -> Callable[[], Coroutine[Any, Any, None]]:
    async def run() -> None:
        log.append(name)

    return run


def _call(call_id: str) -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=call_id, name="tool", arguments="{}")


def _result(call_id: str) -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id=call_id, output="ok")


def _call_ids(agent_ctx: AgentContext) -> list[str | None]:
    return [e.call_id for e in agent_ctx.pending_effects]


# ---------------------------------------------------------------------------
# AgentContext outbox
# ---------------------------------------------------------------------------


class TestOutbox:
    def test_for_tool_call_shares_state_and_tags_effects(self) -> None:
        agent_ctx = _make_agent_ctx()
        bound = agent_ctx.for_tool_call("c1")

        assert bound is not agent_ctx
        assert bound.tool_call_id == "c1"
        assert agent_ctx.tool_call_id is None
        assert bound.cw is agent_ctx.cw
        assert bound.bg_tasks is agent_ctx.bg_tasks
        assert bound.pending_effects is agent_ctx.pending_effects

        bound.defer_effect(_noop)
        agent_ctx.defer_effect(_noop)
        assert _call_ids(agent_ctx) == ["c1", None]

    @pytest.mark.asyncio
    async def test_run_effects_runs_in_order_and_commit_drops_the_batch(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []
        agent_ctx.defer_effect(_recorder(log, "a"))
        agent_ctx.defer_effect(_recorder(log, "b"))

        ran = await agent_ctx.run_effects()

        # Still pending until the checkpoint they ran in is durable.
        assert log == ["a", "b"]
        assert ran == 2
        assert len(agent_ctx.pending_effects) == 2

        agent_ctx.commit_effects(ran)
        assert agent_ctx.pending_effects == []

    @pytest.mark.asyncio
    async def test_failing_effect_keeps_the_whole_batch_pending(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []

        async def boom() -> None:
            raise RuntimeError("boom")

        agent_ctx.defer_effect(_recorder(log, "a"))
        agent_ctx.defer_effect(boom)
        agent_ctx.defer_effect(_recorder(log, "c"))

        with pytest.raises(RuntimeError, match="boom"):
            await agent_ctx.run_effects()

        # Ran up to the failure, but nothing was dropped: a transactional store
        # rolled "a" back with the failed unit, so the retry must redo it.
        assert log == ["a"]
        assert len(agent_ctx.pending_effects) == 3

    @pytest.mark.asyncio
    async def test_effects_deferred_during_a_batch_survive_it(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []
        late = _recorder(log, "late")

        async def registers_another() -> None:
            agent_ctx.defer_effect(late)

        agent_ctx.defer_effect(registers_another)

        ran = await agent_ctx.run_effects()
        agent_ctx.commit_effects(ran)

        assert log == []
        assert [e.run for e in agent_ctx.pending_effects] == [late]

    def test_restore_drops_only_effects_of_pruned_calls(self) -> None:
        # A background task launched earlier (bg1) completes and defers its
        # effect while a later foreground round (fg1) is in flight; the round
        # fails and is pruned. Only fg1's effect goes: bg1's launch is still in
        # the transcript, and an effect deferred outside any call is unscoped.
        agent_ctx = _make_agent_ctx(
            messages=[
                InputMessageItem.from_text("go", role="user"),
                _call("bg1"),
                _result("bg1"),
                _call("fg1"),
                _result("fg1"),
            ]
        )
        agent_ctx.for_tool_call("fg1").defer_effect(_noop)
        agent_ctx.for_tool_call("bg1").defer_effect(_noop)
        agent_ctx.defer_effect(_noop)
        state = agent_ctx.snapshot()

        agent_ctx.cw.truncate_transcript(3)
        agent_ctx.restore(state)

        assert _call_ids(agent_ctx) == ["bg1", None]


# ---------------------------------------------------------------------------
# Through the agent
# ---------------------------------------------------------------------------


class _RecordingStore(InMemoryCheckpointStore):
    """Logs the unit-of-work boundaries and every head/record write."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.fail_next_save = False

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        self.log.append("begin")
        async with super().transaction():
            yield
        self.log.append("end")

    async def save(self, key: str, data: bytes) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("disk full")
        self.log.append("save")
        await super().save(key, data)


def _stamp_tool(log: list[str], store: _RecordingStore | None = None) -> Any:
    @function_tool
    async def stamp(text: str, agent_ctx: AgentContext) -> str:
        """Record ``text`` durably."""

        async def write() -> None:
            log.append("effect")

        agent_ctx.defer_effect(write)
        log.append("tool")
        if store is not None:
            store.fail_next_save = True
        return f"stamped {text}"

    return stamp


def _make_agent(store: _RecordingStore, tool: Any) -> LLMAgent[str, str, None]:
    ctx: SessionContext[None] = SessionContext(checkpoint_store=store, session_key="s1")
    return LLMAgent[str, str, None](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("stamp", '{"text": "x"}', "c1"),
                _text_response("done"),
            ]
        ),
        tools=[tool],
    )


class TestThroughTheAgent:
    @pytest.mark.asyncio
    async def test_effect_runs_once_inside_the_checkpoint_transaction(self) -> None:
        log: list[str] = []
        store = _RecordingStore(log)
        agent = _make_agent(store, _stamp_tool(log))

        out = await agent.run("go")

        assert out.payloads == ["done"]
        assert log.count("effect") == 1
        i_tool = log.index("tool")
        assert "effect" not in log[: i_tool + 1]  # deferred, not run by the tool
        # The checkpoint after the tool round: begin → effect → head save → end.
        after = log[i_tool + 1 :]
        i_begin = after.index("begin")
        i_effect = after.index("effect")
        i_save = after.index("save", i_effect)
        i_end = after.index("end", i_save)
        assert i_begin < i_effect < i_save < i_end
        assert agent.agent_ctx.pending_effects == []

    @pytest.mark.asyncio
    async def test_failed_checkpoint_keeps_the_round_and_its_effect(self) -> None:
        # The tool completed and its round is in the transcript, so settle
        # keeps it (only an in-flight round is pruned). The unit of work that
        # was to persist it failed, so on a transactional store the effect's
        # write was rolled back with the head — the effect must still be
        # pending for the next checkpoint to retry, or the transcript records
        # an artifact that does not exist.
        log: list[str] = []
        store = _RecordingStore(log)
        agent = _make_agent(store, _stamp_tool(log, store))

        with pytest.raises(ProcRunError):
            await agent.run("go")

        calls = [
            m.call_id for m in agent.transcript if isinstance(m, FunctionToolCallItem)
        ]
        assert calls == ["c1"]
        assert _call_ids(agent.agent_ctx) == ["c1"]

    def test_reset_and_replace_transcript_drop_orphaned_effects(self) -> None:
        agent = LLMAgent[str, str, None](
            name="test_agent",
            ctx=SessionContext[None](),
            llm=MockLLM(responses_queue=[]),
        )
        agent.agent_ctx.cw.add_messages(
            [_call("c1"), _result("c1"), _call("c2"), _result("c2")]
        )
        agent.agent_ctx.for_tool_call("c1").defer_effect(_noop)
        agent.agent_ctx.for_tool_call("c2").defer_effect(_noop)
        agent.agent_ctx.defer_effect(_noop)

        agent.replace_transcript([_call("c2"), _result("c2")])
        assert _call_ids(agent.agent_ctx) == ["c2", None]

        agent.reset_transcript()
        assert _call_ids(agent.agent_ctx) == [None]
