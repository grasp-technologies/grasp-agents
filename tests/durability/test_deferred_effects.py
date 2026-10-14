"""
Deferred tool effects: a tool's durable side effect is held in the agent's
outbox and run inside the checkpoint that persists the tool call, so it commits
with the transcript (atomically on a transactional store), and is dropped when
the call it belongs to leaves the transcript. Also covers the checkpoint unit of
work recovering from a save that failed partway.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
)
from grasp_agents.types.response import Response
from tests._helpers import (
    MockLLM,
    _make_agent_ctx,
    _make_usage,
    _text_response,
    _tool_call_response,
    _until,
)


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
    return [e.call_id for e in agent_ctx.effects.pending]


def _shape(messages: Sequence[InputItem]) -> list[tuple[str, str]]:
    shape: list[tuple[str, str]] = []
    for m in messages:
        if isinstance(m, FunctionToolCallItem):
            shape.append(("call", m.call_id))
        elif isinstance(m, FunctionToolOutputItem):
            shape.append(("output", m.call_id))
        elif isinstance(m, InputMessageItem):
            shape.append(("input", m.text))
        else:
            shape.append((type(m).__name__, ""))
    return shape


def _calls_response(*calls: tuple[str, str]) -> Response:
    """One response issuing several tool calls, as ``(tool_name, call_id)``."""
    return Response(
        model="mock",
        output=[
            FunctionToolCallItem(call_id=call_id, name=name, arguments='{"text": "x"}')
            for name, call_id in calls
        ],
        usage=_make_usage(),
    )


# ---------------------------------------------------------------------------
# Outbox
# ---------------------------------------------------------------------------


class TestOutbox:
    def test_for_tool_call_shares_state_and_tags_effects(self) -> None:
        agent_ctx = _make_agent_ctx()
        bound = agent_ctx.for_tool_call("c1")

        assert bound is not agent_ctx
        assert bound.cw is agent_ctx.cw
        assert bound.bg_tasks is agent_ctx.bg_tasks
        assert bound.effects is agent_ctx.effects

        bound.defer_effect(_noop)
        agent_ctx.defer_effect(_noop)
        assert _call_ids(agent_ctx) == ["c1", None]

    @pytest.mark.asyncio
    async def test_run_keeps_the_batch_until_commit(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []
        agent_ctx.defer_effect(_recorder(log, "a"))
        agent_ctx.defer_effect(_recorder(log, "b"))

        ran = await agent_ctx.effects.run(retire_each=False)

        # Still pending until the checkpoint they ran in is durable.
        assert log == ["a", "b"]
        assert ran == 2
        assert len(agent_ctx.effects.pending) == 2

        agent_ctx.effects.commit(ran)
        assert agent_ctx.effects.pending == ()

    @pytest.mark.asyncio
    async def test_retire_each_drops_effects_as_they_succeed(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []
        agent_ctx.defer_effect(_recorder(log, "a"))
        agent_ctx.defer_effect(_recorder(log, "b"))

        assert await agent_ctx.effects.run(retire_each=True) == 0
        assert log == ["a", "b"]
        assert agent_ctx.effects.pending == ()

    @pytest.mark.asyncio
    async def test_a_failing_effect_keeps_what_its_store_cannot_have_kept(
        self,
    ) -> None:
        async def boom() -> None:
            raise RuntimeError("boom")

        # A transactional unit rolls "a" back with the failure, so the whole
        # batch stays pending for the retry.
        rolled_back = _make_agent_ctx()
        log: list[str] = []
        rolled_back.defer_effect(_recorder(log, "a"))
        rolled_back.defer_effect(boom)
        rolled_back.defer_effect(_recorder(log, "c"))
        with pytest.raises(RuntimeError, match="boom"):
            await rolled_back.effects.run(retire_each=False)
        assert log == ["a"]
        assert len(rolled_back.effects.pending) == 3

        # Without transactions "a" stands, so only the failed effect and the
        # ones after it stay pending.
        kept = _make_agent_ctx()
        log = []
        kept.defer_effect(_recorder(log, "a"))
        kept.defer_effect(boom)
        kept.defer_effect(_recorder(log, "c"))
        with pytest.raises(RuntimeError, match="boom"):
            await kept.effects.run(retire_each=True)
        assert log == ["a"]
        pending = kept.effects.pending
        assert len(pending) == 2
        assert pending[0].run is boom

    @pytest.mark.asyncio
    async def test_effects_deferred_during_a_batch_survive_it(self) -> None:
        agent_ctx = _make_agent_ctx()
        log: list[str] = []
        late = _recorder(log, "late")

        async def registers_another() -> None:
            agent_ctx.defer_effect(late)

        agent_ctx.defer_effect(registers_another)

        ran = await agent_ctx.effects.run(retire_each=False)
        agent_ctx.effects.commit(ran)

        assert log == []
        assert [e.run for e in agent_ctx.effects.pending] == [late]

    def test_discard_drops_one_calls_effects(self) -> None:
        agent_ctx = _make_agent_ctx()
        agent_ctx.for_tool_call("c1").defer_effect(_noop)
        agent_ctx.for_tool_call("c1").defer_effect(_noop)
        agent_ctx.for_tool_call("c2").defer_effect(_noop)
        agent_ctx.defer_effect(_noop)

        assert agent_ctx.effects.discard("c1") == 2
        assert _call_ids(agent_ctx) == ["c2", None]
        assert agent_ctx.effects.discard(None) == 1
        assert _call_ids(agent_ctx) == ["c2"]

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
# Stores
# ---------------------------------------------------------------------------


class _Store(InMemoryCheckpointStore):
    """Tracks open transactions and can fail the next save of one key."""

    def __init__(self) -> None:
        super().__init__()
        self.in_transaction = False
        self.fail_save_of: str | None = None

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    async def save(self, key: str, data: bytes) -> None:
        if key == self.fail_save_of:
            self.fail_save_of = None
            raise RuntimeError("disk full")
        await super().save(key, data)


class _RollbackStore(_Store):
    """A transactional store: a unit that fails, at commit included, leaves no trace."""

    transactional: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_commit = False

    def _commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("commit failed")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        data = dict(self._data)
        logs = {k: list(v) for k, v in self._logs.items()}
        self.in_transaction = True
        try:
            yield
            self._commit()
        except BaseException:
            self._data, self._logs = data, logs
            raise
        finally:
            self.in_transaction = False


def _make_agent(
    store: _Store,
    tools: Sequence[Any],
    responses: list[Response] | None = None,
) -> LLMAgent[str, str, None]:
    ctx: SessionContext[None] = SessionContext(checkpoint_store=store, session_key="s1")
    return LLMAgent[str, str, None](
        name="test_agent",
        ctx=ctx,
        llm=MockLLM(
            responses_queue=responses
            if responses is not None
            else [
                _tool_call_response("stamp", '{"text": "x"}', "c1"),
                _text_response("done"),
            ]
        ),
        tools=list(tools),
    )


def _head_key(agent: LLMAgent[str, str, None]) -> str:
    key = agent._checkpoint_store_key(agent.ctx)  # pyright: ignore[reportPrivateUsage]
    assert key is not None
    return key


async def _cold_shape(store: _Store) -> list[tuple[str, str]]:
    cold = _make_agent(store, [], responses=[])
    assert await cold.load_checkpoint() is not None
    return _shape(cold.transcript)


def _stamp(
    on_call: Callable[[], None] = lambda: None,
    effect: Callable[[], Coroutine[Any, Any, None]] = _noop,
) -> Any:
    @function_tool
    async def stamp(text: str, agent_ctx: AgentContext) -> str:
        """Record ``text`` durably."""
        agent_ctx.defer_effect(effect)
        on_call()
        return f"stamped {text}"

    return stamp


def _arm_commit_failure(store: _RollbackStore) -> Callable[[], None]:
    def arm() -> None:
        store.fail_next_commit = True

    return arm


# ---------------------------------------------------------------------------
# Through the agent
# ---------------------------------------------------------------------------


class TestThroughTheAgent:
    @pytest.mark.asyncio
    async def test_effect_runs_once_inside_the_checkpoint_transaction(self) -> None:
        store = _Store()
        in_transaction: list[bool] = []
        ran_before_tool_returned: list[int] = []

        async def write() -> None:
            in_transaction.append(store.in_transaction)

        agent = _make_agent(
            store,
            [
                _stamp(
                    lambda: ran_before_tool_returned.append(len(in_transaction)), write
                )
            ],
        )

        out = await agent.run("go")

        assert out.payloads == ["done"]
        assert ran_before_tool_returned == [0]  # deferred, not run by the tool
        assert in_transaction == [True]
        assert agent.agent_ctx.effects.pending == ()

    @pytest.mark.asyncio
    async def test_failed_checkpoint_keeps_the_round_and_its_effect(self) -> None:
        # The tool completed and its round is in the transcript, so settle
        # keeps it (only an in-flight round is pruned). The unit of work that
        # was to persist it rolled back, the effect's write with it — so the
        # effect must still be pending for the next checkpoint to retry, or the
        # transcript records an artifact that does not exist.
        store = _RollbackStore()
        runs: list[str] = []
        agent = _make_agent(
            store, [_stamp(_arm_commit_failure(store), _recorder(runs, "effect"))]
        )

        with pytest.raises(ProcRunError):
            await agent.run("go")

        calls = [
            m.call_id for m in agent.transcript if isinstance(m, FunctionToolCallItem)
        ]
        assert calls == ["c1"]
        assert runs == ["effect"]
        assert _call_ids(agent.agent_ctx) == ["c1"]

    @pytest.mark.asyncio
    async def test_rolled_back_save_does_not_lose_the_round_from_the_log(
        self,
    ) -> None:
        store = _RollbackStore()
        agent = _make_agent(store, [_stamp(_arm_commit_failure(store))])

        with pytest.raises(ProcRunError):
            await agent.run("go")

        agent.agent_ctx.cw.add_messages(
            [InputMessageItem.from_text("next", role="user")]
        )
        await agent.save_checkpoint()

        live = _shape(agent.transcript)
        assert [kind for kind, _ in live] == ["input", "call", "output", "input"]
        assert await _cold_shape(store) == live

    @pytest.mark.asyncio
    async def test_failed_head_write_does_not_duplicate_the_round(self) -> None:
        # Without transactions the log append lands before the head write
        # fails; the next save must not append the same round again.
        store = _Store()
        head: list[str] = []

        def fail_head() -> None:
            store.fail_save_of = head[0]

        agent = _make_agent(store, [_stamp(fail_head)])
        head.append(_head_key(agent))

        with pytest.raises(ProcRunError):
            await agent.run("go")

        agent.agent_ctx.cw.add_messages(
            [InputMessageItem.from_text("next", role="user")]
        )
        await agent.save_checkpoint()

        live = _shape(agent.transcript)
        assert [kind for kind, _ in live] == ["input", "call", "output", "input"]
        assert await _cold_shape(store) == live

    @pytest.mark.asyncio
    async def test_deferred_flips_survive_a_rolled_back_save(self) -> None:
        store = _RollbackStore()
        agent = _make_agent(store, [], responses=[])
        key = "s1/task/test_agent/bg_1"
        record = TaskRecord(
            session_key="s1",
            task_id="bg_1",
            tool_call_id="c9",
            tool_name="slow",
            status=TaskStatus.COMPLETED,
            result="r",
        )
        await store.save(key, record.model_dump_json().encode())
        agent.agent_ctx.cw.add_messages([InputMessageItem.from_text("go", role="user")])
        bg_tasks = agent.agent_ctx.bg_tasks
        bg_tasks.restore_deferred_delivered({key: {"note_transcript_pos": 1}})

        async def status() -> TaskStatus:
            raw = await store.load(key)
            assert raw is not None
            return TaskRecord.model_validate_json(raw).status

        store.fail_next_commit = True
        with pytest.raises(RuntimeError, match="commit failed"):
            await agent.save_checkpoint()

        assert await status() is TaskStatus.COMPLETED
        assert key in bg_tasks.export_deferred_delivered()

        await agent.save_checkpoint()

        assert await status() is TaskStatus.DELIVERED
        assert bg_tasks.export_deferred_delivered() == {}

    @pytest.mark.asyncio
    async def test_effects_rerun_only_where_the_store_undid_them(self) -> None:
        # Without transactions, an effect that ran before the head write failed
        # stands and is retired: the retry does not run it again.
        plain = _Store()
        runs: list[str] = []
        head: list[str] = []

        def fail_head() -> None:
            plain.fail_save_of = head[0]

        agent = _make_agent(plain, [_stamp(fail_head, _recorder(runs, "effect"))])
        head.append(_head_key(agent))

        with pytest.raises(ProcRunError):
            await agent.run("go")
        assert runs == ["effect"]
        assert agent.agent_ctx.effects.pending == ()

        await agent.save_checkpoint()
        assert runs == ["effect"]

        # With transactions the failed unit rolled the effect's write back, so
        # the retry runs it again and exactly one row survives.
        rolling = _RollbackStore()
        tx_runs: list[str] = []

        async def write_row() -> None:
            tx_runs.append("effect")
            await rolling.save(f"artifact/report-{len(tx_runs)}", b"row")

        tx_agent = _make_agent(
            rolling, [_stamp(_arm_commit_failure(rolling), write_row)]
        )

        with pytest.raises(ProcRunError):
            await tx_agent.run("go")
        assert tx_runs == ["effect"]
        assert await rolling.list_keys("artifact/") == []
        assert len(tx_agent.agent_ctx.effects.pending) == 1

        await tx_agent.save_checkpoint()
        assert tx_runs == ["effect", "effect"]
        assert await rolling.list_keys("artifact/") == ["artifact/report-2"]
        assert tx_agent.agent_ctx.effects.pending == ()

    @pytest.mark.asyncio
    async def test_a_permanently_failing_effect_can_be_discarded(self) -> None:
        store = _Store()

        async def violates() -> None:
            raise RuntimeError("unique constraint violated")

        agent = _make_agent(
            store,
            [_stamp(effect=violates)],
            responses=[
                _tool_call_response("stamp", '{"text": "x"}', "c1"),
                _text_response("done"),
            ],
        )

        with pytest.raises(ProcRunError):
            await agent.run("go")

        with pytest.raises(RuntimeError, match="unique constraint") as exc_info:
            await agent.save_checkpoint()
        notes = getattr(exc_info.value, "__notes__", [])
        assert any("'c1'" in note and "discard" in note for note in notes)

        assert agent.agent_ctx.effects.discard("c1") == 1
        out = await agent.run("again")
        assert out.payloads == ["done"]

    @pytest.mark.asyncio
    async def test_an_in_flight_rounds_effects_are_dropped(self) -> None:
        store = _Store()
        runs: list[str] = []
        stalled = asyncio.Event()

        @function_tool
        async def stall(text: str) -> str:
            """Block until cancelled."""
            stalled.set()
            await asyncio.Event().wait()
            return text

        agent = _make_agent(
            store,
            [_stamp(effect=_recorder(runs, "effect")), stall],
            responses=[
                _calls_response(("stamp", "a"), ("stall", "b")),
                _text_response("done"),
            ],
        )

        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(stalled.wait(), timeout=5)
        await _until(lambda: _call_ids(agent.agent_ctx) == ["a"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert agent.agent_ctx.effects.pending == ()
        assert not any(isinstance(m, FunctionToolCallItem) for m in agent.transcript)

        out = await agent.run("again")
        assert out.payloads == ["done"]
        assert runs == []

    @pytest.mark.asyncio
    async def test_deadline_backgroundable_tools_scope_their_effects(self) -> None:
        store = _Store()
        tagged: list[str | None] = []
        runs: list[str] = []
        stalled = asyncio.Event()

        @function_tool(auto_background_at=5.0)
        async def stamp(text: str, agent_ctx: AgentContext) -> str:
            """Record ``text`` durably; backgrounded if it runs long."""
            agent_ctx.defer_effect(_recorder(runs, "effect"))
            tagged.extend(_call_ids(agent_ctx))
            return f"stamped {text}"

        @function_tool
        async def stall(text: str) -> str:
            """Block until cancelled."""
            stalled.set()
            await asyncio.Event().wait()
            return text

        agent = _make_agent(
            store,
            [stamp, stall],
            responses=[
                _tool_call_response("stamp", '{"text": "x"}', "c7"),
                _text_response("first"),
                _calls_response(("stamp", "a"), ("stall", "b")),
                _text_response("done"),
            ],
        )

        assert (await agent.run("go")).payloads == ["first"]
        assert tagged == ["c7"]
        assert runs == ["effect"]

        # Cancelled while its sibling stalls: the round is pruned, and the
        # effect goes with it instead of running at the next checkpoint.
        task = asyncio.create_task(agent.run("second"))
        await asyncio.wait_for(stalled.wait(), timeout=5)
        await _until(lambda: _call_ids(agent.agent_ctx) == ["a"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert agent.agent_ctx.effects.pending == ()
        assert (await agent.run("third")).payloads == ["done"]
        assert runs == ["effect"]

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
