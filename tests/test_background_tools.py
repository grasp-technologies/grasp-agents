"""
Tests for background tools feature.

Verifies that:
- Background tools return immediately with placeholder and run as asyncio tasks
- Results are delivered as notifications (user messages) in PRE-ACT drain
- Mixed immediate + background calls work correctly
- Loop stays alive (suppresses FINAL_ANSWER) while tasks are pending
- Efficient waiting: FIRST_COMPLETED when LLM has nothing to do
- Multiple background tasks drain correctly
- Background task failures are reported as error notifications
- Cancellation on max_turns and finally block
- BackgroundTaskLaunchedEvent and BackgroundTaskCompletedEvent lifecycle
- @function_tool(auto_background_at=0) works
"""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

from grasp_agents.agent.agent_loop import AgentLoop
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskLaunchedEvent,
    Event,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

# ---------- Infrastructure ----------


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _text_response(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=_make_usage(),
    )


def _tool_call_response(name: str, arguments: str, call_id: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        ],
        usage_with_cost=_make_usage(),
    )


def _multi_tool_call_response(
    calls: list[tuple[str, str, str]],
) -> Response:
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(call_id=cid, name=name, arguments=args)
            for name, args, cid in calls
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        seq = 0
        seq += 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


# --- Tools ---


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, Any, Any]):
    """Immediate tool that echoes input."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        return f"echo: {inp.text}"


class SlowTool(BaseTool[EchoInput, Any, Any]):
    """Background tool that simulates a slow operation."""

    def __init__(self, delay: float = 0.05, name: str = "slow") -> None:
        super().__init__(name=name, description="Slow tool", auto_background_at=0)
        self._delay = delay

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        await asyncio.sleep(self._delay)
        return f"slow: {inp.text}"


class FailingBgTool(BaseTool[EchoInput, Any, Any]):
    """Background tool that raises an exception."""

    def __init__(self) -> None:
        super().__init__(
            name="failing_bg",
            description="Fails",
            auto_background_at=0,
        )

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("bg tool exploded")


class FireAndForgetTool(BaseTool[EchoInput, Any, Any]):
    """Background tool whose result does NOT gate the final answer."""

    def __init__(self, delay: float = 0.05) -> None:
        super().__init__(
            name="fire_and_forget",
            description="Non-blocking background tool",
            auto_background_at=0,
            blocks_final_answer=False,
        )
        self._delay = delay

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        await asyncio.sleep(self._delay)
        return f"fnf: {inp.text}"


class BigOutputTool(BaseTool[EchoInput, Any, Any]):
    """Background tool returning a large result, with a small inline cap."""

    def __init__(self, *, size: int = 1000, cap: int = 100) -> None:
        super().__init__(
            name="big",
            description="Big-output background tool",
            auto_background_at=0,
            max_inline_result_chars=cap,
        )
        self._size = size

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        del inp
        return "X" * self._size


# --- Helpers ---


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
) -> tuple[AgentLoop[None], LLMAgentTranscript, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentTranscript()
    memory.reset(instructions="sys")
    memory.update([InputMessageItem.from_text("go", role="user")])

    ctx = RunContext[None](state=None)
    executor = AgentLoop[None](
        agent_name="test",
        llm=llm,
        transcript=memory,
        ctx=ctx,
        tools=tools,
        max_turns=max_turns,
        stream_llm=False,
    )
    return executor, memory, llm


async def _collect_events(
    executor: AgentLoop[None],
    ctx: RunContext[None],
) -> list[Event[Any]]:
    events: list[Event[Any]] = []
    async for event in executor.execute_stream(exec_id="t"):
        events.append(event)
    return events


# ---------- Tests ----------


class TestBackgroundToolLaunch:
    """Background tools return placeholder and spawn async tasks."""

    @pytest.mark.asyncio
    async def test_background_tool_returns_placeholder_then_delivers_result(self):
        """
        Flow: LLM calls slow(bg) → placeholder returned → LLM says "waiting"
        → PRE-ACT drains result → LLM sees notification → final answer.
        """
        responses = [
            _tool_call_response("slow", '{"text":"research"}', "tc1"),
            # Turn 1: LLM responds while bg task pending (suppressed as final)
            _text_response("waiting for results"),
            # Turn 2: after drain, LLM gives final answer
            _text_response("here are the results"),
        ]
        executor, memory, llm = _make_executor(
            responses,
            tools=[SlowTool(delay=0.1)],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        # Should have launched event
        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        assert len(launched) == 1
        assert launched[0].data.tool_name == "slow"

        # Should have completed event
        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].data.tool_name == "slow"

        # Notification should be in memory (XML-tagged format)
        user_msgs = [
            e
            for e in events
            if isinstance(e, UserMessageEvent) and "task_notification" in str(e.data)
        ]
        assert len(user_msgs) >= 1
        assert "slow: research" in str(user_msgs[0].data)

    @pytest.mark.asyncio
    async def test_background_tool_flag_on_base_tool(self):
        """BaseTool.auto_background_at defaults to None; SlowTool sets 0."""
        echo = EchoTool()
        slow = SlowTool()
        assert echo.auto_background_at is None
        assert slow.auto_background_at == 0


class TestMixedImmediateAndBackground:
    """Immediate and background tools called in the same turn."""

    @pytest.mark.asyncio
    async def test_mixed_calls_execute_correctly(self):
        """
        LLM calls echo(immediate) + slow(bg) in same turn.
        Echo returns real result, slow returns placeholder.
        After drain, slow result appears as notification.
        """
        responses = [
            _multi_tool_call_response(
                [
                    ("echo", '{"text":"fast"}', "tc1"),
                    ("slow", '{"text":"slow_data"}', "tc2"),
                ]
            ),
            # Turn 1: text while bg pending → suppressed
            _text_response("processing"),
            # Turn 2: after drain → final answer
            _text_response("all done"),
        ]
        executor, memory, _ = _make_executor(
            responses,
            tools=[EchoTool(), SlowTool(delay=0.05)],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        assert len(launched) == 1
        assert launched[0].data.tool_name == "slow"

        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(completed) == 1


class TestFinalAnswerSuppression:
    """Loop stays alive while background tasks are pending."""

    @pytest.mark.asyncio
    async def test_final_answer_suppressed_while_tasks_pending(self):
        """
        LLM calls bg tool, then emits what would be a final answer.
        The loop should suppress it and continue until bg task completes.
        """
        responses = [
            _tool_call_response("slow", '{"text":"data"}', "tc1"),
            _text_response("premature final"),  # suppressed
            _text_response("real final"),  # after drain
        ]
        executor, _, llm = _make_executor(
            responses,
            tools=[SlowTool(delay=0.05)],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        await _collect_events(executor, ctx)

        # LLM was called 3 times (tool call + suppressed + final)
        assert llm.call_count == 3


class TestBackgroundTaskFailure:
    """Background task exceptions are delivered as error notifications."""

    @pytest.mark.asyncio
    async def test_failed_bg_task_delivers_error_notification(self):
        """
        A background tool that raises should produce a notification
        with the error message, not crash the loop.
        """
        responses = [
            _tool_call_response("failing_bg", '{"text":"boom"}', "tc1"),
            _text_response("waiting"),  # suppressed while pending
            _text_response("handled error"),  # after drain
        ]
        executor, _, _ = _make_executor(
            responses,
            tools=[FailingBgTool()],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        # Should have completed event even on failure
        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(completed) == 1

        # Error appears in the notification (BaseTool catches and returns ToolErrorInfo)
        notification_msgs = [
            e
            for e in events
            if isinstance(e, UserMessageEvent)
            and "failing_bg" in str(e.data)
            and "failed" in str(e.data)
        ]
        assert len(notification_msgs) == 1
        assert "bg tool exploded" in str(notification_msgs[0].data)


class TestMultipleBackgroundTasks:
    """Multiple background tasks launched and drained."""

    @pytest.mark.asyncio
    async def test_multiple_bg_tasks_all_drain(self):
        """Two background tasks launched in same turn both deliver results."""
        responses = [
            _multi_tool_call_response(
                [
                    ("slow_a", '{"text":"a"}', "tc1"),
                    ("slow_b", '{"text":"b"}', "tc2"),
                ]
            ),
            _text_response("waiting"),  # suppressed (tasks may still be pending)
            _text_response("still waiting"),  # extra in case tasks drain separately
            _text_response("got both"),  # after all drained
        ]
        executor, _, _ = _make_executor(
            responses,
            tools=[
                SlowTool(delay=0.05, name="slow_a"),
                SlowTool(delay=0.05, name="slow_b"),
            ],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        assert len(launched) == 2

        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(completed) == 2

        completed_names = {e.data.tool_name for e in completed}
        assert completed_names == {"slow_a", "slow_b"}

    @pytest.mark.asyncio
    async def test_multiple_bg_tasks_ids_match_launch_to_completion(self):
        """
        Task IDs in placeholders, launched events, notifications, and
        completed events are all consistent and allow correlation.
        """
        responses = [
            _multi_tool_call_response(
                [
                    ("slow_a", '{"text":"aaa"}', "tc1"),
                    ("slow_b", '{"text":"bbb"}', "tc2"),
                ]
            ),
            _text_response("waiting"),  # suppressed (tasks may still be pending)
            # ``drain(wait=True)`` returns on FIRST_COMPLETED, so two equal-delay
            # tasks can drain on separate turns under load — give the loop a
            # spare turn so it never runs out of responses mid-drain.
            _text_response("still waiting"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(
            responses,
            tools=[
                SlowTool(delay=0.05, name="slow_a"),
                SlowTool(delay=0.05, name="slow_b"),
            ],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(launched) == 2
        assert len(completed) == 2

        # Build mappings: task_id → tool_name from launched events
        launched_ids = {e.data.task_id: e.data.tool_name for e in launched}
        completed_ids = {e.data.task_id: e.data.tool_name for e in completed}

        # Same set of task IDs
        assert set(launched_ids) == set(completed_ids)
        # Tool names match per task_id
        for tid, name in launched_ids.items():
            assert name == completed_ids[tid]

        # Placeholder tool outputs contain the matching task IDs
        from grasp_agents.types.events import ToolMessageEvent

        tool_msgs = [e for e in events if isinstance(e, ToolMessageEvent)]
        placeholders = [m for m in tool_msgs if "background" in str(m.data).lower()]
        assert len(placeholders) == 2
        for ph in placeholders:
            # Each placeholder should contain exactly one of the task IDs
            matches = [tid for tid in launched_ids if tid in str(ph.data)]
            assert len(matches) == 1

        # Notification user messages contain the matching task IDs
        notifications = [
            e
            for e in events
            if isinstance(e, UserMessageEvent) and "completed" in str(e.data).lower()
        ]
        assert len(notifications) == 2
        for notif in notifications:
            matches = [tid for tid in launched_ids if tid in str(notif.data)]
            assert len(matches) == 1
            tid = matches[0]
            # The notification tool name matches the launched tool name
            assert launched_ids[tid] in str(notif.data)


class TestCancellationOnMaxTurns:
    """Background tasks are cancelled when max_turns is reached."""

    @pytest.mark.asyncio
    async def test_bg_tasks_cancelled_on_max_turns(self):
        """
        With max_turns=1, bg task still pending at turn limit should
        be cancelled, not left running.
        """
        # Very slow tool that won't finish before max_turns
        very_slow = SlowTool(delay=10.0, name="very_slow")

        responses = [
            _tool_call_response("very_slow", '{"text":"x"}', "tc1"),
            # Turn 1 ACT (consumed before MAX_TURNS check)
            _text_response("still thinking"),
            # _force_generate_final_answer_stream makes another LLM call
            _text_response("forced final"),
        ]
        executor, _, _ = _make_executor(
            responses,
            tools=[very_slow],
            max_turns=1,
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        # Task should have been launched but not completed
        launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
        assert len(launched) == 1

        completed = [e for e in events if isinstance(e, BackgroundTaskCompletedEvent)]
        assert len(completed) == 0

        # Verify tasks dict is clean
        assert not executor.bg_tasks.has_pending


class TestFunctionToolBackground:
    """@function_tool(auto_background_at=0) creates a background tool."""

    @pytest.mark.asyncio
    async def test_function_tool_background_flag(self):
        @function_tool(auto_background_at=0)
        async def slow_research(query: str) -> str:
            """Do slow research."""
            await asyncio.sleep(0.05)
            return f"results for: {query}"

        assert slow_research.auto_background_at == 0
        assert slow_research.name == "slow_research"

    @pytest.mark.asyncio
    async def test_function_tool_default_not_background(self):
        @function_tool
        async def fast_lookup(query: str) -> str:
            """Fast lookup."""
            return f"found: {query}"

        assert fast_lookup.auto_background_at is None


class TestNoBackgroundToolsNoop:
    """When no background tools are used, all new paths are no-ops."""

    @pytest.mark.asyncio
    async def test_immediate_only_unchanged_behavior(self):
        """Standard tool call flow works exactly as before."""
        responses = [
            _tool_call_response("echo", '{"text":"hi"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, llm = _make_executor(
            responses,
            tools=[EchoTool()],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        ctx = RunContext[None]()
        events = await _collect_events(executor, ctx)

        # No background events
        bg_events = [
            e
            for e in events
            if isinstance(
                e,
                BackgroundTaskLaunchedEvent | BackgroundTaskCompletedEvent,
            )
        ]
        assert len(bg_events) == 0

        # Normal 2-call flow
        assert llm.call_count == 2


class TestBlocksFinalAnswerAttribute:
    """``blocks_final_answer`` is a tool attribute, not implied by backgrounding."""

    @pytest.mark.asyncio
    async def test_default_true_and_overrides(self):
        from grasp_agents.tools.bash import Bash

        # Default on BaseTool is True (you wait for what you asked for)...
        assert EchoTool().blocks_final_answer is True
        assert SlowTool().blocks_final_answer is True
        # ...Bash opts out (notify-and-continue)...
        assert Bash().blocks_final_answer is False
        # ...and any tool can opt out explicitly.
        assert FireAndForgetTool().blocks_final_answer is False

    @pytest.mark.asyncio
    async def test_function_tool_forwards_flag(self):
        @function_tool(auto_background_at=0, blocks_final_answer=False)
        async def bgfn(query: str) -> str:
            """A fire-and-forget function tool."""
            return query

        assert bgfn.auto_background_at == 0
        assert bgfn.blocks_final_answer is False


class TestFireAndForgetSpawn:
    """A spawned task with ``blocks_final_answer=False`` never gates the answer."""

    @pytest.mark.asyncio
    async def test_non_blocking_spawn_does_not_suppress_final_answer(self):
        # Slow enough to still be running when the final answer is emitted, so a
        # *blocking* task would suppress it — a non-blocking one must not.
        responses = [
            _tool_call_response("fire_and_forget", '{"text":"x"}', "tc1"),
            _text_response("final"),
        ]
        executor, _, llm = _make_executor(
            responses,
            tools=[FireAndForgetTool(delay=10.0)],
        )
        executor.final_answer_extractor = (
            lambda *, exec_id, response=None, **kw: response.output_text
            if response and not response.tool_call_items
            else None
        )

        await _collect_events(executor, RunContext[None]())

        # Only 2 LLM calls (tool call + final answer): no suppression turn,
        # because the non-blocking task never registers as pending.
        assert llm.call_count == 2


class TestCapAndDeferDelivery:
    """A large result is excerpted in the note + retained for a TaskOutput read."""

    @pytest.mark.asyncio
    async def test_large_result_excerpted_kept_and_pollable(self):
        from grasp_agents.tools.task_tools import TaskIdInput, TaskOutput

        tool = BigOutputTool(size=1000, cap=100)
        executor, _, _ = _make_executor([], tools=[tool])
        mgr = executor.bg_tasks
        ctx = executor.ctx

        call = FunctionToolCallItem(call_id="c1", name="big", arguments='{"text":"x"}')
        _note, event = await mgr.spawn(
            call,
            tool,
            EchoInput(text="x"),
            ctx=ctx,
            exec_id="t",
            agent_ctx=executor.agent_ctx,
        )
        task_id = event.data.task_id

        await mgr.wait_idle()
        notes = [
            str(e.data)
            async for e in mgr.drain(exec_id="t", ctx=ctx)
            if isinstance(e, UserMessageEvent)
        ]
        assert len(notes) == 1
        # Excerpted, with a pointer to TaskOutput for the rest.
        assert "chars omitted" in notes[0]
        assert "TaskOutput" in notes[0]
        # Even though BigOutputTool blocks the final answer by default, once its
        # note is delivered it no longer gates it (announced), yet it is kept so
        # the full result is still readable.
        assert not mgr.has_pending
        assert task_id in mgr._tasks  # pyright: ignore[reportPrivateUsage]

        out = await TaskOutput(mgr)._run(TaskIdInput(task_id=task_id))
        assert out.status == "completed"
        assert out.result == "X" * 1000  # full, untruncated
        # Reading a finished task drops it.
        assert task_id not in mgr._tasks  # pyright: ignore[reportPrivateUsage]


class TestDurableTaskRecords:
    """Every backgrounded task gets a TaskRecord; resume surfaces interrupted ones."""

    @pytest.mark.asyncio
    async def test_nonresumable_spawn_persists_record_and_resume_interrupts(self):
        from grasp_agents.agent.background_tasks import BackgroundTaskManager
        from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
        from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
        from grasp_agents.durability.store_keys import task_prefix
        from grasp_agents.durability.task_record import TaskRecord, TaskStatus

        store = InMemoryCheckpointStore()
        ctx = RunContext[None](state=None, checkpoint_store=store, session_key="s1")

        # A non-None path is required for a backgrounded call to be keyed +
        # persisted (``make_tool_call_path(None, ...)`` is ``None``).
        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="t", transcript=transcript, tools={}, path=[]
        )

        # A non-resumable, long-running spawned task.
        tool = SlowTool(delay=10.0)
        call = FunctionToolCallItem(call_id="c1", name="slow", arguments='{"text":"x"}')
        await mgr.spawn(call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t")

        # It got a PENDING record even though the tool is not resumable.
        keys = await store.list_keys(task_prefix("s1"))
        recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
        assert recs
        assert all(r.status == TaskStatus.PENDING for r in recs)

        # Simulate a crash: drop the in-flight task without finalizing the record.
        for pt in list(mgr._tasks.values()):  # pyright: ignore[reportPrivateUsage]
            pt.task.cancel()

        # A fresh manager on the same session = a restart.
        t2 = LLMAgentTranscript()
        t2.reset(instructions="sys")
        mgr2 = BackgroundTaskManager[None](
            agent_name="t", transcript=t2, tools={}, path=[]
        )
        await mgr2.resume_durable(ctx=ctx, exec_id="t")

        joined = "\n".join(str(m) for m in t2.messages)
        assert "Session resumed from a checkpoint" in joined  # the framing line
        assert "interrupted" in joined
        assert "slow" in joined

        # The record is now terminal, so a second resume surfaces nothing.
        t3 = LLMAgentTranscript()
        t3.reset(instructions="sys")
        mgr3 = BackgroundTaskManager[None](
            agent_name="t", transcript=t3, tools={}, path=[]
        )
        await mgr3.resume_durable(ctx=ctx, exec_id="t")
        assert not any("interrupted" in str(m) for m in t3.messages)

    @pytest.mark.asyncio
    async def test_failed_spawn_persists_failed_record(self):
        from grasp_agents.agent.background_tasks import BackgroundTaskManager
        from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
        from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
        from grasp_agents.durability.store_keys import task_prefix
        from grasp_agents.durability.task_record import TaskRecord, TaskStatus

        store = InMemoryCheckpointStore()
        ctx = RunContext[None](state=None, checkpoint_store=store, session_key="s1")
        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="t", transcript=transcript, tools={}, path=[]
        )

        call = FunctionToolCallItem(
            call_id="c1", name="failing_bg", arguments='{"text":"x"}'
        )
        _note, event = await mgr.spawn(
            call, FailingBgTool(), EchoInput(text="x"), ctx=ctx, exec_id="t"
        )
        # Wait for the task to finish so _consume persists the outcome.
        await mgr.get(event.data.task_id).task

        # A genuine runtime failure is recorded as FAILED with its error (so a
        # crash before drain leaves a terminal record, not a re-runnable PENDING).
        keys = await store.list_keys(task_prefix("s1"))
        rec = TaskRecord.model_validate_json(await store.load(keys[0]))
        assert rec.status == TaskStatus.FAILED
        assert rec.error is not None
        assert "exploded" in rec.error
