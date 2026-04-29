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
- @function_tool(background=True) works
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

from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.function_tool import function_tool
from grasp_agents.agent.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
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
        super().__init__(name=name, description="Slow tool", background=True)
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
            background=True,
        )

    async def _run(
        self,
        inp: EchoInput,
        *,
        ctx: Any = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("bg tool exploded")


# --- Helpers ---


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
) -> tuple[AgentLoop[None], LLMAgentMemory, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentMemory()
    memory.reset(instructions="sys")
    memory.update([InputMessageItem.from_text("go", role="user")])

    executor = AgentLoop[None](
        agent_name="test",
        llm=llm,
        memory=memory,
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
    async for event in executor.execute_stream(ctx=ctx, exec_id="t"):
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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
        """BaseTool.background defaults to False, SlowTool sets True."""
        echo = EchoTool()
        slow = SlowTool()
        assert echo.background is False
        assert slow.background is True


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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
            _text_response("waiting"),
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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
        for tid in launched_ids:
            assert launched_ids[tid] == completed_ids[tid]

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
    """@function_tool(background=True) creates a background tool."""

    @pytest.mark.asyncio
    async def test_function_tool_background_flag(self):
        @function_tool(background=True)
        async def slow_research(query: str) -> str:
            """Do slow research."""
            await asyncio.sleep(0.05)
            return f"results for: {query}"

        assert slow_research.background is True
        assert slow_research.name == "slow_research"

    @pytest.mark.asyncio
    async def test_function_tool_default_not_background(self):
        @function_tool
        async def fast_lookup(query: str) -> str:
            """Fast lookup."""
            return f"found: {query}"

        assert fast_lookup.background is False


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
            lambda *, ctx, exec_id, response=None, **kw: response.output_text
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
