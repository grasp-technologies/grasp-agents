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
    ctx: RunContext[None] | None = None,
) -> tuple[AgentLoop[None], LLMAgentTranscript, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentTranscript()
    memory.reset(instructions="sys")
    memory.update([InputMessageItem.from_text("go", role="user")])

    ctx = ctx if ctx is not None else RunContext[None](state=None)
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
    async def test_launch_note_wait_guidance_depends_on_blocking(self):
        """
        A blocking task's launch note tells the model it can reply tool-free to
        wait (the loop waits); a non-blocking one tells it to continue/answer —
        a tool-free reply there would finalize with a useless "I'll wait …".
        """
        from grasp_agents.agent.background_tasks import BackgroundTaskManager
        from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript

        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="a", transcript=transcript, tools={}, path=[]
        )
        blocking, nonblocking = SlowTool(), FireAndForgetTool()
        assert blocking.blocks_final_answer and not nonblocking.blocks_final_answer
        note_b = mgr._launch_note(  # pyright: ignore[reportPrivateUsage]
            blocking, "bg_1", backgrounded_after=None, log_path=None
        )
        note_n = mgr._launch_note(  # pyright: ignore[reportPrivateUsage]
            nonblocking, "bg_2", backgrounded_after=None, log_path=None
        )
        assert "reply WITHOUT calling any tool" in note_b
        assert "runs independently" not in note_b
        assert "runs independently" in note_n
        assert "reply WITHOUT calling any tool" not in note_n

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


class TestMaxTurnsKeepsBackgroundTasks:
    """Background tasks survive max_turns — resources are session-scoped."""

    @pytest.mark.asyncio
    async def test_bg_tasks_survive_max_turns(self):
        """
        With max_turns=1, a bg task still pending at the turn limit keeps
        running after the forced final answer; only an explicit teardown
        (``cancel_all`` — what ``LLMAgent.aclose`` calls) releases it.
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

        # The task outlives the run; explicit teardown releases it.
        assert executor.bg_tasks.has_live_tasks
        await executor.bg_tasks.cancel_all(ctx=ctx)
        assert not executor.bg_tasks.has_live_tasks


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

        # Default is True everywhere — you wait for what you asked for (the model
        # can't wait; only the loop can, by gating the answer on the result).
        assert EchoTool().blocks_final_answer is True
        assert SlowTool().blocks_final_answer is True
        assert Bash().blocks_final_answer is True
        # ...any tool can opt out explicitly for genuine fire-and-forget.
        assert Bash(blocks_final_answer=False).blocks_final_answer is False
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
    """A large result is excerpted in the completion note, then the task dropped."""

    @pytest.mark.asyncio
    async def test_large_result_excerpted_then_dropped(self):
        tool = BigOutputTool(size=1000, cap=100)
        executor, _, _ = _make_executor([], tools=[tool])
        mgr = executor.bg_tasks
        ctx = executor.ctx

        call = FunctionToolCallItem(call_id="c1", name="big", arguments='{"text":"x"}')
        _note, event = await mgr.run_backgroundable(
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
        # Excerpted — the full output belongs in the .grasp log, not the
        # transcript (this tool streams nothing, so there is no pointer here).
        assert "chars omitted" in notes[0]
        # Once its note is delivered the task no longer gates the answer and is
        # dropped — nothing is retained in memory for a poll.
        assert not mgr.has_pending
        assert task_id not in mgr._tasks  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_truncated_result_written_to_sidecar_file(self, tmp_path):
        """
        A non-streaming tool's over-cap result has no streamed ``.log`` to point
        at, so drain persists the full result to a ``.result`` sidecar and the
        excerpt marker points there — recoverable with ``Read`` / ``Grep``.
        """
        import re
        from pathlib import Path

        from grasp_agents.sandbox import local_environment

        env = local_environment(allowed_roots=[tmp_path])
        ctx = RunContext[None](environment=env)
        tool = BigOutputTool(size=1000, cap=100)
        executor, _, _ = _make_executor([], tools=[tool], ctx=ctx)
        mgr = executor.bg_tasks

        call = FunctionToolCallItem(call_id="c1", name="big", arguments='{"text":"x"}')
        _note, event = await mgr.run_backgroundable(
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
        assert "chars omitted" in notes[0]

        # The marker points at a .result sidecar holding the full result, even
        # though no .log was streamed.
        match = re.search(r"full output in (.+?)\]", notes[0])
        assert match is not None
        result_path = Path(match.group(1).strip())
        assert result_path.suffix == ".result"
        assert result_path.read_text().count("X") == 1000
        assert not list((tmp_path / ".grasp" / "tasks").glob("*.log"))
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
        await mgr.run_backgroundable(
            call, tool, EchoInput(text="x"), ctx=ctx, exec_id="t"
        )

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
        injected = await mgr2.resume_durable(ctx=ctx, exec_id="t")

        joined = "\n".join(str(m) for m in t2.messages)
        assert "Resumed from a checkpoint" in joined  # the framing line
        assert "interrupted" in joined
        assert "slow" in joined

        # resume_durable RETURNS exactly the messages it injected, so the caller
        # (LLMAgent._process_stream) can stream them — no transcript message
        # stays hidden from the event stream.
        assert injected
        assert t2.messages[-len(injected):] == injected

        # The record stays PENDING until a checkpoint persists the notice —
        # a crash before that must re-surface it on the next resume.
        recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
        assert all(r.status == TaskStatus.PENDING for r in recs)

        # flush_delivered (called after the agent checkpoint) makes the
        # record terminal, so a second resume surfaces nothing.
        await mgr2.flush_delivered(ctx=ctx)
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
        _note, event = await mgr.run_backgroundable(
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

    @pytest.mark.asyncio
    async def test_respawn_reruns_from_args_when_no_checkpoint(self):
        """
        A sub-agent interrupted *before* its first checkpoint is re-run from the
        spawning call's persisted arguments — not lost to "no session to resume".

        This is the failure mode that left re-spawned researchers returning null
        / raw inner-tool output: a child cancelled before it checkpointed has
        nothing to resume, so resume must replay it from ``tool_call_arguments``.
        """
        from datetime import UTC, datetime

        from grasp_agents.agent.background_tasks import BackgroundTaskManager
        from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
        from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
        from grasp_agents.durability.task_record import TaskRecord, TaskStatus
        from grasp_agents.tools.agent_tool import AgentTool

        store = InMemoryCheckpointStore()
        ctx = RunContext[None](state=None, checkpoint_store=store, session_key="s1")

        # A resumable sub-agent whose child answers in one LLM turn (no tools).
        child_llm = MockLLM(
            model_name="mock",
            responses_queue=[_text_response("ocean is deep and blue")],
        )
        researcher = AgentTool[None](
            name="researcher",
            description="Researches a topic",
            llm=child_llm,
            sys_prompt="Answer in one sentence.",
            auto_background_at=0,
        )

        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="coordinator",
            transcript=transcript,
            tools={"researcher": researcher},
            path=[],
        )

        # A PENDING record with the original args but NO child checkpoint on
        # disk — exactly the state a crash leaves before the child's first
        # AFTER_INPUT checkpoint.
        task_key = mgr._task_store_key(ctx, "c1")  # pyright: ignore[reportPrivateUsage]
        assert task_key is not None
        await store.save(
            task_key,
            TaskRecord(
                session_key="s1",
                task_id="bg_1",
                tool_call_id="c1",
                tool_name="researcher",
                tool_call_arguments='{"prompt": "research the ocean"}',
                status=TaskStatus.PENDING,
                started_at=datetime.now(UTC),
            )
            .model_dump_json()
            .encode(),
        )

        await mgr.resume_durable(ctx=ctx, exec_id="t")
        # The child was re-spawned; drive it to completion.
        await mgr.get("bg_1").task

        rec = TaskRecord.model_validate_json(await store.load(task_key))
        assert rec.status == TaskStatus.COMPLETED, rec.status
        # The re-run produced the child's actual final answer, not null.
        assert rec.result == "ocean is deep and blue", rec.result
        # ...and it got there by actually re-running the child (its LLM was
        # called), not by some empty-stream shortcut.
        assert child_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_resume_reserves_task_ids_to_avoid_collision(self):
        """
        After resume, a NEW tool call must not be assigned a ``bg_N`` that
        collides with a re-spawned task's id.

        A re-spawned task keeps its original id while the resumed manager's
        counter restarts at zero, so without reservation a new call would reuse
        ``bg_1``/``bg_2`` — overwriting the re-spawned task in ``_tasks`` and
        crossing their result deliveries (the re-spawned task's real result is
        persisted but never delivered; the new call's null/raw result is).
        """
        from datetime import UTC, datetime

        from grasp_agents.agent.background_tasks import BackgroundTaskManager
        from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
        from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
        from grasp_agents.durability.task_record import TaskRecord, TaskStatus
        from grasp_agents.tools.agent_tool import AgentTool

        store = InMemoryCheckpointStore()
        ctx = RunContext[None](state=None, checkpoint_store=store, session_key="s1")

        researcher = AgentTool[None](
            name="researcher",
            description="Researches a topic",
            llm=MockLLM(model_name="mock", responses_queue=[_text_response("done")]),
            sys_prompt="Answer.",
            auto_background_at=0,
        )
        transcript = LLMAgentTranscript()
        transcript.reset(instructions="sys")
        mgr = BackgroundTaskManager[None](
            agent_name="coordinator",
            transcript=transcript,
            tools={"researcher": researcher},
            path=[],
        )

        # Two interrupted tasks with the original run's ids bg_1, bg_2.
        for tid, cid in [("bg_1", "c1"), ("bg_2", "c2")]:
            key = mgr._task_store_key(ctx, cid)  # pyright: ignore[reportPrivateUsage]
            assert key is not None
            await store.save(
                key,
                TaskRecord(
                    session_key="s1",
                    task_id=tid,
                    tool_call_id=cid,
                    tool_name="researcher",
                    tool_call_arguments='{"prompt": "x"}',
                    status=TaskStatus.PENDING,
                    started_at=datetime.now(UTC),
                )
                .model_dump_json()
                .encode(),
            )

        await mgr.resume_durable(ctx=ctx, exec_id="t")

        # A new call this run gets a fresh id past every re-spawned one.
        assert mgr._next_id() == "bg_3"  # pyright: ignore[reportPrivateUsage]

        await mgr.cancel_all(ctx=ctx)


class _StreamingTool(BaseTool[EchoInput, Any, Any]):
    """Yields a ``ToolStreamEvent`` (optionally already stamped) then a result."""

    def __init__(self, *, preset_dest: str | None = None) -> None:
        super().__init__(name="streamer", description="Streams output")
        self._preset_dest = preset_dest

    async def _run(self, inp, *, ctx=None, **kwargs):
        del inp, ctx, kwargs
        return "done"

    async def _run_stream(self, inp, *, exec_id=None, **kwargs):
        del inp, kwargs
        from grasp_agents.types.events import ToolOutputEvent, ToolStreamEvent

        yield ToolStreamEvent(
            data="chunk",
            source="streamer",
            exec_id=exec_id,
            destination=self._preset_dest,
        )
        yield ToolOutputEvent(data="done", source="streamer", exec_id=exec_id)


class TestToolStreamDestinationStamping:
    """``BaseTool.run_stream`` stamps the owning agent so the UI can route."""

    @pytest.mark.asyncio
    async def test_run_stream_stamps_owning_agent(self):
        from grasp_agents.types.events import ToolStreamEvent

        tool = _StreamingTool()
        executor, _, _ = _make_executor([], tools=[tool])  # AgentLoop name "test"
        events = [
            e
            async for e in tool.run_stream(
                EchoInput(text="x"), agent_ctx=executor.agent_ctx
            )
        ]
        streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
        assert streamed
        assert all(e.destination == "test" for e in streamed)

    @pytest.mark.asyncio
    async def test_run_stream_preserves_inner_destination(self):
        # A nested sub-agent's already-stamped event must NOT be re-stamped by an
        # outer wrapper running with a different agent_ctx.
        from grasp_agents.types.events import ToolStreamEvent

        tool = _StreamingTool(preset_dest="inner_agent")
        executor, _, _ = _make_executor([], tools=[tool])
        events = [
            e
            async for e in tool.run_stream(
                EchoInput(text="x"), agent_ctx=executor.agent_ctx
            )
        ]
        streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
        assert streamed
        assert all(e.destination == "inner_agent" for e in streamed)


class TestFrontTrim:
    """
    Drain front-trims a surviving task's buffer: events consumed by both the
    bubble and flush cursors are dropped, bounding memory for a chatty command,
    while a terminal result event is preserved for ``_result_of``.
    """

    @pytest.mark.asyncio
    async def test_drain_trims_running_task_buffer(self):
        from grasp_agents.agent.background_tasks import PendingTask
        from grasp_agents.types.events import ToolStreamEvent

        executor, _, _ = _make_executor([])
        mgr = executor.bg_tasks
        ctx = executor.ctx

        async def _never() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(_never())
        pt = PendingTask(
            task_id="bg_1",
            tool_name="x",
            exec_id="t",
            task=task,
            blocks_final_answer=False,
        )
        for i in range(10):
            pt.events.append(ToolStreamEvent(data=f"s{i}", source="x"))
        mgr._tasks["bg_1"] = pt  # pyright: ignore[reportPrivateUsage]

        # drain bubbles + flushes the 10 events (cursor → 10) in one pass then
        # trims the consumed prefix → buffer emptied, cursor reset.
        _ = [e async for e in mgr.drain(exec_id="t", ctx=ctx)]
        assert pt.events == []
        assert pt.cursor == 0

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_trim_stops_before_a_terminal_result(self):
        from grasp_agents.agent.background_tasks import (
            BackgroundTaskManager,
            PendingTask,
        )
        from grasp_agents.types.events import ToolOutputEvent, ToolStreamEvent

        async def _noop() -> None: ...

        task = asyncio.create_task(_noop())
        await task

        pt = PendingTask(task_id="bg_1", tool_name="x", exec_id="t", task=task)
        pt.events.append(ToolStreamEvent(data="s0", source="x"))
        pt.events.append(ToolOutputEvent(data="RESULT", source="x"))  # terminal
        pt.events.append(ToolStreamEvent(data="s2", source="x"))
        pt.cursor = 3

        BackgroundTaskManager._trim_consumed(pt)  # pyright: ignore[reportPrivateUsage]

        # Trim stops before the terminal result at index 1 → only ``s0`` dropped,
        # so a later ``_result_of`` still finds the result.
        assert len(pt.events) == 2
        assert isinstance(pt.events[0], ToolOutputEvent)
        assert pt.cursor == 2

    @pytest.mark.asyncio
    async def test_run_stream_without_agent_ctx_leaves_unset(self):
        from grasp_agents.types.events import ToolStreamEvent

        tool = _StreamingTool()
        events = [e async for e in tool.run_stream(EchoInput(text="x"))]
        streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
        assert streamed
        assert all(e.destination is None for e in streamed)
