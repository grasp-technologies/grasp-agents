"""
Integration tests for LLMPolicyExecutor with the new Response/StreamEvent pipeline.

Tests the full flow: memory → LLM → response → memory update → tool calling → repeat.
Uses a mock LLM to avoid real API calls.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

from grasp_agents.llm import LLM
from grasp_agents.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm_policy_executor import LLMPolicyExecutor
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import OutputTextContentPart
from grasp_agents.types.events import (
    Event,
    LLMStreamEvent,
    ToolCallEvent,
    ToolMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseFailed,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

# ---------- Mock LLM ----------


@dataclass(frozen=True)
class MockLLM(LLM):
    """LLM that returns pre-configured responses."""

    responses_queue: list[Response] | None = None

    def __post_init__(self):
        if self.responses_queue is None:
            object.__setattr__(self, "responses_queue", [])
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses_queue, "MockLLM: no more responses in queue"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
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


# ---------- Mock Tool ----------


class AddInput(BaseModel):
    a: int
    b: int


class AddTool(BaseTool[AddInput, Any, Any]):
    name: str = "add"
    description: str = "Add two numbers"
    in_type: type[AddInput] = AddInput

    async def run(
        self, inp: AddInput, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> int:
        return inp.a + inp.b


# ---------- Helpers ----------


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _make_text_response(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputTextContentPart(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=_make_usage(),
    )


def _make_tool_call_response(
    name: str,
    arguments: str,
    call_id: str,
    text: str = "",
) -> Response:
    output = []
    if text:
        output.append(
            OutputMessageItem(
                content_parts=[OutputTextContentPart(text=text)],
                status="completed",
            )
        )
    output.append(FunctionToolCallItem(call_id=call_id, name=name, arguments=arguments))
    return Response(
        model="mock",
        output_items=output,
        usage_with_cost=_make_usage(),
    )


def _with_final_answer_checker(
    executor: LLMPolicyExecutor,
) -> LLMPolicyExecutor:
    """
    Add a check_for_final_answer_impl that stops on text-only responses.

    The default executor never returns a final answer from check_for_final_answer
    (it's designed to be overridden by LLMAgent hooks). This helper mimics the
    standard LLMAgent behavior: if the last response has no tool calls, treat
    the output text as the final answer.
    """

    def _check(*, ctx, call_id, response=None, **kwargs):  # noqa: ARG001
        if response and not response.tool_call_items:
            return response.output_text or None
        return None

    executor.check_for_final_answer_impl = _check  # type: ignore[assignment]
    return executor


async def _collect_events(
    executor, ctx, call_id="test"
) -> tuple[list[Event[Any]], Response | None]:
    events: list[Event[Any]] = []
    response: Response | None = None
    extra: dict[str, Any] = {}
    async for event in executor.execute_stream(
        ctx=ctx, call_id=call_id, extra_llm_settings=extra
    ):
        events.append(event)
        if isinstance(event, LLMStreamEvent) and isinstance(
            event.data, ResponseCompleted
        ):
            response = event.data.response
    return events, response


# ---------- Tests ----------


class TestExecutorTextResponse:
    """Test executor with a simple text response (no tool calls)."""

    @pytest.mark.asyncio
    async def test_single_turn_text_response(self):
        """Single LLM call returns text → final answer."""
        response = _make_text_response("The answer is 42.")
        llm = MockLLM(model_name="mock", responses_queue=[response])
        memory = LLMAgentMemory()
        memory.reset(instructions="Be helpful.")

        executor = LLMPolicyExecutor(
            agent_name="test_agent",
            llm=llm,
            memory=memory,
            tools=None,
            max_turns=3,
            stream_llm_responses=False,
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("What is 42?", role="user")])

        events, last_response = await _collect_events(executor, ctx)

        # Should have LLMStreamEvent events (synthesized)
        llm_events = [e for e in events if isinstance(e, LLMStreamEvent)]
        assert len(llm_events) > 0

        # Response should be in context
        assert len(ctx.responses["test_agent"]) == 1

        # Memory should contain: system + user + output items
        assert len(memory.messages) >= 3

        # Final answer should be the text
        assert last_response is not None
        assert executor.get_final_answer(last_response) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_streaming_mode(self):
        """Streaming mode also produces LLMStreamEvents."""
        response = _make_text_response("Hello!")
        llm = MockLLM(model_name="mock", responses_queue=[response])
        memory = LLMAgentMemory()
        memory.reset(instructions="sys")

        executor = LLMPolicyExecutor(
            agent_name="agent",
            llm=llm,
            memory=memory,
            tools=None,
            max_turns=1,
            stream_llm_responses=True,
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("Hi", role="user")])
        events, last_response = await _collect_events(executor, ctx)

        llm_events = [e for e in events if isinstance(e, LLMStreamEvent)]
        assert len(llm_events) > 0
        assert last_response is not None
        assert executor.get_final_answer(last_response) == "Hello!"


class TestExecutorToolCalling:
    """Test executor with tool calls and responses."""

    @pytest.mark.asyncio
    async def test_tool_call_and_final_answer(self):
        """LLM calls a tool, gets result, then provides final answer."""
        tool_response = _make_tool_call_response(
            name="add",
            arguments='{"a": 2, "b": 3}',
            call_id="tc_1",
            text="Let me add those.",
        )
        final_response = _make_text_response("2 + 3 = 5")

        llm = MockLLM(
            model_name="mock",
            responses_queue=[tool_response, final_response],
        )
        memory = LLMAgentMemory()
        memory.reset(instructions="You can add numbers.")

        tool = AddTool()
        executor = _with_final_answer_checker(
            LLMPolicyExecutor(
                agent_name="calc",
                llm=llm,
                memory=memory,
                tools=[tool],
                max_turns=3,
                stream_llm_responses=False,
            )
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("Add 2 and 3", role="user")])

        events, last_response = await _collect_events(executor, ctx)

        # Should have tool call events
        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].data.name == "add"

        # Should have tool message events
        tool_msg_events = [e for e in events if isinstance(e, ToolMessageEvent)]
        assert len(tool_msg_events) == 1

        # LLM called twice (tool call + final answer)
        assert llm.call_count == 2

        # Memory should contain the full conversation
        # system, user, output+tool_call, tool_output, final_output
        assert len(memory.messages) >= 5

        # Verify tool output is in memory
        tool_outputs = [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0].call_id == "tc_1"

        # Final answer
        assert last_response is not None
        assert executor.get_final_answer(last_response) == "2 + 3 = 5"

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """
        Executor stops after max_turns even if LLM keeps calling tools.

        With max_turns=2, the flow is:
          1. First generate (outside loop): tool_call → call 1
          2. Loop iter 1: check→None, turns=0<2 → call_tools, generate → call 2 (tool_call), turns=1
          3. Loop iter 2: check→None, turns=1<2 → call_tools, generate → call 3 (tool_call), turns=2
          4. Loop iter 3: check→None, turns=2>=2 → force_generate → call 4 (text)
        Total: 4 LLM calls (3 tool calls + 1 forced text answer).
        """
        responses: list[Response] = [
            _make_tool_call_response(
                name="add", arguments='{"a": 1, "b": 1}', call_id=f"tc_{i}"
            )
            for i in range(3)
        ]
        responses.append(_make_text_response("Forced answer"))

        llm = MockLLM(model_name="mock", responses_queue=responses)
        memory = LLMAgentMemory()
        memory.reset(instructions="sys")

        tool = AddTool()
        executor = LLMPolicyExecutor(
            agent_name="agent",
            llm=llm,
            memory=memory,
            tools=[tool],
            max_turns=2,
            stream_llm_responses=False,
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("loop", role="user")])
        events, last_response = await _collect_events(executor, ctx)

        # 4 LLM calls: first generate + 2 loop generates + force_generate
        assert llm.call_count == 4

        # Only 2 tool rounds actually executed: the 3rd tool_call response
        # is generated but its tools are never called (force_generate fires first)
        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_call_events) == 2

        # Final answer was forced
        assert last_response is not None
        assert executor.get_final_answer(last_response) == "Forced answer"


class TestExecutorUsageTracking:
    """Test that usage tracking works through the executor."""

    @pytest.mark.asyncio
    async def test_usage_tracked_in_context(self):
        """Response usage is tracked in RunContext.usage_tracker."""
        response = _make_text_response("answer")
        llm = MockLLM(model_name="mock", responses_queue=[response])
        memory = LLMAgentMemory()
        memory.reset(instructions="sys")

        executor = LLMPolicyExecutor(
            agent_name="test_agent",
            llm=llm,
            memory=memory,
            tools=None,
            max_turns=1,
            stream_llm_responses=False,
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("q", role="user")])
        await _collect_events(executor, ctx)

        # Usage should be tracked
        assert "test_agent" in ctx.usage_tracker.usages
        usage = ctx.usage_tracker.usages["test_agent"]
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


class TestExecutorMemoryIntegrity:
    """Test that memory stores the correct item types after execution."""

    @pytest.mark.asyncio
    async def test_memory_item_types_after_tool_use(self):
        """Memory contains correct Responses API item types after agentic loop."""
        tool_response = _make_tool_call_response(
            name="add",
            arguments='{"a": 1, "b": 2}',
            call_id="tc_1",
            text="Computing...",
        )
        final_response = _make_text_response("3")

        llm = MockLLM(
            model_name="mock",
            responses_queue=[tool_response, final_response],
        )
        memory = LLMAgentMemory()
        memory.reset(instructions="calc")

        executor = _with_final_answer_checker(
            LLMPolicyExecutor(
                agent_name="agent",
                llm=llm,
                memory=memory,
                tools=[AddTool()],
                max_turns=3,
                stream_llm_responses=False,
            )
        )

        ctx = RunContext[None]()
        memory.update([InputMessageItem.from_text("1+2", role="user")])
        await _collect_events(executor, ctx)

        # Verify each message type
        types = [type(m).__name__ for m in memory.messages]
        assert types[0] == "InputMessageItem"  # system
        assert types[1] == "InputMessageItem"  # user

        # After first LLM call: OutputMessageItem + FunctionToolCallItem
        assert "OutputMessageItem" in types
        assert "FunctionToolCallItem" in types

        # After tool execution: FunctionToolOutputItem
        assert "FunctionToolOutputItem" in types

        # After second LLM call: OutputMessageItem (final answer)
        # Last message should be the final answer
        assert isinstance(memory.messages[-1], OutputMessageItem)


# ---------- LLM retry + validation tests ----------


@dataclass(frozen=True)
class FailThenSucceedLLM(LLM):
    """LLM that fails N times then succeeds. Used to test retry logic."""

    fail_count: int = 1
    success_response: Response | None = None

    def __post_init__(self):
        object.__setattr__(self, "_attempts", 0)

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        attempts = self._attempts  # type: ignore[attr-defined]
        object.__setattr__(self, "_attempts", attempts + 1)
        if attempts < self.fail_count:
            raise RuntimeError(f"Simulated failure #{attempts + 1}")
        assert self.success_response is not None
        return self.success_response

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        yield ResponseCreated(response=response, sequence_number=1)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=idx + 2)
            yield OutputItemDone(item=item, output_index=idx, sequence_number=idx + 3)
        yield ResponseCompleted(response=response, sequence_number=100)  # type: ignore[arg-type]


class TestStreamRetryEvents:
    """Test that ResponseFailed is emitted when a stream retries."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="ResponseFailed emission is commented out in llm.py")
    async def test_retry_emits_response_failed(self):
        """When stream fails and retries, ResponseFailed is yielded before retry."""
        success = _make_text_response("Success after retry")
        llm = FailThenSucceedLLM(
            model_name="mock",
            fail_count=1,
            success_response=success,
            max_response_retries=1,
        )

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(
            [InputMessageItem.from_text("test", role="user")]
        ):
            events.append(event)

        # Should have a ResponseFailed followed by the successful stream
        failed_events = [e for e in events if isinstance(e, ResponseFailed)]
        assert len(failed_events) == 1
        assert failed_events[0].response.status == "failed"
        assert failed_events[0].response.error is not None
        assert "Simulated failure" in failed_events[0].response.error.message

        # Should also have a successful ResponseCompleted
        completed_events = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed_events) == 1

    @pytest.mark.asyncio
    async def test_no_retry_event_when_no_retries_configured(self):
        """Without retries, the error just raises (no ResponseFailed emitted)."""
        success = _make_text_response("Never reached")
        llm = FailThenSucceedLLM(
            model_name="mock",
            fail_count=1,
            success_response=success,
            max_response_retries=0,
        )

        with pytest.raises(RuntimeError, match="Simulated failure"):
            async for _ in llm.generate_response_stream(
                [InputMessageItem.from_text("test", role="user")]
            ):
                pass

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="ResponseFailed emission is commented out in llm.py")
    async def test_multiple_retries_emit_multiple_failures(self):
        """Multiple retries emit multiple ResponseFailed events."""
        success = _make_text_response("Finally!")
        llm = FailThenSucceedLLM(
            model_name="mock",
            fail_count=3,
            success_response=success,
            max_response_retries=3,
        )

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(
            [InputMessageItem.from_text("test", role="user")]
        ):
            events.append(event)

        failed_events = [e for e in events if isinstance(e, ResponseFailed)]
        assert len(failed_events) == 3
        for i, fe in enumerate(failed_events):
            assert f"#{i + 1}" in fe.response.error.message


class TestResponseSchemaValidation:
    """Test that response_schema validation works in generate_response."""

    @pytest.mark.asyncio
    async def test_schema_validation_passes(self):
        """Valid JSON matching the schema passes validation."""
        response = _make_text_response('{"name": "Alice", "age": 30}')
        llm = MockLLM(
            model_name="mock",
            responses_queue=[response],
        )

        class Person(BaseModel):
            name: str
            age: int

        result = await llm.generate_response(
            [InputMessageItem.from_text("test", role="user")],
            response_schema=Person,
        )
        assert result.output_text == '{"name": "Alice", "age": 30}'

    @pytest.mark.asyncio
    async def test_schema_validation_fails_and_retries(self):
        """Invalid JSON triggers retry when max_response_retries > 0."""
        bad_response = _make_text_response("not valid json")
        good_response = _make_text_response('{"name": "Bob", "age": 25}')
        llm = MockLLM(
            model_name="mock",
            responses_queue=[bad_response, good_response],
            max_response_retries=1,
        )

        class Person(BaseModel):
            name: str
            age: int

        result = await llm.generate_response(
            [InputMessageItem.from_text("test", role="user")],
            response_schema=Person,
        )
        assert result.output_text == '{"name": "Bob", "age": 25}'
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_schema_validation_skipped_when_tool_calls(self):
        """Schema validation is skipped when response contains tool calls."""
        response = Response(
            model="mock",
            output_items=[
                FunctionToolCallItem(
                    call_id="c1", name="search", arguments='{"q": "test"}'
                )
            ],
            usage_with_cost=_make_usage(),
        )
        llm = MockLLM(
            model_name="mock",
            responses_queue=[response],
        )

        class SomeSchema(BaseModel):
            result: str

        # Should not raise even though there's no text matching the schema
        result = await llm.generate_response(
            [InputMessageItem.from_text("test", role="user")],
            response_schema=SomeSchema,
        )
        assert len(result.tool_call_items) == 1

    @pytest.mark.asyncio
    async def test_schema_validation_raises_without_retries(self):
        """Schema validation failure raises when max_response_retries=0."""
        from grasp_agents.errors import JSONSchemaValidationError

        bad_response = _make_text_response("totally invalid")
        llm = MockLLM(
            model_name="mock",
            responses_queue=[bad_response],
            max_response_retries=0,
        )

        class Strict(BaseModel):
            value: int

        with pytest.raises((JSONSchemaValidationError, Exception)):
            await llm.generate_response(
                [InputMessageItem.from_text("test", role="user")],
                response_schema=Strict,
            )
