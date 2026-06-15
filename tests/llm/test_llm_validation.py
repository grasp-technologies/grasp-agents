"""
Tests for LLM validation and retry logic.

Focuses on:
- Validation errors carry the right type and context (tool name, schema)
- Retry loop only retries validation errors, API errors propagate immediately
- Stream retry emits ResponseRetrying with correct sequence_number
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel

from grasp_agents.llm.llm import LLM
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageRefusal, OutputMessageText
from grasp_agents.types.errors import (
    LLMResponseRefusalError,
    LLMResponseValidationError,
    LLMToolCallValidationError,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_errors import LlmInternalServerError
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseRetrying,
)
from grasp_agents.types.response import Response

# ---------- Mocks ----------


def _text_response(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
    )


def _tool_call_response(name: str, arguments: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(call_id="tc_1", name=name, arguments=arguments)
        ],
    )


def _refusal_response(text: str = "I can't help with that.") -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageRefusal(refusal=text)],
                status="completed",
            )
        ],
    )


def _content_filter_response() -> Response:
    return Response(
        model="mock",
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="content_filter"),
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text="")], status="incomplete"
            )
        ],
    )


class AddInput(BaseModel):
    a: int
    b: int


class AddTool(BaseTool[AddInput, Any, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="add", description="Add two numbers", **kwargs)

    async def _run(self, inp: AddInput, **kwargs: Any) -> int:
        return inp.a + inp.b


@dataclass(frozen=True)
class MockLLM(LLM):
    """Returns pre-configured responses from a queue."""

    responses: list[Response] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses, "MockLLM: no more responses"
        return self.responses.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input, tools=tools, output_schema=output_schema, tool_choice=tool_choice
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


@dataclass(frozen=True)
class ErrorLLM(LLM):
    """Always raises a given error — used to test propagation behavior."""

    error_to_raise: Exception = field(default_factory=lambda: RuntimeError("boom"))

    def __post_init__(self):
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        raise self.error_to_raise

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        raise self.error_to_raise
        yield


_USER_MSG = [InputMessageItem.from_text("go", role="user")]


# ---------- Validation error types ----------


class TestValidationErrorTypes:
    """Validation wrapping: errors carry the right type and context."""

    @pytest.mark.asyncio
    async def test_bad_tool_args_raise_tool_call_error_with_name(self):
        """Bad tool arguments → LLMToolCallValidationError mentioning tool name."""
        llm = MockLLM(
            model_name="mock",
            responses=[_tool_call_response("add", '{"a": "nope", "b": 1}')],
        )
        with pytest.raises(LLMToolCallValidationError, match="add"):
            await llm.generate_response(_USER_MSG, tools={"add": AddTool()})

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_tool_call_error(self):
        """Tool call for a tool not in the tools dict → LLMToolCallValidationError."""
        llm = MockLLM(
            model_name="mock",
            responses=[_tool_call_response("nonexistent", "{}")],
        )
        with pytest.raises(LLMToolCallValidationError, match="not available"):
            await llm.generate_response(_USER_MSG, tools={"add": AddTool()})

    @pytest.mark.asyncio
    async def test_multiple_bad_calls_surface_all_errors(self):
        """Two bad tool calls → both errors carried, not just the first."""
        response = Response(
            model="mock",
            output_items=[
                FunctionToolCallItem(
                    call_id="tc_1", name="add", arguments='{"a": "nope", "b": 1}'
                ),
                FunctionToolCallItem(call_id="tc_2", name="ghost", arguments="{}"),
            ],
        )
        llm = MockLLM(model_name="mock", responses=[response])
        with pytest.raises(LLMToolCallValidationError) as ei:
            await llm.generate_response(_USER_MSG, tools={"add": AddTool()})
        err = ei.value
        assert {call_id for call_id, _, _ in err.failed_calls} == {"tc_1", "tc_2"}
        # Both failures are reflected in the human-readable message too.
        assert "add" in str(err)
        assert "not available" in str(err)

    @pytest.mark.asyncio
    async def test_output_schema_raises_response_validation_error(self):
        """Bad response text + output_schema → LLMResponseValidationError."""
        llm = MockLLM(
            model_name="mock",
            responses=[_text_response("garbage")],
        )

        class StrictModel(BaseModel):
            value: int

        with pytest.raises(LLMResponseValidationError):
            await llm.generate_response(_USER_MSG, output_schema=StrictModel)


class TestRefusalAndContentFilter:
    """Refusals / content filters raise a dedicated, non-retryable error."""

    @pytest.mark.asyncio
    async def test_refusal_part_raises_refusal_error(self):
        llm = MockLLM(model_name="mock", responses=[_refusal_response()])
        with pytest.raises(LLMResponseRefusalError):
            await llm.generate_response(_USER_MSG)

    @pytest.mark.asyncio
    async def test_content_filter_raises_refusal_error(self):
        llm = MockLLM(model_name="mock", responses=[_content_filter_response()])
        with pytest.raises(LLMResponseRefusalError) as ei:
            await llm.generate_response(_USER_MSG)
        assert ei.value.reason == "content_filter"

    @pytest.mark.asyncio
    async def test_refusal_takes_precedence_over_schema(self):
        """A refusal raises the refusal error, not a schema-validation error."""
        llm = MockLLM(model_name="mock", responses=[_refusal_response()])

        class M(BaseModel):
            v: int

        with pytest.raises(LLMResponseRefusalError):
            await llm.generate_response(_USER_MSG, output_schema=M)

    @pytest.mark.asyncio
    async def test_refusal_is_not_retried(self):
        """Even with validation retries available, a refusal is terminal."""
        llm = MockLLM(
            model_name="mock",
            responses=[_refusal_response(), _text_response('{"v": 1}')],
            retry_policy=RetryPolicy(validation_retries=3),
        )
        with pytest.raises(LLMResponseRefusalError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_refusal_raised_from_stream_path(self):
        llm = MockLLM(model_name="mock", responses=[_refusal_response()])
        with pytest.raises(LLMResponseRefusalError):
            async for _ in llm.generate_response_stream(_USER_MSG):
                pass


# ---------- Retry behavior ----------


class TestRetryBehavior:
    """Retry loop: only validation errors are retried, API errors propagate."""

    @pytest.mark.asyncio
    async def test_api_error_not_retried_by_validation_loop(self):
        """LlmInternalServerError propagates immediately; validation won't catch it."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmInternalServerError(
                "server down",
                response=httpx.Response(
                    500, request=httpx.Request("POST", "https://test")
                ),
                body=None,
            ),
            retry_policy=RetryPolicy(api_retries=0, validation_retries=3),
        )
        with pytest.raises(LlmInternalServerError):
            await llm.generate_response(_USER_MSG)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_api_error_not_retried_by_validation_loop_stream(self):
        """LlmInternalServerError propagates from stream; validation won't catch it."""
        llm = ErrorLLM(
            model_name="mock",
            error_to_raise=LlmInternalServerError(
                "server down",
                response=httpx.Response(
                    500, request=httpx.Request("POST", "https://test")
                ),
                body=None,
            ),
            retry_policy=RetryPolicy(api_retries=0, validation_retries=3),
        )
        with pytest.raises(LlmInternalServerError):
            async for _ in llm.generate_response_stream(_USER_MSG):
                pass
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_validation_retry_then_success(self):
        """Bad response retried, second attempt succeeds."""
        llm = MockLLM(
            model_name="mock",
            responses=[_text_response("bad"), _text_response('{"v": 1}')],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        class M(BaseModel):
            v: int

        result = await llm.generate_response(_USER_MSG, output_schema=M)
        assert result.output_text == '{"v": 1}'
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self):
        """All retries fail → raises the validation error, not RuntimeError."""
        llm = MockLLM(
            model_name="mock",
            responses=[_text_response("bad1"), _text_response("bad2")],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        class M(BaseModel):
            v: int

        with pytest.raises(LLMResponseValidationError):
            await llm.generate_response(_USER_MSG, output_schema=M)
        assert llm.call_count == 2


# ---------- Stream retry events ----------


class TestStreamRetry:
    """Stream retry emits ResponseRetrying with correct state."""

    @pytest.mark.asyncio
    async def test_stream_retry_emits_response_retrying(self):
        """Failed validation mid-stream → ResponseRetrying → success."""
        llm = MockLLM(
            model_name="mock",
            responses=[_text_response("bad"), _text_response('{"v": 1}')],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        class M(BaseModel):
            v: int

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(_USER_MSG, output_schema=M):
            events.append(event)

        retrying = [e for e in events if isinstance(e, ResponseRetrying)]
        assert len(retrying) == 1
        assert retrying[0].attempt == 1

        # Stream eventually succeeds
        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 2  # one failed attempt + one success

    @pytest.mark.asyncio
    async def test_response_retrying_sequence_number(self):
        """ResponseRetrying.sequence_number = last streamed event's seq + 1."""
        llm = MockLLM(
            model_name="mock",
            responses=[_text_response("bad"), _text_response('{"v": 1}')],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        class M(BaseModel):
            v: int

        events: list[LlmEvent] = []
        async for event in llm.generate_response_stream(_USER_MSG, output_schema=M):
            events.append(event)

        retrying_idx = next(
            i for i, e in enumerate(events) if isinstance(e, ResponseRetrying)
        )
        last_before = events[retrying_idx - 1]
        assert isinstance(last_before, ResponseCompleted)
        assert events[retrying_idx].sequence_number == last_before.sequence_number + 1
