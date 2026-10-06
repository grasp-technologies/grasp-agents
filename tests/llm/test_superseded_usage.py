"""
Validation retries: the terminal event is withheld for a response that fails
validation, the usage of replaced attempts travels on the final response, and
an abandoned attempt's stream is closed before the next one starts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.errors import LLMToolCallValidationError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    InputItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted
from grasp_agents.types.response import Response, ResponseUsage
from tests._helpers import AddTool
from tests.llm.test_llm_validation import _USER_MSG, MockLLM


class _Schema(BaseModel):
    v: int


def _usage(input_tokens: int, output_tokens: int) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _text(text: str, usage: ResponseUsage) -> Response:
    return Response(
        model="mock",
        output=[
            OutputMessageItem(
                content=[OutputMessageText(text=text)], status="completed"
            )
        ],
        usage=usage,
    )


def _bad_call(usage: ResponseUsage) -> Response:
    return Response(
        model="mock",
        output=[FunctionToolCallItem(call_id="c1", name="nope", arguments="{}")],
        usage=usage,
    )


class TestSupersededUsage:
    @pytest.mark.asyncio
    async def test_non_streaming_final_response_carries_replaced_usage(self) -> None:
        llm = MockLLM(
            model_name="mock",
            responses=[
                _text("bad", _usage(10, 5)),
                _text("worse", _usage(11, 6)),
                _text('{"v": 1}', _usage(20, 7)),
            ],
            retry_policy=RetryPolicy(validation_retries=2),
        )

        response = await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert response.usage == _usage(20, 7)
        assert response.superseded_usage == _usage(21, 11)

    @pytest.mark.asyncio
    async def test_streaming_final_response_carries_replaced_usage(self) -> None:
        llm = MockLLM(
            model_name="mock",
            responses=[_text("bad", _usage(10, 5)), _text('{"v": 1}', _usage(20, 7))],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        events = [
            e
            async for e in llm.generate_response_stream(
                _USER_MSG, output_schema=_Schema
            )
        ]

        (completed,) = [e for e in events if isinstance(e, ResponseCompleted)]
        assert completed.response.usage == _usage(20, 7)
        assert completed.response.superseded_usage == _usage(10, 5)

    @pytest.mark.asyncio
    async def test_exhausted_retries_report_replaced_usage_on_the_error(self) -> None:
        llm = MockLLM(
            model_name="mock",
            responses=[_bad_call(_usage(10, 5)), _bad_call(_usage(12, 6))],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        with pytest.raises(LLMToolCallValidationError) as info:
            await llm.generate_response(_USER_MSG, tools={"add": AddTool()})

        assert info.value.response is not None
        assert info.value.response.usage == _usage(12, 6)
        assert info.value.response.superseded_usage == _usage(10, 5)

    @pytest.mark.asyncio
    async def test_first_attempt_has_no_replaced_usage(self) -> None:
        llm = MockLLM(
            model_name="mock",
            responses=[_text('{"v": 1}', _usage(20, 7))],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        response = await llm.generate_response(_USER_MSG, output_schema=_Schema)

        assert response.superseded_usage is None


@dataclass(frozen=True)
class _TrackingLLM(MockLLM):
    log: list[str] = field(default_factory=list)

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> AsyncIterator[LlmEvent]:
        self.log.append("started")
        try:
            async for event in super()._generate_response_stream_once(
                input,
                tools=tools,
                output_schema=output_schema,
                tool_choice=tool_choice,
                **extra,
            ):
                yield event
        finally:
            self.log.append("closed")


class TestAbandonedAttemptIsClosed:
    @pytest.mark.asyncio
    async def test_failed_attempt_stream_closes_before_the_retry_starts(self) -> None:
        llm = _TrackingLLM(
            model_name="mock",
            responses=[_text("bad", _usage(10, 5)), _text('{"v": 1}', _usage(20, 7))],
            retry_policy=RetryPolicy(validation_retries=1),
        )

        async for _ in llm.generate_response_stream(_USER_MSG, output_schema=_Schema):
            pass

        assert llm.log == ["started", "closed", "started", "closed"]
