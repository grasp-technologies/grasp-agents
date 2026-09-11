"""
LiteLLM integration tests that call real LLM APIs (OpenAI by default).

Skipped by default. Run with:
    uv run pytest -m integration -k litellm
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ResponseCompleted,
)

if TYPE_CHECKING:
    from grasp_agents.llm.cloud_llm import CloudLLM
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.response import Response


class Capital(BaseModel):
    country: str = Field(description="Country name")
    capital: str = Field(description="Capital city name")


@pytest.mark.integration
class TestLiteLLMIntegration:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM

        # LiteLLM reads OPENAI_API_KEY from env automatically
        return LiteLLM(
            model_name="gpt-5.4-nano",
            llm_settings={"max_completion_tokens": 100},
        )

    @pytest.mark.asyncio
    async def test_generate_text(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("Say 'hello' and nothing else.")]
        response = await llm.generate_response(input_items)

        assert response.status == "completed"
        assert len(response.output) >= 1
        assert isinstance(response.output[0], OutputMessageItem)
        assert "hello" in response.output_text.lower()

    @pytest.mark.asyncio
    async def test_stream_text(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("Say 'hello' and nothing else.")]
        events = [event async for event in llm.generate_response_stream(input_items)]
        text_deltas = [
            e for e in events if isinstance(e, OutputMessageTextPartTextDelta)
        ]
        completed = [e for e in events if isinstance(e, ResponseCompleted)]

        assert len(text_deltas) > 0
        assert len(completed) == 1
        assert "hello" in completed[0].response.output_text.lower()

    @pytest.mark.asyncio
    async def test_tool_roundtrip(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "What is 17 + 25? Use the add tool, then tell me the result."
        )
        response = await llm.generate_response(
            [user_msg], tools=tools, tool_choice="required"
        )
        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in response.tool_call_items
        ]

        full_input = [user_msg, *response.output, *tool_outputs]
        final_response = await llm.generate_response(full_input, tools=tools)

        assert "42" in final_response.output_text


@pytest.mark.integration
class TestLiteLLMStructuredOutput:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM

        return LiteLLM(
            model_name="gpt-5.4-nano",
            llm_settings={"max_completion_tokens": 200},
            apply_output_schema_via_provider=True,
        )

    @pytest.mark.asyncio
    async def test_structured_output(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("What is the capital of France?")]
        response = await llm.generate_response(input_items, output_schema=Capital)

        parsed = Capital.model_validate_json(response.output_text)
        assert parsed.capital.lower() == "paris"
        assert parsed.country.lower() == "france"

    @pytest.mark.asyncio
    async def test_stream_structured_output(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("What is the capital of France?")]
        events = [
            event
            async for event in llm.generate_response_stream(
                input_items, output_schema=Capital
            )
        ]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        parsed = Capital.model_validate_json(response.output_text)
        assert parsed.capital.lower() == "paris"
        assert parsed.country.lower() == "france"


def _execute_parallel_tools(
    tool_calls: list[FunctionToolCallItem],
) -> list[FunctionToolOutputItem]:
    tool_outputs: list[FunctionToolOutputItem] = []
    for tc in tool_calls:
        args = json.loads(tc.arguments)
        result = args["a"] + args["b"] if tc.name == "add" else args["a"] * args["b"]
        tool_outputs.append(
            FunctionToolOutputItem.from_tool_result(call_id=tc.call_id, output=result)
        )
    return tool_outputs


@pytest.mark.integration
class TestLiteLLMParallelToolUse:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM

        return LiteLLM(
            model_name="gpt-5.4-nano",
            llm_settings={"max_completion_tokens": 256},
        )

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(
        self,
        llm: CloudLLM,
        parallel_tools: dict[str, BaseTool[Any, Any, Any]],
    ) -> None:
        """Model should call add and multiply in parallel, then answer."""
        user_msg = InputMessageItem.from_text(
            "I need two results: (1) add 17 and 25, (2) multiply 6 and 7. "
            "Use both tools in parallel, then report both results."
        )
        r1 = await llm.generate_response(
            [user_msg], tools=parallel_tools, tool_choice="required"
        )

        assert len(r1.tool_call_items) == 2, (
            f"Expected 2 parallel tool calls, got {len(r1.tool_call_items)}"
        )
        tool_names = {tc.name for tc in r1.tool_call_items}
        assert tool_names == {"add", "multiply"}

        tool_outputs = _execute_parallel_tools(r1.tool_call_items)
        full_input = [user_msg, *r1.output, *tool_outputs]
        r2 = await llm.generate_response(full_input, tools=parallel_tools)

        assert r2.status == "completed"
        assert "42" in r2.output_text

    @pytest.mark.asyncio
    async def test_stream_parallel_tool_calls(
        self,
        llm: CloudLLM,
        parallel_tools: dict[str, BaseTool[Any, Any, Any]],
    ) -> None:
        """Streaming: parallel tool calls should round-trip correctly."""
        user_msg = InputMessageItem.from_text(
            "I need two results: (1) add 17 and 25, (2) multiply 6 and 7. "
            "Use both tools in parallel, then report both results."
        )
        events1 = [
            event
            async for event in llm.generate_response_stream(
                [user_msg],
                tools=parallel_tools,
                tool_choice="required",
            )
        ]
        completed1 = [e for e in events1 if isinstance(e, ResponseCompleted)]
        assert len(completed1) == 1
        r1 = completed1[0].response

        assert len(r1.tool_call_items) == 2
        tool_names = {tc.name for tc in r1.tool_call_items}
        assert tool_names == {"add", "multiply"}

        tool_outputs = _execute_parallel_tools(r1.tool_call_items)
        full_input = [user_msg, *r1.output, *tool_outputs]
        events2 = [
            event
            async for event in llm.generate_response_stream(
                full_input, tools=parallel_tools
            )
        ]
        completed2 = [e for e in events2 if isinstance(e, ResponseCompleted)]
        assert len(completed2) == 1
        r2 = completed2[0].response

        assert r2.status == "completed"
        assert "42" in r2.output_text


async def _stream_with_snapshots(
    llm: CloudLLM, input_items: list[Any], **kwargs: Any
) -> tuple[Response, list[OutputItem]]:
    """Stream a response; copy every item at the moment it is emitted."""
    emitted: list[OutputItem] = []
    response: Response | None = None
    async for event in llm.generate_response_stream(input_items, **kwargs):
        if isinstance(event, OutputItemDone):
            emitted.append(event.item.model_copy(deep=True))
        elif isinstance(event, ResponseCompleted):
            response = event.response
    assert response is not None
    return response, emitted


@pytest.mark.integration
class TestLiteLLMGeminiThoughtSignatures:
    """
    Gemini signs the first non-thought part of a turn and rejects a replay that
    lacks the signature. A signed item must carry its signature the moment it
    is streamed out, and the signature must reach the wire on the next turn.
    """

    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM

        return LiteLLM(
            model_name="gemini/gemini-3.1-flash-lite",
            api_provider=APIProvider(
                name="gemini", base_url=None, api_key=google_api_key
            ),
            llm_settings={"reasoning_effort": "low", "max_completion_tokens": 1024},
        )

    @pytest.mark.asyncio
    async def test_stream_tool_call_signature_round_trip(
        self, llm: CloudLLM, parallel_tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text("What is 17 + 25? Use the add tool.")
        r1, emitted = await _stream_with_snapshots(
            llm, [user_msg], tools=parallel_tools, tool_choice="required"
        )

        calls = [i for i in emitted if isinstance(i, FunctionToolCallItem)]
        assert calls
        assert calls[0].provider_specific_fields
        assert calls[0].provider_specific_fields["thought_signature"]
        assert all(
            i.encrypted_content is None for i in emitted if isinstance(i, ReasoningItem)
        )
        assert [i.model_dump() for i in emitted] == [i.model_dump() for i in r1.output]

        tool_outputs = _execute_parallel_tools(r1.tool_call_items)
        r2, _ = await _stream_with_snapshots(
            llm, [user_msg, *r1.output, *tool_outputs], tools=parallel_tools
        )
        assert r2.status == "completed"
        assert "42" in r2.output_text

    @pytest.mark.asyncio
    async def test_tool_call_signature_round_trip(
        self, llm: CloudLLM, parallel_tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text("What is 17 + 25? Use the add tool.")
        r1 = await llm.generate_response(
            [user_msg], tools=parallel_tools, tool_choice="required"
        )

        assert r1.tool_call_items
        assert r1.tool_call_items[0].provider_specific_fields
        assert r1.tool_call_items[0].provider_specific_fields["thought_signature"]

        tool_outputs = _execute_parallel_tools(r1.tool_call_items)
        r2 = await llm.generate_response(
            [user_msg, *r1.output, *tool_outputs], tools=parallel_tools
        )
        assert r2.status == "completed"
        assert "42" in r2.output_text

    @pytest.mark.asyncio
    async def test_stream_text_answer_signature_round_trip(self, llm: CloudLLM) -> None:
        user_msg = InputMessageItem.from_text(
            "Why is the sky blue? Answer in one sentence."
        )
        r1, emitted = await _stream_with_snapshots(llm, [user_msg])

        (message,) = [i for i in emitted if isinstance(i, OutputMessageItem)]
        assert message.provider_specific_fields
        assert len(message.provider_specific_fields["thought_signatures"]) == 1
        assert [i.model_dump() for i in emitted] == [i.model_dump() for i in r1.output]

        follow_up = InputMessageItem.from_text("Now say it in three words.")
        r2, _ = await _stream_with_snapshots(llm, [user_msg, *r1.output, follow_up])
        assert r2.status == "completed"
        assert r2.output_text
