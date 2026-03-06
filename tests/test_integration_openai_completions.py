"""
OpenAI Completions API integration tests that call real LLM APIs.

Skipped by default. Run with:
    uv run pytest -m integration -k openai_completions
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.types.items import (
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import (
    ResponseCompleted,
    TextDelta,
)

if TYPE_CHECKING:
    from grasp_agents.cloud_llm import CloudLLM
    from grasp_agents.types.tool import BaseTool


class Capital(BaseModel):
    country: str = Field(description="Country name")
    capital: str = Field(description="Capital city name")


@pytest.mark.integration
class TestOpenAICompletionsIntegration:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.openai_completions.completions_llm import (
            OpenAILLM,
        )

        return OpenAILLM(
            model_name="openai/gpt-4.1-nano",
            llm_settings={"max_completion_tokens": 100},
        )

    @pytest.mark.asyncio
    async def test_generate_text(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("Say 'hello' and nothing else.")]
        response = await llm.generate_response(input_items)

        assert response.status == "completed"
        assert len(response.output_items) >= 1
        assert isinstance(response.output_items[0], OutputMessageItem)
        assert "hello" in response.output_text.lower()
        assert response.usage_with_cost is not None
        assert response.usage_with_cost.input_tokens > 0
        assert response.usage_with_cost.output_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_text(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("Say 'hello' and nothing else.")]
        events = [event async for event in llm.generate_response_stream(input_items)]
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
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

        full_input = [user_msg, *response.output_items, *tool_outputs]
        final_response = await llm.generate_response(full_input, tools=tools)

        assert "42" in final_response.output_text


@pytest.mark.integration
class TestOpenAICompletionsStructuredOutput:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.openai_completions.completions_llm import (
            OpenAILLM,
        )

        return OpenAILLM(
            model_name="openai/gpt-4.1-nano", apply_response_schema_via_provider=True
        )

    @pytest.mark.asyncio
    async def test_structured_output(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("What is the capital of France?")]
        response = await llm.generate_response(input_items, response_schema=Capital)

        parsed = Capital.model_validate_json(response.output_text)
        assert parsed.capital.lower() == "paris"
        assert parsed.country.lower() == "france"

    @pytest.mark.asyncio
    async def test_stream_structured_output(self, llm: CloudLLM) -> None:
        input_items = [InputMessageItem.from_text("What is the capital of France?")]
        events = [
            event
            async for event in llm.generate_response_stream(
                input_items, response_schema=Capital
            )
        ]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        parsed = Capital.model_validate_json(response.output_text)
        assert parsed.capital.lower() == "paris"
        assert parsed.country.lower() == "france"
