"""
Gemini integration tests that call real LLM APIs.

Skipped by default. Run with:
    uv run pytest -m integration -k gemini
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from openai.types.responses.response_function_web_search import ActionOpenPage
from pydantic import BaseModel, Field

from grasp_agents.cloud_llm import APIProvider
from grasp_agents.types.content import OutputMessageText, UrlCitation
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_events import (
    OutputMessageTextPartTextDelta,
    ResponseCompleted,
)

if TYPE_CHECKING:
    from grasp_agents.cloud_llm import CloudLLM
    from grasp_agents.types.tool import BaseTool


class Capital(BaseModel):
    country: str = Field(description="Country name")
    capital: str = Field(description="Capital city name")


@pytest.mark.integration
class TestGeminiIntegration:
    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-2.0-flash",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={"max_output_tokens": 100},
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

        full_input = [user_msg, *response.output_items, *tool_outputs]
        final_response = await llm.generate_response(full_input, tools=tools)

        assert "42" in final_response.output_text


@pytest.mark.integration
class TestGeminiReasoningContinuity:
    """
    Reasoning traces (thought parts with signatures) must survive
    tool-call round-trips. Gemini requires thought_signature to be
    passed back for reasoning continuity.

    Gemini >=3.0 enforces signature validation (corrupted → error).
    """

    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-3.1-pro-preview",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={
                "max_output_tokens": 4096,
                "thinking_config": {
                    "thinking_level": "medium",
                    "include_thoughts": True,
                },
            },
        )

    @pytest.mark.asyncio
    async def test_reasoning_continuity_across_tool_call(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "Think carefully step by step through ALL combinations. "
            "From the set {3, 7, 11, 14, 19, 23, 28, 31, 37, 42}, find ALL pairs "
            "whose sum is exactly 45. List each pair you check and whether it works. "
            "Then use the add tool to verify EACH valid pair you found."
        )
        r1 = await llm.generate_response([user_msg], tools=tools)

        # First turn: reasoning + tool call
        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert len(r1.tool_call_items) >= 1

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Pass ALL output_items (reasoning + tool call) + tool outputs back
        full_input = [user_msg, *r1.output_items, *tool_outputs]
        r2 = await llm.generate_response(full_input, tools=tools)

        assert r2.status == "completed"
        assert "45" in r2.output_text

    @pytest.mark.asyncio
    async def test_stream_reasoning_continuity_across_tool_call(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "Think carefully step by step through ALL combinations. "
            "From the set {3, 7, 11, 14, 19, 23, 28, 31, 37, 42}, find ALL pairs "
            "whose sum is exactly 45. List each pair you check and whether it works. "
            "Then use the add tool to verify EACH valid pair you found."
        )
        # First turn: streaming
        events1 = [
            event
            async for event in llm.generate_response_stream([user_msg], tools=tools)
        ]
        completed1 = [e for e in events1 if isinstance(e, ResponseCompleted)]
        assert len(completed1) == 1
        r1 = completed1[0].response

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert len(r1.tool_call_items) >= 1

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Second turn: streaming with reasoning continuity
        full_input = [user_msg, *r1.output_items, *tool_outputs]
        events2 = [
            event
            async for event in llm.generate_response_stream(full_input, tools=tools)
        ]

        completed2 = [e for e in events2 if isinstance(e, ResponseCompleted)]
        assert len(completed2) == 1
        r2 = completed2[0].response
        assert r2.status == "completed"
        assert "45" in r2.output_text


@pytest.mark.integration
class TestGeminiCorruptedSignature:
    """Corrupting thought_signature must cause an API error."""

    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-3.1-pro-preview",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={
                "max_output_tokens": 4096,
                "thinking_config": {
                    "thinking_level": "medium",
                    "include_thoughts": True,
                },
            },
        )

    @pytest.mark.asyncio
    async def test_corrupted_signature_causes_api_error(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        """Corrupting thought_signature on function call items must error."""
        user_msg = InputMessageItem.from_text(
            "Think carefully step by step through ALL combinations. "
            "From the set {3, 7, 11, 14, 19, 23, 28, 31, 37, 42}, find ALL pairs "
            "whose sum is exactly 45. List each pair you check and whether it works. "
            "Then use the add tool to verify EACH valid pair you found."
        )
        r1 = await llm.generate_response([user_msg], tools=tools)

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert len(r1.tool_call_items) >= 1

        # Verify tool calls carry thought signatures
        signed_tcs = [
            tc
            for tc in r1.tool_call_items
            if (tc.provider_specific_fields or {}).get("thought_signature")
        ]
        assert len(signed_tcs) >= 1, "Tool calls must have thought_signature"

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Corrupt the thought_signature on function call items
        corrupted_items = [
            item.model_copy(
                update={"provider_specific_fields": {"thought_signature": "CORRUPTED"}}
            )
            if isinstance(item, FunctionToolCallItem)
            and (item.provider_specific_fields or {}).get("thought_signature")
            else item
            for item in r1.output_items
        ]

        full_input = [user_msg, *corrupted_items, *tool_outputs]

        with pytest.raises(Exception):  # noqa: B017, PT011
            await llm.generate_response(full_input, tools=tools)

    @pytest.mark.asyncio
    async def test_missing_signature_causes_api_error(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        """Stripping thought_signature from function call items must error."""
        user_msg = InputMessageItem.from_text(
            "Think carefully step by step through ALL combinations. "
            "From the set {3, 7, 11, 14, 19, 23, 28, 31, 37, 42}, find ALL pairs "
            "whose sum is exactly 45. List each pair you check and whether it works. "
            "Then use the add tool to verify EACH valid pair you found."
        )
        r1 = await llm.generate_response([user_msg], tools=tools)

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert len(r1.tool_call_items) >= 1

        signed_tcs = [
            tc
            for tc in r1.tool_call_items
            if (tc.provider_specific_fields or {}).get("thought_signature")
        ]
        assert len(signed_tcs) >= 1, "Tool calls must have thought_signature"

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Strip thought_signature from function call items
        stripped_items = [
            item.model_copy(update={"provider_specific_fields": None})
            if isinstance(item, FunctionToolCallItem)
            else item
            for item in r1.output_items
        ]

        full_input = [user_msg, *stripped_items, *tool_outputs]

        with pytest.raises(Exception):  # noqa: B017, PT011
            await llm.generate_response(full_input, tools=tools)


@pytest.mark.integration
class TestGeminiWebSearch:
    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={
                "max_output_tokens": 4096,
                "google_search": {},
            },
        )

    @pytest.mark.asyncio
    async def test_web_search(self, llm: CloudLLM) -> None:
        """Google Search grounding should produce sources and annotations."""
        input_items = [
            InputMessageItem.from_text(
                "What were the major tech news headlines this week?"
            )
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text
        assert response.web_search is not None
        assert len(response.web_search.sources) > 0

        msg = response.message_items[0]
        all_annotations = [
            ann
            for part in msg.content_parts
            if isinstance(part, OutputMessageText)
            for ann in part.annotations
        ]
        assert len(all_annotations) > 0
        assert all_annotations[0].type == "url_citation"
        assert isinstance(all_annotations[0], UrlCitation)
        assert all_annotations[0].provider_specific_fields["gemini:grounded_text"]

    @pytest.mark.asyncio
    async def test_stream_web_search(self, llm: CloudLLM) -> None:
        """Streaming Google Search: sources, annotations, grounded_text."""
        input_items = [
            InputMessageItem.from_text(
                "What were the major tech news headlines this week?"
            )
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text
        assert response.web_search is not None
        assert len(response.web_search.sources) > 0

        msg = response.message_items[0]
        all_annotations = [
            ann
            for part in msg.content_parts
            if isinstance(part, OutputMessageText)
            for ann in part.annotations
        ]
        assert len(all_annotations) > 0
        assert all_annotations[0].type == "url_citation"
        assert isinstance(all_annotations[0], UrlCitation)
        assert all_annotations[0].provider_specific_fields["gemini:grounded_text"]


@pytest.mark.integration
class TestGeminiStructuredOutput:
    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={"max_output_tokens": 200},
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


@pytest.mark.integration
class TestGeminiUrlContext:
    @pytest.fixture
    def llm(self, google_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider=APIProvider(
                name="google",
                base_url=None,
                api_key=google_api_key,
            ),
            llm_settings={
                "max_output_tokens": 4096,
                "url_context": True,
            },
        )

    @pytest.mark.asyncio
    async def test_url_context(self, llm: CloudLLM) -> None:
        """UrlContext should produce WebSearchCallItem with ActionOpenPage."""
        input_items = [
            InputMessageItem.from_text(
                "Read https://httpbin.org/html and summarize the content."
            )
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text
        wf_items = [
            i
            for i in response.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, ActionOpenPage)
        ]
        assert len(wf_items) >= 1
        assert wf_items[0].action.url  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_stream_url_context(self, llm: CloudLLM) -> None:
        """Streaming UrlContext should produce WebSearchCallItem events."""
        input_items = [
            InputMessageItem.from_text(
                "Read https://httpbin.org/html and summarize the content."
            )
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text

        wf_items = [
            i
            for i in response.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, ActionOpenPage)
        ]
        assert len(wf_items) >= 1

    @pytest.mark.asyncio
    async def test_url_context_unreachable(self, llm: CloudLLM) -> None:
        """Unreachable URL should produce a failed WebSearchCallItem."""
        input_items = [
            InputMessageItem.from_text(
                "Read https://this-domain-does-not-exist-abc123xyz.invalid/page "
                "and tell me what it says."
            )
        ]
        response = await llm.generate_response(input_items)

        wf_items = [
            i
            for i in response.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, ActionOpenPage)
        ]
        assert len(wf_items) >= 1
        assert any(wf.status == "failed" for wf in wf_items)
