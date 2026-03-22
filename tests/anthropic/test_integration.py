"""
Anthropic integration tests that call real LLM APIs.

Skipped by default. Run with:
    uv run pytest -m integration -k anthropic
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
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_events import (
    OutputItemDone,
    OutputMessageTextDelta,
    ResponseCompleted,
)

if TYPE_CHECKING:
    from grasp_agents.cloud_llm import CloudLLM
    from grasp_agents.types.tool import BaseTool


class Capital(BaseModel):
    country: str = Field(description="Country name")
    capital: str = Field(description="Capital city name")


@pytest.mark.integration
class TestAnthropicIntegration:
    @pytest.fixture
    def llm(self, anthropic_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model_name="claude-haiku-4-5-20251001",
            api_provider=APIProvider(
                name="anthropic",
                base_url=None,
                api_key=anthropic_api_key,
            ),
            llm_settings={"max_tokens": 100},
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
        text_deltas = [e for e in events if isinstance(e, OutputMessageTextDelta)]
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
class TestAnthropicReasoningContinuity:
    """
    Reasoning traces (thinking blocks) must survive tool-call round-trips.

    Anthropic requires thinking blocks with signatures to be passed back
    in the assistant message for the next turn.
    """

    @pytest.fixture
    def llm(self, anthropic_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model_name="claude-haiku-4-5-20251001",
            api_provider=APIProvider(
                name="anthropic",
                base_url=None,
                api_key=anthropic_api_key,
            ),
            llm_settings={
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        )

    @pytest.mark.asyncio
    async def test_reasoning_continuity_across_tool_call(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        # Anthropic does not allow tool_choice="required" with thinking
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
        assert "42" in r2.output_text

    @pytest.mark.asyncio
    async def test_stream_reasoning_continuity_across_tool_call(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        # First turn: streaming
        # Anthropic does not allow tool_choice="required" with thinking
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
        assert "42" in r2.output_text

    @pytest.mark.asyncio
    async def test_corrupted_signature_causes_api_error(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        """Corrupting thinking block signature must cause an API error."""
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        r1 = await llm.generate_response([user_msg], tools=tools)

        assert len(r1.reasoning_items) >= 1
        assert len(r1.tool_call_items) >= 1

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Corrupt the signature on reasoning items
        corrupted_items = [
            item.model_copy(update={"encrypted_content": "CORRUPTED"})
            if isinstance(item, ReasoningItem) and item.encrypted_content
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
        """Stripping encrypted_content from reasoning items must error."""
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        r1 = await llm.generate_response([user_msg], tools=tools)

        assert len(r1.reasoning_items) >= 1
        assert len(r1.tool_call_items) >= 1

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Strip encrypted_content from reasoning items
        stripped_items = [
            item.model_copy(update={"encrypted_content": None})
            if isinstance(item, ReasoningItem)
            else item
            for item in r1.output_items
        ]

        full_input = [user_msg, *stripped_items, *tool_outputs]

        with pytest.raises(Exception):  # noqa: B017, PT011
            await llm.generate_response(full_input, tools=tools)


@pytest.mark.integration
class TestAnthropicWebSearch:
    @pytest.fixture
    def llm(self, anthropic_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model_name="claude-sonnet-4-6",
            api_provider=APIProvider(
                name="anthropic",
                base_url=None,
                api_key=anthropic_api_key,
            ),
            llm_settings={
                "max_tokens": 4096,
                "web_search": {"type": "web_search_20250305", "name": "web_search"},
            },
            anthropic_client_timeout=120.0,
        )

    @pytest.mark.asyncio
    async def test_web_search(self, llm: CloudLLM) -> None:
        """Web search should produce WebSearchCallItem, sources, and citations."""
        input_items = [
            InputMessageItem.from_text(
                "What were the major tech news headlines this week?"
            )
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text
        assert response.web_search is not None
        assert len(response.web_search.sources) > 0

        # WebSearchCallItem should appear in output_items
        ws_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) >= 1
        ws = ws_items[0]
        assert ws.status == "completed"
        assert ws.provider_specific_fields is not None
        encrypted = ws.provider_specific_fields.get("anthropic:encrypted_content", {})
        assert len(encrypted) > 0

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
        assert all_annotations[0].provider_specific_fields["anthropic:cited_text"]

    @pytest.mark.asyncio
    async def test_stream_web_search(self, llm: CloudLLM) -> None:
        """Streaming web search should produce WebSearchCallItem events."""
        input_items = [
            InputMessageItem.from_text(
                "What were the major tech news headlines this week?"
            )
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        # WebSearchCallItem should appear via OutputItemDone
        ws_done = [
            e
            for e in events
            if isinstance(e, OutputItemDone) and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_done) >= 1
        assert ws_done[0].item.provider_specific_fields is not None

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text
        assert response.web_search is not None
        assert len(response.web_search.sources) > 0

        # Citations should be captured from citations_delta events
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
        assert all_annotations[0].provider_specific_fields["anthropic:cited_text"]

    @pytest.mark.asyncio
    async def test_web_search_multi_turn(self, llm: CloudLLM) -> None:
        """WebSearchCallItem round-trips: 2nd turn receives prior search context."""
        user_msg = InputMessageItem.from_text(
            "What is the latest Python release version? Be brief."
        )
        r1 = await llm.generate_response([user_msg])

        ws_items = [i for i in r1.output_items if isinstance(i, WebSearchCallItem)]
        assert len(ws_items) >= 1, "First turn should produce web search items"

        # Round-trip: pass all output_items (including WebSearchCallItem) back
        follow_up = InputMessageItem.from_text("What was the release date?")
        full_input = [user_msg, *r1.output_items, follow_up]
        r2 = await llm.generate_response(full_input)

        assert r2.output_text
        assert r2.status == "completed"

    @pytest.mark.asyncio
    async def test_stream_web_search_multi_turn(self, llm: CloudLLM) -> None:
        """Streaming multi-turn: 2nd streamed turn receives prior search context."""
        user_msg = InputMessageItem.from_text(
            "What is the latest Python release version? Be brief."
        )
        # First turn: non-streaming to get output_items
        r1 = await llm.generate_response([user_msg])

        ws_items = [i for i in r1.output_items if isinstance(i, WebSearchCallItem)]
        assert len(ws_items) >= 1, "First turn should produce web search items"

        # Second turn: streaming with prior context
        follow_up = InputMessageItem.from_text("What was the release date?")
        full_input = [user_msg, *r1.output_items, follow_up]
        events = [event async for event in llm.generate_response_stream(full_input)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text
        assert response.status == "completed"


@pytest.mark.integration
class TestAnthropicStructuredOutput:
    @pytest.fixture
    def llm(self, anthropic_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.anthropic.anthropic_llm import (
            AnthropicLLM,
        )

        return AnthropicLLM(
            model_name="claude-haiku-4-5-20251001",
            api_provider=APIProvider(
                name="anthropic",
                base_url=None,
                api_key=anthropic_api_key,
            ),
            llm_settings={"max_tokens": 200},
            apply_response_schema_via_provider=True,
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
class TestAnthropicWebFetch:
    @pytest.fixture
    def llm(self, anthropic_api_key: str) -> CloudLLM:
        from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model_name="claude-sonnet-4-6",
            api_provider=APIProvider(
                name="anthropic",
                base_url=None,
                api_key=anthropic_api_key,
            ),
            llm_settings={
                "max_tokens": 4096,
                "web_fetch": {
                    "type": "web_fetch_20260209",
                    "name": "web_fetch",
                    "max_uses": 2,
                },
            },
            anthropic_client_timeout=120.0,
        )

    @pytest.mark.asyncio
    async def test_web_fetch(self, llm: CloudLLM) -> None:
        """Web fetch should produce WebSearchCallItem with ActionOpenPage."""
        input_items = [
            InputMessageItem.from_text(
                "Fetch https://httpbin.org/html and summarize it briefly."
            )
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text
        wf_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(wf_items) >= 1
        wf = wf_items[0]
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.status == "completed"
        assert wf.action.url

    @pytest.mark.asyncio
    async def test_stream_web_fetch(self, llm: CloudLLM) -> None:
        """Streaming web fetch should produce WebSearchCallItem events."""
        input_items = [
            InputMessageItem.from_text(
                "Fetch https://httpbin.org/html and summarize it briefly."
            )
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        ws_done = [
            e
            for e in events
            if isinstance(e, OutputItemDone)
            and isinstance(e.item, WebSearchCallItem)
            and isinstance(e.item.action, ActionOpenPage)
        ]
        assert len(ws_done) >= 1

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text

    @pytest.mark.asyncio
    async def test_web_fetch_multi_turn(self, llm: CloudLLM) -> None:
        """Multi-turn: fetch a page, then ask a follow-up about it."""
        user_msg = InputMessageItem.from_text(
            "Fetch https://httpbin.org/html and tell me the author's name."
        )
        r1 = await llm.generate_response([user_msg])

        wf_items = [i for i in r1.output_items if isinstance(i, WebSearchCallItem)]
        assert len(wf_items) >= 1

        follow_up = InputMessageItem.from_text(
            "What was the title of the page you fetched?"
        )
        full_input = [user_msg, *r1.output_items, follow_up]
        r2 = await llm.generate_response(full_input)

        assert r2.output_text
        assert r2.status == "completed"

    @pytest.mark.asyncio
    async def test_web_fetch_unreachable(self, llm: CloudLLM) -> None:
        """Unreachable URL should produce a failed WebSearchCallItem."""
        input_items = [
            InputMessageItem.from_text(
                "Fetch https://this-domain-does-not-exist-abc123xyz.invalid/page "
                "and tell me what it says."
            )
        ]
        response = await llm.generate_response(input_items)

        wf_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(wf_items) >= 1
        assert any(
            wf.status == "failed" and isinstance(wf.action, ActionOpenPage)
            for wf in wf_items
        )
