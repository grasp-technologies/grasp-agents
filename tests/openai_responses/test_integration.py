"""
OpenAI Responses API integration tests that call real LLM APIs.

Skipped by default. Run with:
    uv run pytest -m integration -k openai_responses
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.types.content import OutputMessageText, UrlCitation
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OpenPageAction,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_events import (
    OutputItemDone,
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
class TestOpenAIResponsesIntegration:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-4.1-nano", llm_settings={"max_output_tokens": 100}
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
class TestOpenAIResponsesStructuredOutput:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-4.1-nano",
            llm_settings={"max_output_tokens": 200},
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
class TestOpenAIResponsesWebSearch:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-5.2",
            llm_settings={
                "max_output_tokens": 4096,
                "web_search": {"type": "web_search_preview"},
            },
        )

    @pytest.mark.asyncio
    async def test_web_search(self, llm: CloudLLM) -> None:
        """Responses API web search should produce WebSearchCallItem and citations."""
        input_items = [
            InputMessageItem.from_text("Tell me the current NASDAQ Composite index.")
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text

        # WebSearchCallItem should appear in output_items
        ws_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) >= 1

        msg = response.message_items[0]
        all_citations = [
            c
            for part in msg.content_parts
            if isinstance(part, OutputMessageText)
            for c in part.citations
        ]
        assert len(all_citations) > 0
        assert isinstance(all_citations[0], UrlCitation)
        assert all_citations[0].url

    @pytest.mark.asyncio
    async def test_stream_web_search(self, llm: CloudLLM) -> None:
        """Streaming web search should produce WebSearchCallItem and citations."""
        input_items = [
            InputMessageItem.from_text("Tell me the current NASDAQ Composite index.")
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text

        ws_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) >= 1

        msg = response.message_items[0]
        all_citations = [
            c
            for part in msg.content_parts
            if isinstance(part, OutputMessageText)
            for c in part.citations
        ]
        assert len(all_citations) > 0
        assert isinstance(all_citations[0], UrlCitation)
        assert all_citations[0].url


@pytest.mark.integration
class TestOpenAIResponsesReasoningContinuity:
    """
    Reasoning traces must survive tool-call round-trips.

    The Responses API passes items directly (no message conversion),
    so reasoning items are natively preserved. This verifies the
    full round-trip works and each turn produces reasoning summaries.
    """

    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-5.2",
            llm_settings={
                "max_output_tokens": 4096,
                "reasoning": {"effort": "low", "summary": "auto"},
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
        r1 = await llm.generate_response(
            [user_msg], tools=tools, tool_choice="required"
        )

        # First turn: reasoning + tool call
        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert r1.reasoning_items[0].summary_text
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
        events1 = [
            event
            async for event in llm.generate_response_stream(
                [user_msg], tools=tools, tool_choice="required"
            )
        ]
        completed1 = [e for e in events1 if isinstance(e, ResponseCompleted)]
        assert len(completed1) == 1
        r1 = completed1[0].response

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"
        assert r1.reasoning_items[0].summary_text
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


@pytest.mark.integration
class TestOpenAIResponsesCorruptedEncryptedContent:
    """
    Corrupting encrypted_content must cause an API error.

    Uses store=False + include=["reasoning.encrypted_content"] to get
    encrypted reasoning items, then corrupts them before the second turn.
    """

    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-5-mini",
            llm_settings={
                "max_output_tokens": 4096,
                "reasoning": {"effort": "low", "summary": "auto"},
                "store": False,
                "include": ["reasoning.encrypted_content"],
            },
        )

    @pytest.mark.asyncio
    async def test_corrupted_encrypted_content_causes_api_error(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        r1 = await llm.generate_response(
            [user_msg], tools=tools, tool_choice="required"
        )

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"

        # Verify we got encrypted_content
        encrypted_items = [i for i in r1.reasoning_items if i.encrypted_content]
        assert len(encrypted_items) >= 1, "Reasoning items must have encrypted_content"

        assert len(r1.tool_call_items) >= 1

        tool_outputs = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output=json.loads(tc.arguments)["a"] + json.loads(tc.arguments)["b"],
            )
            for tc in r1.tool_call_items
        ]

        # Corrupt the encrypted_content on reasoning items
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
    async def test_missing_encrypted_content_causes_api_error(
        self, llm: CloudLLM, tools: dict[str, BaseTool[Any, Any, Any]]
    ) -> None:
        """Stripping encrypted_content from reasoning items must error."""
        user_msg = InputMessageItem.from_text(
            "From the set {6, 11, 17, 19, 25, 33}, find the unique pair "
            "whose sum is exactly 42. Work through the combinations, "
            "then use the add tool to verify your answer."
        )
        r1 = await llm.generate_response(
            [user_msg], tools=tools, tool_choice="required"
        )

        assert len(r1.reasoning_items) >= 1, "First turn must produce reasoning"

        encrypted_items = [i for i in r1.reasoning_items if i.encrypted_content]
        assert len(encrypted_items) >= 1, "Reasoning items must have encrypted_content"

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
class TestOpenAIResponsesWebFetch:
    """Web search with browsing can produce OpenPageAction items."""

    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-5.4",
            llm_settings={
                "max_output_tokens": 4096,
                "web_search": {"type": "web_search_preview"},
            },
        )

    @pytest.mark.asyncio
    async def test_web_fetch_via_search(self, llm: CloudLLM) -> None:
        """Ask to read a specific URL — web search should browse it."""
        input_items = [
            InputMessageItem.from_text(
                "Go to https://httpbin.org/html and tell me the author's name "
                "mentioned on that page."
            )
        ]
        response = await llm.generate_response(input_items)

        assert response.output_text
        ws_items = [
            i for i in response.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) >= 1

        open_page_items = [i for i in ws_items if isinstance(i.action, OpenPageAction)]
        assert len(open_page_items) >= 1
        assert open_page_items[0].action.url

    @pytest.mark.asyncio
    async def test_stream_web_fetch_via_search(self, llm: CloudLLM) -> None:
        """Streaming: OpenPageAction items appear in stream events."""
        input_items = [
            InputMessageItem.from_text(
                "Go to https://httpbin.org/html and tell me the author's name "
                "mentioned on that page."
            )
        ]
        events = [event async for event in llm.generate_response_stream(input_items)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        response = completed[0].response
        assert response.output_text

        open_page_items = [
            i
            for i in response.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, OpenPageAction)
        ]
        assert len(open_page_items) >= 1

    @pytest.mark.asyncio
    async def test_web_fetch_unreachable(self, llm: CloudLLM) -> None:
        """Unreachable URL — OpenAI returns OpenPageAction with url=None."""
        input_items = [
            InputMessageItem.from_text(
                "Go to https://this-domain-does-not-exist-abc123xyz.invalid/page "
                "and tell me what it says."
            )
        ]
        response = await llm.generate_response(input_items)

        open_page_items = [
            i
            for i in response.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, OpenPageAction)
        ]
        # OpenAI still produces OpenPageAction but with url=None for unreachable
        assert len(open_page_items) >= 1
        assert any(op.action.url is None for op in open_page_items)

    @pytest.mark.asyncio
    async def test_web_fetch_multi_turn(self, llm: CloudLLM) -> None:
        """Multi-turn: browse a page, then ask a follow-up about it."""
        user_msg = InputMessageItem.from_text(
            "Go to https://httpbin.org/html and tell me the author's name."
        )
        r1 = await llm.generate_response([user_msg])

        open_page_items = [
            i
            for i in r1.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, OpenPageAction)
        ]
        assert len(open_page_items) >= 1

        follow_up = InputMessageItem.from_text(
            "What was the title of the page you visited?"
        )
        full_input = [user_msg, *r1.output_items, follow_up]
        r2 = await llm.generate_response(full_input)

        assert r2.output_text
        assert r2.status == "completed"

    @pytest.mark.asyncio
    async def test_stream_web_fetch_multi_turn(self, llm: CloudLLM) -> None:
        """Streaming multi-turn: browse then follow up."""
        user_msg = InputMessageItem.from_text(
            "Go to https://httpbin.org/html and tell me the author's name."
        )
        r1 = await llm.generate_response([user_msg])

        open_page_items = [
            i
            for i in r1.output_items
            if isinstance(i, WebSearchCallItem) and isinstance(i.action, OpenPageAction)
        ]
        assert len(open_page_items) >= 1

        follow_up = InputMessageItem.from_text(
            "What was the title of the page you visited?"
        )
        full_input = [user_msg, *r1.output_items, follow_up]
        events = [event async for event in llm.generate_response_stream(full_input)]

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert len(completed) == 1
        assert completed[0].response.output_text
        assert completed[0].response.status == "completed"


def _execute_parallel_tools(
    tool_calls: list[FunctionToolCallItem],
) -> list[FunctionToolOutputItem]:
    tool_outputs: list[FunctionToolOutputItem] = []
    for tc in tool_calls:
        args = json.loads(tc.arguments)
        if tc.name == "add":
            result = args["a"] + args["b"]
        else:
            result = args["a"] * args["b"]
        tool_outputs.append(
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id, output=result
            )
        )
    return tool_outputs


@pytest.mark.integration
class TestOpenAIResponsesParallelToolUse:
    @pytest.fixture
    def llm(self, openai_api_key: str) -> CloudLLM:  # noqa: ARG002
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-4.1-nano",
            llm_settings={"max_output_tokens": 256},
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
        full_input = [user_msg, *r1.output_items, *tool_outputs]
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
        full_input = [user_msg, *r1.output_items, *tool_outputs]
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
