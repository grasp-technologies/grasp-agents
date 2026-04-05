"""Unit tests for OpenAI Responses API converters."""

from __future__ import annotations

from typing import Any

from openai.types.responses import Response as OAIResponse

from grasp_agents.llm_providers.openai_responses.response_to_provider_inputs import (
    items_to_provider_inputs,
)
from grasp_agents.llm_providers.openai_responses.tool_converters import (
    to_api_tool,
    to_api_tool_choice,
)
from grasp_agents.types.items import OpenPageAction, SearchAction, WebSearchCallItem
from grasp_agents.types.response import Response as InternalResponse
from grasp_agents.types.tool import NamedToolChoice


def _make_add_tool() -> Any:
    """Reuse the same AddTool definition from conftest."""
    from tests.conftest import AddTool

    return AddTool()


class TestResponsesToolConverters:
    def test_to_api_tool(self) -> None:
        tool = _make_add_tool()
        result = to_api_tool(tool)

        assert result["type"] == "function"
        assert result["name"] == "add"
        assert result["description"] == "Add two integers and return their sum."
        assert result["strict"] is True
        assert "properties" in result["parameters"]

    def test_to_api_tool_explicit_strict_false(self) -> None:
        tool = _make_add_tool()
        result = to_api_tool(tool, strict=False)

        assert result["strict"] is False

    def test_to_api_tool_choice_auto(self) -> None:
        assert to_api_tool_choice("auto") == "auto"

    def test_to_api_tool_choice_required(self) -> None:
        assert to_api_tool_choice("required") == "required"

    def test_to_api_tool_choice_named(self) -> None:
        result = to_api_tool_choice(NamedToolChoice(name="add"))

        assert result["type"] == "function"
        assert result["name"] == "add"


# ==== WebSearchCallItem with OpenPageAction ====


_BASE_RESPONSE: dict[str, Any] = {
    "object": "response",
    "created_at": 1700000000,
    "model": "gpt-4.1-mini",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def _make_oai_response_with_web_fetch() -> OAIResponse:
    """Build a minimal OpenAI Response containing a web_search_call with OpenPageAction."""
    return OAIResponse.model_validate(
        {
            **_BASE_RESPONSE,
            "id": "resp_test_wf",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_open_page_1",
                    "status": "completed",
                    "action": {
                        "type": "open_page",
                        "url": "https://example.com/page",
                    },
                },
                {
                    "type": "web_search_call",
                    "id": "ws_search_1",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "example query",
                        "queries": ["example query"],
                        "sources": [],
                    },
                },
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Page content.",
                            "annotations": [],
                        },
                    ],
                },
            ],
        }
    )


class TestWebFetchRoundtrip:
    """WebSearchCallItem(OpenPageAction) survives model_dump/model_validate."""

    def test_response_to_internal(self) -> None:
        """OAI Response with OpenPageAction → InternalResponse preserves action type."""
        oai_resp = _make_oai_response_with_web_fetch()
        internal = InternalResponse.model_validate(
            oai_resp.model_dump(warnings="none", by_alias=True)
        )

        ws_items = [
            i for i in internal.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) == 2

        # OpenPageAction
        open_page = ws_items[0]
        assert isinstance(open_page.action, OpenPageAction)
        assert open_page.action.url == "https://example.com/page"
        assert open_page.status == "completed"

        # SearchAction
        search = ws_items[1]
        assert isinstance(search.action, SearchAction)
        assert "example query" in search.action.queries

    def test_sanitize_strips_provider_fields(self) -> None:
        """items_to_provider_inputs removes provider_specific_fields from items."""
        item = WebSearchCallItem(
            id="ws_test",
            status="completed",
            action=OpenPageAction(url="https://example.com"),
            provider_specific_fields={"anthropic:data": "should be stripped"},
        )
        sanitized = items_to_provider_inputs([item])

        assert len(sanitized) == 1
        assert "provider_specific_fields" not in sanitized[0]
        assert sanitized[0]["action"]["type"] == "open_page"
        assert sanitized[0]["action"]["url"] == "https://example.com"

    def test_failed_status_preserved(self) -> None:
        """Failed WebSearchCallItem round-trips correctly."""
        oai_data = {
            **_BASE_RESPONSE,
            "id": "resp_fail",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_fail_1",
                    "status": "failed",
                    "action": {
                        "type": "open_page",
                        "url": "https://unreachable.invalid",
                    },
                },
            ],
        }
        oai_resp = OAIResponse.model_validate(oai_data)
        internal = InternalResponse.model_validate(
            oai_resp.model_dump(warnings="none", by_alias=True)
        )

        ws_items = [
            i for i in internal.output_items if isinstance(i, WebSearchCallItem)
        ]
        assert len(ws_items) == 1
        assert ws_items[0].status == "failed"
        assert isinstance(ws_items[0].action, OpenPageAction)
        assert ws_items[0].action.url == "https://unreachable.invalid"
