"""Unit tests for OpenAI Responses API tool converters."""

from __future__ import annotations

from typing import Any

from grasp_agents.llm_providers.openai_responses.tool_converters import (
    to_api_tool,
    to_api_tool_choice,
)
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
