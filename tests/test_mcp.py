"""Tests for MCP client integration.

Uses the test MCP server at tests/mcp_test_server.py via stdio transport.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.mcp.client import MCPClient, MCPServerStdio
from grasp_agents.mcp.json_schema import json_schema_to_pydantic
from grasp_agents.mcp.tool import MCPTool

_SERVER_PATH = str(Path(__file__).parent / "mcp_test_server.py")
_SERVER_CONFIG = MCPServerStdio(command=sys.executable, args=[_SERVER_PATH])


# ---------- json_schema_to_pydantic ----------


class TestJsonSchemaToPydantic:
    def test_simple_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }
        model = json_schema_to_pydantic(schema, "AddInput")
        instance = model(a=1, b=2)
        assert instance.model_dump() == {"a": 1, "b": 2}

    def test_optional_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        model = json_schema_to_pydantic(schema, "Person")
        instance = model(name="Alice")
        assert instance.model_dump() == {"name": "Alice", "age": None}

    def test_array_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        model = json_schema_to_pydantic(schema, "Tagged")
        instance = model(tags=["a", "b"])
        assert instance.model_dump() == {"tags": ["a", "b"]}

    def test_boolean_and_number_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"},
                "score": {"type": "number"},
            },
            "required": ["active", "score"],
        }
        model = json_schema_to_pydantic(schema, "Record")
        instance = model(active=True, score=3.14)
        assert instance.model_dump() == {"active": True, "score": 3.14}

    def test_empty_schema(self) -> None:
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic(schema, "Empty")
        instance = model()
        assert instance.model_dump() == {}


# ---------- MCPClient integration tests ----------


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_connect_and_discover_tools(self) -> None:
        """Connect to test server and discover available tools."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = client.tools()
            tool_names = {t.name for t in tools}
            assert "add" in tool_names
            assert "echo" in tool_names
            assert "failing_tool" in tool_names
            assert "slow_tool" in tool_names

    @pytest.mark.asyncio
    async def test_tools_are_mcp_tool_instances(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = client.tools()
            for tool in tools:
                assert isinstance(tool, MCPTool)

    @pytest.mark.asyncio
    async def test_tool_has_description(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            assert tools["add"].description == "Add two numbers."

    @pytest.mark.asyncio
    async def test_tool_has_valid_input_schema(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            in_type = tools["add"].in_type
            assert issubclass(in_type, BaseModel)
            # Should accept a=1, b=2
            instance = in_type(a=1, b=2)
            assert instance.model_dump() == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_call_add_tool(self) -> None:
        """Call the 'add' tool and verify the result."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["add"](a=3, b=4)
            assert not result.isError
            assert result.content[0].text == "7"

    @pytest.mark.asyncio
    async def test_call_echo_tool(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["echo"](message="hello world")
            assert not result.isError
            assert result.content[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_failing_tool_returns_error(self) -> None:
        """Failing MCP tool returns McpToolResult with isError=True."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["failing_tool"]()
            assert result.isError

    @pytest.mark.asyncio
    async def test_tool_timeout(self) -> None:
        """MCP tool with short timeout returns ToolErrorInfo."""
        from grasp_agents.types.events import ToolErrorInfo

        async with MCPClient(
            "test", server=_SERVER_CONFIG, tool_timeout=0.01
        ) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["slow_tool"]()
            assert isinstance(result, ToolErrorInfo)
            assert "Timed out" in result.error or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_structured_output_schema_discovered(self) -> None:
        """Tool with outputSchema has struct_output_schema auto-inferred."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            multiply = tools["multiply"]
            assert multiply.struct_output_schema is not None
            assert issubclass(multiply.struct_output_schema, BaseModel)
            fields = multiply.struct_output_schema.model_fields
            assert "result" in fields
            assert "operation" in fields

    @pytest.mark.asyncio
    async def test_structured_output_result(self) -> None:
        """Calling a structured output tool returns McpToolResult with structuredContent."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            multiply = tools["multiply"]
            result = await multiply(a=3.0, b=4.0)
            assert not result.isError
            # structuredContent has the validated data
            assert result.structuredContent is not None
            assert result.structuredContent["result"] == 12.0
            assert result.structuredContent["operation"] == "multiply"

    @pytest.mark.asyncio
    async def test_tool_without_output_schema(self) -> None:
        """Tools without outputSchema have struct_output_schema=None."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            tool = tools["no_schema_tool"]
            assert tool.struct_output_schema is None
            result = await tool(x=42)
            assert not result.isError

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        """Accessing tools before connect() raises RuntimeError."""
        client = MCPClient("test", server=_SERVER_CONFIG)
        with pytest.raises(RuntimeError, match="Not connected"):
            client.tools()

    @pytest.mark.asyncio
    async def test_connect_twice_is_idempotent(self) -> None:
        """Calling connect() twice doesn't error."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            await client.connect()  # second call
            tools = client.tools()
            assert len(tools) > 0
