"""
Tests for MCP client integration.

Uses the test MCP server at tests/mcp_test_server.py via stdio transport.
"""

import asyncio
import math
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.mcp.client import MCPClient, MCPServerStdio
from grasp_agents.mcp.json_schema import json_schema_to_pydantic
from grasp_agents.mcp.resource import MCPListResourcesTool, MCPReadResourceTool
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
        instance = model(active=True, score=math.pi)
        assert instance.model_dump() == {"active": True, "score": math.pi}

    def test_empty_schema(self) -> None:
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic(schema, "Empty")
        instance = model()
        assert instance.model_dump() == {}

    def test_enum_field(self) -> None:
        from enum import StrEnum

        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                },
                "name": {"type": "string"},
            },
            "required": ["status", "name"],
        }
        model = json_schema_to_pydantic(schema, "WithEnum")
        instance = model(status="active", name="test")
        assert isinstance(instance.status, StrEnum)
        assert instance.status == "active"
        assert instance.model_dump(mode="json") == {
            "status": "active",
            "name": "test",
        }

    def test_union_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ],
                },
            },
            "required": ["value"],
        }
        model = json_schema_to_pydantic(schema, "WithUnion")
        assert model(value="hello").value == "hello"
        assert model(value=42).value == 42

    def test_nested_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["x", "y"],
                },
            },
            "required": ["point"],
        }
        model = json_schema_to_pydantic(schema, "WithNested")
        instance = model(point={"x": 1.0, "y": 2.0})
        assert isinstance(instance.point, BaseModel)
        assert instance.model_dump() == {"point": {"x": 1.0, "y": 2.0}}

    def test_ref_and_defs(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Filter": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "op": {
                            "type": "string",
                            "enum": ["eq", "gt", "lt"],
                        },
                    },
                    "required": ["field", "op"],
                },
            },
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Filter"},
                },
            },
            "required": ["filters"],
        }
        model = json_schema_to_pydantic(schema, "WithRefs")
        instance = model(filters=[{"field": "age", "op": "gt"}])
        assert len(instance.filters) == 1
        assert instance.filters[0].field == "age"
        assert instance.filters[0].op == "gt"

    def test_default_value(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        model = json_schema_to_pydantic(schema, "WithDefault")
        instance = model(query="test")
        assert instance.limit == 10
        assert instance.query == "test"

    def test_field_descriptions_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                },
            },
            "required": ["query"],
        }
        model = json_schema_to_pydantic(schema, "WithDesc")
        fields = model.model_fields
        assert fields["query"].description == "The search query string"
        assert fields["limit"].description == "Max results to return"

    def test_const_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "type": {"const": "point"},
                "x": {"type": "number"},
            },
            "required": ["type", "x"],
        }
        model = json_schema_to_pydantic(schema, "WithConst")
        instance = model(type="point", x=1.0)
        assert instance.type == "point"
        assert instance.model_dump() == {"type": "point", "x": 1.0}

    def test_allof_merge(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "Extra": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"],
                },
            },
            "properties": {
                "person": {
                    "allOf": [
                        {"$ref": "#/$defs/Base"},
                        {"$ref": "#/$defs/Extra"},
                    ],
                },
            },
            "required": ["person"],
        }
        model = json_schema_to_pydantic(schema, "WithAllOf")
        instance = model(person={"name": "Alice", "age": 30})
        assert instance.person.name == "Alice"
        assert instance.person.age == 30

    def test_allof_inline(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                            "required": ["id"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                            },
                        },
                    ],
                },
            },
            "required": ["item"],
        }
        model = json_schema_to_pydantic(schema, "InlineAllOf")
        instance = model(item={"id": 1, "label": "test"})
        assert instance.item.id == 1
        assert instance.item.label == "test"

    def test_recursive_ref_does_not_crash(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/TreeNode"},
                        },
                    },
                    "required": ["value"],
                },
            },
            "properties": {
                "root": {"$ref": "#/$defs/TreeNode"},
            },
            "required": ["root"],
        }
        model = json_schema_to_pydantic(schema, "WithRecursion")
        # Should not crash — recursive ref degrades to dict[str, Any]
        instance = model(root={"value": "top"})
        assert instance.root.value == "top"

    def test_enum_unsafe_member_names(self) -> None:
        """Enum values like 'in-progress' or '123' become valid identifiers."""
        from enum import StrEnum

        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["in-progress", "done", "123"],
                },
            },
            "required": ["status"],
        }
        model = json_schema_to_pydantic(schema, "UnsafeEnum")
        instance = model(status="in-progress")
        assert isinstance(instance.status, StrEnum)
        assert instance.status == "in-progress"
        assert model(status="123").status == "123"

    def test_integer_enum(self) -> None:
        from enum import IntEnum

        schema = {
            "type": "object",
            "properties": {
                "priority": {"enum": [1, 2, 3]},
            },
            "required": ["priority"],
        }
        model = json_schema_to_pydantic(schema, "IntEnumModel")
        instance = model(priority=2)
        assert isinstance(instance.priority, IntEnum)
        assert instance.priority == 2

    def test_mixed_enum(self) -> None:
        from enum import Enum

        schema = {
            "type": "object",
            "properties": {
                "value": {"enum": ["auto", 1, True]},
            },
            "required": ["value"],
        }
        model = json_schema_to_pydantic(schema, "MixedEnumModel")
        instance = model(value="auto")
        assert isinstance(instance.value, Enum)

    def test_ref_cache_returns_same_type(self) -> None:
        """Repeated $ref to the same definition reuses the same class."""
        schema = {
            "type": "object",
            "$defs": {
                "Point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["x", "y"],
                },
            },
            "properties": {
                "start": {"$ref": "#/$defs/Point"},
                "end": {"$ref": "#/$defs/Point"},
            },
            "required": ["start", "end"],
        }
        model = json_schema_to_pydantic(schema, "TwoPoints")
        instance = model(
            start={"x": 0, "y": 0},
            end={"x": 1, "y": 1},
        )
        assert type(instance.start) is type(instance.end)

    def test_default_null(self) -> None:
        """'default': null is distinct from no default."""
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "default": None},
                "tag": {"type": "string"},
            },
            "required": [],
        }
        model = json_schema_to_pydantic(schema, "DefaultNull")
        instance = model()
        assert instance.label is None
        assert instance.tag is None

    def test_enum_member_collision(self) -> None:
        """Values that sanitize to the same name get disambiguated."""
        schema = {
            "type": "object",
            "properties": {
                "sep": {
                    "type": "string",
                    "enum": ["a-b", "a b"],
                },
            },
            "required": ["sep"],
        }
        model = json_schema_to_pydantic(schema, "Collision")
        # Both values must be representable
        assert model(sep="a-b").sep == "a-b"
        assert model(sep="a b").sep == "a b"

    def test_enum_keyword_member(self) -> None:
        """Python keywords in enum values get suffixed."""
        schema = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["class", "return", "normal"],
                },
            },
            "required": ["mode"],
        }
        model = json_schema_to_pydantic(schema, "KeywordEnum")
        assert model(mode="class").mode == "class"
        assert model(mode="return").mode == "return"


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
    async def test_tools_include_mcp_and_resource_tools(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = client.tools()
            mcp_tools = [t for t in tools if isinstance(t, MCPTool)]
            resource_tools = [
                t
                for t in tools
                if isinstance(t, (MCPListResourcesTool, MCPReadResourceTool))
            ]
            assert len(mcp_tools) >= 1
            assert len(resource_tools) == 2

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

    @pytest.mark.asyncio
    async def test_server_capabilities_exposed(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            caps = client.server_capabilities
            assert caps is not None
            assert caps.tools is not None
            assert caps.resources is not None
            assert caps.prompts is not None


# ---------- Resource tools ----------


class TestMCPResources:
    @pytest.mark.asyncio
    async def test_resource_tools_auto_generated(self) -> None:
        """Resource tools are added when server supports resources."""
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tool_names = {t.name for t in client.tools()}
            assert "test_list_resources" in tool_names
            assert "test_read_resource" in tool_names

    @pytest.mark.asyncio
    async def test_list_resources_returns_resources(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["test_list_resources"]()
            assert "docs://readme" in result
            assert "Project readme" in result

    @pytest.mark.asyncio
    async def test_list_resources_includes_templates(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["test_list_resources"]()
            assert "Resource Templates:" in result
            assert "{item_id}" in result

    @pytest.mark.asyncio
    async def test_read_static_resource(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["test_read_resource"](uri="docs://readme")
            assert "# Test Project" in result

    @pytest.mark.asyncio
    async def test_read_template_resource(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            tools = {t.name: t for t in client.tools()}
            result = await tools["test_read_resource"](uri="docs://items/42")
            assert '"id": "42"' in result


# ---------- Prompts (developer API) ----------


class TestMCPPrompts:
    @pytest.mark.asyncio
    async def test_list_prompts(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            result = await client.list_prompts()
            prompt_names = {p.name for p in result.prompts}
            assert "greet" in prompt_names
            assert "summarize" in prompt_names

    @pytest.mark.asyncio
    async def test_get_prompt(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            result = await client.get_prompt("greet", {"name": "Alice"})
            assert len(result.messages) >= 1
            text = result.messages[0].content.text
            assert "Alice" in text

    @pytest.mark.asyncio
    async def test_get_prompt_with_default_arg(self) -> None:
        async with MCPClient("test", server=_SERVER_CONFIG) as client:
            result = await client.get_prompt("summarize", {"text": "Hello world"})
            assert len(result.messages) >= 1
            text = result.messages[0].content.text
            assert "Hello world" in text
            assert "brief" in text

    @pytest.mark.asyncio
    async def test_prompts_not_connected_raises(self) -> None:
        client = MCPClient("test", server=_SERVER_CONFIG)
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.list_prompts()
