"""Tests for the @function_tool decorator."""

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.function_tool import FunctionTool, function_tool
from grasp_agents.run_context import RunContext
from grasp_agents.types.events import ToolErrorInfo
from grasp_agents.types.tool import BaseTool


# ---------- Basic usage ----------


class TestFunctionToolBasic:
    def test_bare_decorator_async(self) -> None:
        @function_tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert isinstance(add, FunctionTool)
        assert isinstance(add, BaseTool)
        assert add.name == "add"
        assert add.description == "Add two numbers."

    def test_bare_decorator_sync(self) -> None:
        @function_tool
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        assert isinstance(multiply, FunctionTool)
        assert multiply.name == "multiply"

    def test_decorator_with_options(self) -> None:
        @function_tool(name="calculator", description="Sums values", timeout=5.0)
        async def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "calculator"
        assert add.description == "Sums values"
        assert add.timeout == 5.0

    def test_input_model_from_annotations(self) -> None:
        @function_tool
        async def greet(name: str, excited: bool = False) -> str:
            """Greet someone."""
            return f"Hello, {name}{'!' if excited else '.'}"

        model = greet.in_type
        assert issubclass(model, BaseModel)

        # Required field
        fields = model.model_fields
        assert "name" in fields
        assert fields["name"].is_required()

        # Optional field with default
        assert "excited" in fields
        assert not fields["excited"].is_required()
        assert fields["excited"].default is False

    def test_no_docstring_uses_empty_description(self) -> None:
        @function_tool
        async def silent(x: int) -> int:
            return x

        assert silent.description == ""


# ---------- Execution ----------


class TestFunctionToolExecution:
    @pytest.mark.asyncio
    async def test_call_async_function(self) -> None:
        @function_tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add(a=3, b=4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_call_sync_function(self) -> None:
        @function_tool
        def multiply(x: int, y: int) -> int:
            """Multiply."""
            return x * y

        result = await multiply(x=3, y=4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_default_values(self) -> None:
        @function_tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet."""
            return f"{greeting}, {name}!"

        result = await greet(name="World")
        assert result == "Hello, World!"

        result2 = await greet(name="World", greeting="Hi")
        assert result2 == "Hi, World!"

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        @function_tool
        async def failing(x: int) -> int:
            """Fails."""
            msg = "boom"
            raise ValueError(msg)

        result = await failing(x=1)
        assert isinstance(result, ToolErrorInfo)
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        import asyncio

        @function_tool(timeout=0.01)
        async def slow(x: int) -> int:
            """Slow."""
            await asyncio.sleep(10)
            return x

        result = await slow(x=1)
        assert isinstance(result, ToolErrorInfo)
        assert result.timed_out is True
        assert "Timed out" in result.error


# ---------- Context and call_id passthrough ----------


class TestFunctionToolContext:
    @pytest.mark.asyncio
    async def test_ctx_passthrough(self) -> None:
        @function_tool
        async def check_ctx(x: int, *, ctx: RunContext[Any] | None = None) -> str:
            """Check context."""
            return "has_ctx" if ctx is not None else "no_ctx"

        # ctx should NOT be in the input model
        assert "ctx" not in check_ctx.in_type.model_fields

        # Without ctx
        result = await check_ctx(x=1)
        assert result == "no_ctx"

    @pytest.mark.asyncio
    async def test_call_id_passthrough(self) -> None:
        @function_tool
        async def check_id(x: int, *, call_id: str | None = None) -> str:
            """Check call_id."""
            return call_id or "none"

        assert "call_id" not in check_id.in_type.model_fields

        result = await check_id(x=1, call_id="abc")
        assert result == "abc"

    @pytest.mark.asyncio
    async def test_ctx_and_call_id_both(self) -> None:
        @function_tool
        async def both(
            x: int, *, ctx: RunContext[Any] | None = None, call_id: str | None = None
        ) -> str:
            """Both special params."""
            parts = []
            if ctx is not None:
                parts.append("ctx")
            if call_id is not None:
                parts.append(call_id)
            return ",".join(parts) or "empty"

        fields = both.in_type.model_fields
        assert "x" in fields
        assert "ctx" not in fields
        assert "call_id" not in fields


# ---------- Schema generation ----------


class TestFunctionToolSchema:
    def test_json_schema_matches_annotations(self) -> None:
        @function_tool
        async def search(query: str, max_results: int = 10) -> str:
            """Search for something."""
            return query

        schema = search.in_type.model_json_schema()
        props = schema["properties"]
        assert "query" in props
        assert "max_results" in props
        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"
        assert props["max_results"]["default"] == 10
        assert schema["required"] == ["query"]
