"""
Decorator to create a BaseTool from a plain function.

Usage::

    @function_tool
    async def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    # With options:
    @function_tool(name="calculator", timeout=10.0)
    async def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    # With RunContext access:
    @function_tool
    async def greet(name: str, *, ctx: RunContext[MyState]) -> str:
        \"\"\"Greet by name.\"\"\"
        ctx.state.greeted = True
        return f"Hello, {name}!"
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, overload

from pydantic import BaseModel, create_model

from .run_context import RunContext
from .types.tool import BaseTool, ToolProgressCallback

# Parameters with these names are passed through from the executor,
# not included in the tool's input schema.
_SPECIAL_PARAMS = {"ctx", "call_id"}


def _build_input_model(
    fn: Any,
    hints: dict[str, Any],
    sig: inspect.Signature,
) -> type[BaseModel]:
    """Build a Pydantic model from function parameters."""
    field_definitions: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in _SPECIAL_PARAMS:
            continue

        annotation = hints.get(param_name, Any)

        if param.default is not inspect.Parameter.empty:
            field_definitions[param_name] = (annotation, param.default)
        else:
            field_definitions[param_name] = (annotation, ...)

    model_name = f"{fn.__name__}_input"
    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]


def _has_special_param(sig: inspect.Signature, name: str) -> bool:
    return name in sig.parameters


class FunctionTool(BaseTool[BaseModel, Any, Any]):
    """A tool created from a plain function via @function_tool."""

    def __init__(
        self,
        *,
        fn: Any,
        name: str,
        description: str,
        input_model: type[BaseModel],
        is_async: bool,
        has_ctx: bool,
        has_call_id: bool,
        timeout: float | None = None,
    ) -> None:
        super().__init__(name=name, description=description, timeout=timeout)
        self._fn = fn
        self._resolved_in_type = input_model
        self._is_async = is_async
        self._has_ctx = has_ctx
        self._has_call_id = has_call_id

    @property
    def in_type(self) -> type[BaseModel]:
        return self._resolved_in_type

    async def _run(
        self,
        inp: BaseModel,
        *,
        ctx: RunContext[Any] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> Any:
        kwargs = inp.model_dump()
        if self._has_ctx:
            kwargs["ctx"] = ctx
        if self._has_call_id:
            kwargs["call_id"] = call_id

        if self._is_async:
            return await self._fn(**kwargs)
        return await asyncio.to_thread(self._fn, **kwargs)


@overload
def function_tool(fn: Any, /) -> FunctionTool: ...


@overload
def function_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Any: ...


def function_tool(
    fn: Any | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Any:
    """
    Create a BaseTool from a function.

    Can be used as a bare decorator or with keyword arguments::

        @function_tool
        async def add(a: int, b: int) -> int: ...

        @function_tool(name="calculator", timeout=5.0)
        async def add(a: int, b: int) -> int: ...
    """

    def _wrap(f: Any) -> FunctionTool:
        sig = inspect.signature(f)
        hints = _get_type_hints_safe(f)
        input_model = _build_input_model(f, hints, sig)

        tool_name = name or f.__name__
        tool_description = description or inspect.getdoc(f) or ""

        return FunctionTool(
            fn=f,
            name=tool_name,
            description=tool_description,
            input_model=input_model,
            is_async=asyncio.iscoroutinefunction(f),
            has_ctx=_has_special_param(sig, "ctx"),
            has_call_id=_has_special_param(sig, "call_id"),
            timeout=timeout,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _get_type_hints_safe(fn: Any) -> dict[str, Any]:
    """Get type hints, handling forward references gracefully."""
    try:
        return {k: v for k, v in fn.__annotations__.items() if k != "return"}
    except Exception:
        return {}
