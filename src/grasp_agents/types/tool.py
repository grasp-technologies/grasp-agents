import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter

from grasp_agents.generics_utils import AutoInstanceAttributesMixin
from grasp_agents.run_context import CtxT, RunContext
from grasp_agents.tracing_decorators import tool

from .events import Event, ToolErrorEvent, ToolErrorInfo

logger = logging.getLogger(__name__)

_InT = TypeVar("_InT", bound=BaseModel)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class NamedToolChoice(BaseModel):
    name: str


# TODO: expand to support more options
ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


@runtime_checkable
class ToolProgressCallback(Protocol):
    """Protocol for reporting tool execution progress."""

    async def __call__(
        self, progress: float, total: float | None, message: str | None
    ) -> None: ...


@tool(name="tool", method_name="__call__")  # type: ignore
class BaseTool(
    AutoInstanceAttributesMixin,
    ABC,
    Generic[_InT, _OutT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        *,
        name: str,
        description: str,
        timeout: float | None = None,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        self._in_type: type[_InT]
        self._out_type: type[_OutT_co]

        super().__init__()

        self.name = name
        self.description = description
        self.timeout = timeout
        self.tracing_exclude_input_fields = tracing_exclude_input_fields

    @property
    def in_type(self) -> type[_InT]:
        return self._in_type

    @property
    def out_type(self) -> type[_OutT_co]:
        return self._out_type

    @abstractmethod
    async def _run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> _OutT_co:
        pass

    async def _run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> AsyncIterator[Event[Any]]:
        from .events import ToolOutputEvent  # avoid circular import

        out = await self._run(
            inp, ctx=ctx, call_id=call_id, progress_callback=progress_callback
        )
        yield ToolOutputEvent(data=out, src_name=self.name, call_id=call_id)

    def _on_error_impl(self, error: Exception) -> ToolErrorInfo:
        logger.warning("Tool '%s' failed: %s", self.name, error)

        return ToolErrorInfo(tool_name=self.name, error=str(error), timed_out=False)

    def _on_error(self, error: Exception) -> ToolErrorInfo:
        if isinstance(error, asyncio.TimeoutError):
            logger.warning("Tool '%s' timed out after %ss", self.name, self.timeout)
            return ToolErrorInfo(
                tool_name=self.name,
                error=f"Timed out after {self.timeout}s",
                timed_out=True,
            )

        return self._on_error_impl(error)

    async def _run_with_timeout(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> _OutT_co | ToolErrorInfo:
        input_args = TypeAdapter(self.in_type).validate_python(inp)
        try:
            coro = self._run(
                input_args,
                ctx=ctx,
                call_id=call_id,
                progress_callback=progress_callback,
            )
            if self.timeout is not None:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
            else:
                result = await coro
            return TypeAdapter(self.out_type).validate_python(result)
        except Exception as e:
            return self._on_error(e)

    async def _run_stream_with_timeout(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> AsyncIterator[Event[Any]]:
        input_args = TypeAdapter(self.in_type).validate_python(inp)
        try:
            stream = self._run_stream(
                input_args,
                ctx=ctx,
                call_id=call_id,
                progress_callback=progress_callback,
            )
            if self.timeout is not None:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            anext(stream), timeout=self.timeout
                        )
                    except StopAsyncIteration:
                        break
                    yield event
            else:
                async for event in stream:
                    yield event
        except Exception as e:
            error_data = self._on_error(e)
            yield ToolErrorEvent(data=error_data, src_name=self.name, call_id=call_id)

    async def __call__(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        **kwargs: Any,
    ) -> _OutT_co | ToolErrorInfo:
        return await self._run_with_timeout(
            kwargs,  # type: ignore[arg-type]
            ctx=ctx,
            call_id=call_id,
            progress_callback=progress_callback,
        )

    async def run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> _OutT_co | ToolErrorInfo:
        return await self._run_with_timeout(
            inp, ctx=ctx, call_id=call_id, progress_callback=progress_callback
        )

    async def run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        call_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in self._run_stream_with_timeout(
            inp, ctx=ctx, call_id=call_id, progress_callback=progress_callback
        ):
            yield event
