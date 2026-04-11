import asyncio
import copy as copy_mod
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter

from grasp_agents.run_context import CtxT, RunContext
from grasp_agents.telemetry import SpanKind, traced
from grasp_agents.utils.generics import AutoInstanceAttributesMixin

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


@traced(name="tool", method_name="__call__", span_kind=SpanKind.TOOL)
class BaseTool(
    AutoInstanceAttributesMixin,
    ABC,
    Generic[_InT, _OutT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    name: str = ""
    description: str = ""

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        background: bool = False,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        self._in_type: type[_InT]
        self._out_type: type[_OutT_co]

        super().__init__()

        if name is not None:
            self.name = name

        if description is not None:
            self.description = description

        if not self.name:
            raise ValueError(f"{type(self).__name__} must have a non-empty name")

        self.timeout = timeout
        self.background = background
        self.tracing_exclude_input_fields = tracing_exclude_input_fields
        self._llm_in_type: type[BaseModel] | None = None

    @property
    def in_type(self) -> type[_InT]:
        return self._in_type

    @property
    def out_type(self) -> type[_OutT_co]:
        return self._out_type

    @property
    def llm_in_type(self) -> type[BaseModel]:
        """Schema the LLM sees for tool calls. Defaults to ``in_type``."""
        return self._llm_in_type or self._in_type

    @llm_in_type.setter
    def llm_in_type(self, value: type[BaseModel]) -> None:
        self._llm_in_type = value

    # --- Internal execution (implemented by subclasses) ---

    @abstractmethod
    async def _run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> _OutT_co:
        pass

    async def _run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        from .events import ToolOutputEvent  # avoid circular import

        out = await self._run(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            session_id=session_id,
        )
        yield ToolOutputEvent(data=out, source=self.name, exec_id=exec_id)

    # --- Error handling ---

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

    # --- Timeout wrappers ---

    async def _stream_with_timeout(
        self,
        stream: AsyncIterator[Event[Any]],
        *,
        exec_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        """Yield events from *stream*, applying timeout and error handling."""
        try:
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
            yield ToolErrorEvent(data=error_data, source=self.name, exec_id=exec_id)

    async def _run_with_timeout(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> _OutT_co | ToolErrorInfo:
        try:
            coro = self._run(
                inp,
                ctx=ctx,
                exec_id=exec_id,
                progress_callback=progress_callback,
                session_id=session_id,
            )
            if self.timeout is not None:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
            else:
                result = await coro
            return result  # type: ignore[return-value]
        except Exception as e:
            return self._on_error(e)

    async def _run_stream_with_timeout(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        stream = self._run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            session_id=session_id,
        )
        async for event in self._stream_with_timeout(stream, exec_id=exec_id):
            yield event

    # --- Public API ---

    async def __call__(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        **kwargs: Any,
    ) -> _OutT_co | ToolErrorInfo:
        inp = TypeAdapter(self.in_type).validate_python(kwargs)
        return await self._run_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
        )

    async def run(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> _OutT_co | ToolErrorInfo:
        return await self._run_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            session_id=session_id,
        )

    async def run_stream(
        self,
        inp: _InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in self._run_stream_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            session_id=session_id,
        ):
            yield event

    # --- Session persistence (overridden by resumable tools) ---

    @property
    def resumable(self) -> bool:
        return False

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        """Resume from a session checkpoint. Override in resumable tools."""
        raise NotImplementedError(f"{type(self).__name__} does not support resume")
        yield  # type: ignore[unreachable]  # makes this an async generator

    # --- Copy ---

    # Attributes that should be shared (not deepcopied) across copies.
    # Subclasses add entries for shared resources like network sessions.
    _copy_shared_attrs: ClassVar[frozenset[str]] = frozenset()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        for attr in self._copy_shared_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                memo[id(val)] = val
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in copy_mod.deepcopy(dict(self.__dict__), memo).items():
            object.__setattr__(new, k, v)
        return new

    def copy(self) -> Self:
        """
        Deep copy with shared attributes preserved by reference.

        Attributes listed in ``_copy_shared_attrs`` are kept as-is
        (via ``__deepcopy__``); everything else is deep-copied.
        """
        return copy_mod.deepcopy(self)
