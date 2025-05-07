from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol

from pydantic import BaseModel

from .run_context import CtxT, RunContextWrapper
from .typing.io import AgentID, AgentPayload, OutT, StateT
from .typing.tool import BaseTool


class ParseOutputHandler(Protocol[OutT, CtxT]):
    def __call__(
        self, *args: Any, ctx: RunContextWrapper[CtxT] | None, **kwargs: Any
    ) -> OutT: ...


class BaseAgent(ABC, Generic[OutT, StateT, CtxT]):
    @abstractmethod
    def __init__(
        self,
        agent_id: AgentID,
        *,
        out_schema: type[OutT] = AgentPayload,
        **kwargs: Any,
    ) -> None:
        self._state: StateT
        self._agent_id = agent_id
        self._out_schema = out_schema
        self._parse_output_impl: ParseOutputHandler[OutT, CtxT] | None = None

    def parse_output_handler(
        self, func: ParseOutputHandler[OutT, CtxT]
    ) -> ParseOutputHandler[OutT, CtxT]:
        self._parse_output_impl = func

        return func

    def _parse_output(
        self, *args: Any, ctx: RunContextWrapper[CtxT] | None = None, **kwargs: Any
    ) -> OutT:
        if self._parse_output_impl:
            return self._parse_output_impl(*args, ctx=ctx, **kwargs)

        return self._out_schema()

    @property
    def agent_id(self) -> AgentID:
        return self._agent_id

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def out_schema(self) -> type[OutT]:
        return self._out_schema

    @abstractmethod
    async def run(
        self,
        inp_items: Any,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def as_tool(
        self, tool_name: str, tool_description: str, tool_strict: bool = True
    ) -> BaseTool[BaseModel, BaseModel, CtxT]:
        pass
