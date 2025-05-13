from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from ..run_context import CtxT, RunContextWrapper
else:
    CtxT = TypeVar("CtxT")

    class RunContextWrapper(Generic[CtxT]):
        """Runtime placeholder so RunContextWrapper[CtxT] works"""


_ToolInT = TypeVar("_ToolInT", bound=BaseModel, contravariant=True)  # noqa: PLC0105
_ToolOutT = TypeVar("_ToolOutT", covariant=True)  # noqa: PLC0105


class ToolCall(BaseModel):
    id: str
    tool_name: str
    tool_arguments: str


class BaseTool(BaseModel, ABC, Generic[_ToolInT, _ToolOutT, CtxT]):
    name: str
    description: str
    in_schema: type[_ToolInT]
    out_schema: type[_ToolOutT]

    # Supported by OpenAI API
    strict: bool | None = None

    @abstractmethod
    async def run(
        self, inp: _ToolInT, ctx: RunContextWrapper[CtxT] | None = None
    ) -> _ToolOutT:
        pass

    async def run_batch(
        self, inp_batch: Sequence[_ToolInT], ctx: RunContextWrapper[CtxT] | None = None
    ) -> Sequence[_ToolOutT]:
        return await asyncio.gather(*[self.run(inp, ctx=ctx) for inp in inp_batch])

    async def __call__(
        self, ctx: RunContextWrapper[CtxT] | None = None, **kwargs: Any
    ) -> _ToolOutT:
        result = await self.run(self.in_schema(**kwargs), ctx=ctx)

        return TypeAdapter(self.out_schema).validate_python(result)


ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | BaseTool[BaseModel, Any, Any]
)
