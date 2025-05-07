from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..run_context import CtxT, RunContextWrapper
else:
    CtxT = TypeVar("CtxT")

    class RunContextWrapper(Generic[CtxT]):
        """Runtime placeholder so RunContextWrapper[CtxT] works"""


ToolInT = TypeVar("ToolInT", bound=BaseModel, contravariant=True)  # noqa: PLC0105
ToolOutT = TypeVar("ToolOutT", bound=BaseModel, covariant=True)  # noqa: PLC0105


class ToolCall(BaseModel):
    id: str
    tool_name: str
    tool_arguments: str


class BaseTool(BaseModel, ABC, Generic[ToolInT, ToolOutT, CtxT]):
    name: str
    description: str
    in_schema: type[ToolInT]
    out_schema: type[ToolOutT]

    # Supported by OpenAI API
    strict: bool | None = None

    @abstractmethod
    async def run(
        self, inp: ToolInT, ctx: RunContextWrapper[CtxT] | None = None
    ) -> ToolOutT:
        pass

    async def __call__(
        self, ctx: RunContextWrapper[CtxT] | None = None, **kwargs: Any
    ) -> ToolOutT:
        result = await self.run(self.in_schema(**kwargs), ctx=ctx)

        return self.out_schema.model_validate(result)


ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | BaseTool[BaseModel, BaseModel, Any]
)
