from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, PrivateAttr, TypeAdapter

from ..generics_utils import AutoInstanceAttributesMixin

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


class BaseTool(
    AutoInstanceAttributesMixin, BaseModel, ABC, Generic[_ToolInT, _ToolOutT, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_schema",
        1: "_out_schema",
    }

    name: str
    description: str

    _in_schema: type[_ToolInT] = PrivateAttr()
    _out_schema: type[_ToolOutT] = PrivateAttr()

    # Supported by OpenAI API
    strict: bool | None = None

    @property
    def in_schema(self) -> type[_ToolInT]:  # type: ignore[reportInvalidTypeVarUse]
        # Exposing the type of a contravariant variable only, should be type safe
        return self._in_schema

    @property
    def out_schema(self) -> type[_ToolOutT]:
        return self._out_schema

    @abstractmethod
    async def run(
        self, inp: _ToolInT, ctx: RunContextWrapper[CtxT] | None = None
    ) -> _ToolOutT:
        pass

    async def __call__(
        self, ctx: RunContextWrapper[CtxT] | None = None, **kwargs: Any
    ) -> _ToolOutT:
        result = await self.run(self._in_schema(**kwargs), ctx=ctx)

        return TypeAdapter(self._out_schema).validate_python(result)


ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | BaseTool[BaseModel, Any, Any]
)
