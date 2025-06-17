from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, PrivateAttr, TypeAdapter

from ..generics_utils import AutoInstanceAttributesMixin

if TYPE_CHECKING:
    from ..run_context import CtxT, RunContext
else:
    CtxT = TypeVar("CtxT")

    class RunContext(Generic[CtxT]):
        """Runtime placeholder so RunContext[CtxT] works"""


_InT_contra = TypeVar("_InT_contra", bound=BaseModel, contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class ToolCall(BaseModel):
    id: str
    tool_name: str
    tool_arguments: str


class BaseTool(
    AutoInstanceAttributesMixin,
    BaseModel,
    ABC,
    Generic[_InT_contra, _OutT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    name: str
    description: str

    _in_type: type[_InT_contra] = PrivateAttr()
    _out_type: type[_OutT_co] = PrivateAttr()

    # _in_type_adapter: TypeAdapter[_InT_contra] = PrivateAttr()
    # _out_type_adapter: TypeAdapter[_OutT_co] = PrivateAttr()

    # def model_post_init(self, context: Any) -> None:
    #     self._in_type_adapter = TypeAdapter(self._in_type)
    #     self._out_type_adapter = TypeAdapter(self._out_type)

    @property
    def in_type(self) -> type[_InT_contra]:  # type: ignore[reportInvalidTypeVarUse]
        # Exposing the type of a contravariant variable only, should be type safe
        return self._in_type

    @property
    def out_type(self) -> type[_OutT_co]:
        return self._out_type

    # @property
    # def in_type_adapter(self) -> TypeAdapter[_InT_contra]:
    #     return self._in_type_adapter

    # @property
    # def out_type_adapter(self) -> TypeAdapter[_OutT_co]:
    #     return self._out_type_adapter

    @abstractmethod
    async def run(
        self, inp: _InT_contra, ctx: RunContext[CtxT] | None = None
    ) -> _OutT_co:
        pass

    async def __call__(
        self, ctx: RunContext[CtxT] | None = None, **kwargs: Any
    ) -> _OutT_co:
        input_args = TypeAdapter(self._in_type).validate_python(kwargs)
        output = await self.run(input_args, ctx=ctx)

        return TypeAdapter(self._out_type).validate_python(output)


class NamedToolChoice(BaseModel):
    name: str


ToolChoice: TypeAlias = Literal["none", "auto", "required"] | NamedToolChoice
