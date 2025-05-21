from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic

from pydantic import TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .run_context import CtxT, RunContextWrapper
from .typing.io import AgentID, OutT, StateT
from .typing.tool import BaseTool


class BaseAgent(AutoInstanceAttributesMixin, ABC, Generic[OutT, StateT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_out_type"}

    @abstractmethod
    def __init__(self, agent_id: AgentID, **kwargs: Any) -> None:
        self._out_type: type[OutT]
        self._state: StateT

        super().__init__()

        self._agent_id = agent_id
        self._out_type_adapter: TypeAdapter[OutT] = TypeAdapter(self._out_type)

    @property
    def out_type(self) -> type[OutT]:
        return self._out_type

    @property
    def agent_id(self) -> AgentID:
        return self._agent_id

    @property
    def state(self) -> StateT:
        return self._state

    @abstractmethod
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def as_tool(
        self, tool_name: str, tool_description: str, tool_strict: bool = True
    ) -> BaseTool[Any, OutT, CtxT]:
        pass
