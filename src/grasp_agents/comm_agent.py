import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final

from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import SkipJsonSchema

from .agent_message import AgentMessage
from .agent_message_pool import AgentMessagePool
from .base_agent import BaseAgent
from .run_context import CtxT, RunContextWrapper
from .typing.io import AgentID, AgentState, InT, OutT, StateT
from .typing.tool import BaseTool

logger = logging.getLogger(__name__)


class DynCommPayload(BaseModel):
    selected_recipient_ids: SkipJsonSchema[Sequence[AgentID]]


_EH_OutT = TypeVar("_EH_OutT", contravariant=True)  # noqa: PLC0105
_EH_StateT = TypeVar("_EH_StateT", bound=AgentState, contravariant=True)  # noqa: PLC0105


class ExitHandler(Protocol[_EH_OutT, _EH_StateT, CtxT]):
    def __call__(
        self,
        output_message: AgentMessage[_EH_OutT, _EH_StateT],
        ctx: RunContextWrapper[CtxT] | None,
    ) -> bool: ...


class CommunicatingAgent(
    BaseAgent[OutT, StateT, CtxT], Generic[InT, OutT, StateT, CtxT]
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        agent_id: AgentID,
        *,
        recipient_ids: Sequence[AgentID] | None = None,
        message_pool: AgentMessagePool[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        self._in_type: type[InT]
        super().__init__(agent_id=agent_id, **kwargs)

        self._rcv_args_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)
        self.recipient_ids = recipient_ids or []

        self._message_pool = message_pool or AgentMessagePool()
        self._is_listening = False
        self._exit_impl: ExitHandler[OutT, StateT, CtxT] | None = None

    @property
    def in_type(self) -> type[InT]:  # type: ignore
        # Exposing the type of a contravariant variable only, should be safe
        return self._in_type

    def _validate_routing(self, payloads: Sequence[OutT]) -> Sequence[AgentID]:
        if all(isinstance(p, DynCommPayload) for p in payloads):
            payloads_ = cast("Sequence[DynCommPayload]", payloads)
            selected_recipient_ids_per_payload = [
                set(p.selected_recipient_ids or []) for p in payloads_
            ]
            assert all(
                x == selected_recipient_ids_per_payload[0]
                for x in selected_recipient_ids_per_payload
            ), "All payloads must have the same recipient IDs for dynamic routing"

            assert payloads_[0].selected_recipient_ids is not None
            selected_recipient_ids = payloads_[0].selected_recipient_ids

            assert all(rid in self.recipient_ids for rid in selected_recipient_ids), (
                "Dynamic routing is enabled, but recipient IDs are not in "
                "the allowed agent's recipient IDs"
            )

            return selected_recipient_ids

        if all((not isinstance(p, DynCommPayload)) for p in payloads):
            return self.recipient_ids

        raise ValueError(
            "All payloads must be either DCommAgentPayload or not DCommAgentPayload"
        )

    async def post_message(self, message: AgentMessage[OutT, StateT]) -> None:
        self._validate_routing(message.payloads)

        await self._message_pool.post(message)

    @abstractmethod
    async def run(
        self,
        inp_items: Any | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        rcv_message: AgentMessage[InT, AgentState] | None = None,
        entry_point: bool = False,
        forbid_state_change: bool = False,
        **kwargs: Any,
    ) -> AgentMessage[OutT, StateT]:
        pass

    async def run_and_post(
        self, ctx: RunContextWrapper[CtxT] | None = None, **run_kwargs: Any
    ) -> None:
        output_message = await self.run(
            ctx=ctx, rcv_message=None, entry_point=True, **run_kwargs
        )
        await self.post_message(output_message)

    def exit_handler(
        self, func: ExitHandler[OutT, StateT, CtxT]
    ) -> ExitHandler[OutT, StateT, CtxT]:
        self._exit_impl = func

        return func

    def _exit_condition(
        self,
        output_message: AgentMessage[OutT, StateT],
        ctx: RunContextWrapper[CtxT] | None,
    ) -> bool:
        if self._exit_impl:
            return self._exit_impl(output_message=output_message, ctx=ctx)

        return False

    async def _message_handler(
        self,
        message: AgentMessage[Any, AgentState],
        ctx: RunContextWrapper[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        rcv_message = cast("AgentMessage[InT, AgentState]", message)
        out_message = await self.run(ctx=ctx, rcv_message=rcv_message, **run_kwargs)

        if self._exit_condition(output_message=out_message, ctx=ctx):
            await self._message_pool.stop_all()
            return

        if self.recipient_ids:
            await self.post_message(out_message)

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    async def start_listening(
        self, ctx: RunContextWrapper[CtxT] | None = None, **run_kwargs: Any
    ) -> None:
        if self._is_listening:
            return

        self._is_listening = True
        self._message_pool.register_message_handler(
            agent_id=self.agent_id,
            handler=self._message_handler,
            ctx=ctx,
            **run_kwargs,
        )

    async def stop_listening(self) -> None:
        self._is_listening = False
        await self._message_pool.unregister_message_handler(self.agent_id)

    @final
    def as_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_strict: bool = True,
    ) -> BaseTool[InT, OutT, Any]:  # type: ignore[override]
        # Will check if InT is a BaseModel at runtime
        agent_instance = self
        in_type = agent_instance.in_type
        out_type = agent_instance.out_type
        if not issubclass(in_type, BaseModel):
            raise TypeError(
                "Cannot create a tool from an agent with "
                f"non-BaseModel input type: {in_type}"
            )

        class AgentTool(BaseTool[in_type, out_type, Any]):
            name: str = tool_name
            description: str = tool_description
            strict: bool | None = tool_strict

            async def run(
                self,
                inp: InT,
                ctx: RunContextWrapper[CtxT] | None = None,
            ) -> OutT:
                rcv_args = in_type.model_validate(inp)
                rcv_message = AgentMessage[in_type, AgentState](
                    payloads=[rcv_args],
                    sender_id="<tool_user>",
                    recipient_ids=[agent_instance.agent_id],
                )
                agent_result = await agent_instance.run(
                    rcv_message=rcv_message,
                    entry_point=False,
                    forbid_state_change=True,
                    ctx=ctx,
                )

                return agent_result.payloads[0]

        return AgentTool()  # type: ignore[return-value]
