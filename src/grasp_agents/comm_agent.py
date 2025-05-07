import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeVar, cast, final

from pydantic import BaseModel

from .agent_message import AgentMessage
from .agent_message_pool import AgentMessagePool
from .base_agent import BaseAgent
from .run_context import CtxT, RunContextWrapper
from .typing.io import AgentID, AgentPayload, AgentState, InT, OutT, StateT
from .typing.tool import BaseTool

logger = logging.getLogger(__name__)

_EH_OutT = TypeVar("_EH_OutT", bound=AgentPayload, contravariant=True)  # noqa: PLC0105
_EH_StateT = TypeVar("_EH_StateT", bound=AgentState, contravariant=True)  # noqa: PLC0105


class ExitHandler(Protocol[_EH_OutT, _EH_StateT, CtxT]):
    def __call__(
        self,
        output_message: AgentMessage[_EH_OutT, _EH_StateT],
        agent_state: _EH_StateT,
        ctx: RunContextWrapper[CtxT] | None,
    ) -> bool: ...


class CommunicatingAgent(
    BaseAgent[OutT, StateT, CtxT], Generic[InT, OutT, StateT, CtxT]
):
    def __init__(
        self,
        agent_id: AgentID,
        *,
        out_schema: type[OutT] = AgentPayload,
        rcv_args_schema: type[InT] = AgentPayload,
        recipient_ids: Sequence[AgentID] | None = None,
        message_pool: AgentMessagePool[CtxT] | None = None,
        dynamic_routing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, out_schema=out_schema, **kwargs)
        self._message_pool = message_pool or AgentMessagePool()

        self._dynamic_routing = dynamic_routing

        self._is_listening = False
        self._exit_impl: ExitHandler[OutT, StateT, CtxT] | None = None

        self._rcv_args_schema = rcv_args_schema
        self.recipient_ids = recipient_ids or []

    @property
    def rcv_args_schema(self) -> type[InT]:  # type: ignore[reportInvalidTypeVarUse]
        return self._rcv_args_schema

    @property
    def dynamic_routing(self) -> bool:
        return self._dynamic_routing

    def _parse_output(
        self,
        *args: Any,
        rcv_args: InT | None = None,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> OutT:
        if self._parse_output_impl:
            return self._parse_output_impl(*args, rcv_args=rcv_args, ctx=ctx, **kwargs)

        return self._out_schema()

    def _validate_dynamic_routing(self, payloads: Sequence[OutT]) -> Sequence[AgentID]:
        assert all((p.selected_recipient_ids is not None) for p in payloads), (
            "Dynamic routing is enabled, but some payloads have no recipient IDs"
        )

        selected_recipient_ids_per_payload = [
            set(p.selected_recipient_ids or []) for p in payloads
        ]
        assert all(
            x == selected_recipient_ids_per_payload[0]
            for x in selected_recipient_ids_per_payload
        ), "All payloads must have the same recipient IDs for dynamic routing"

        assert payloads[0].selected_recipient_ids is not None
        selected_recipient_ids = payloads[0].selected_recipient_ids

        assert all(rid in self.recipient_ids for rid in selected_recipient_ids), (
            "Dynamic routing is enabled, but recipient IDs are not in "
            "the allowed agent's recipient IDs"
        )

        return selected_recipient_ids

    def _validate_static_routing(self, payloads: Sequence[OutT]) -> Sequence[AgentID]:
        assert all((p.selected_recipient_ids is None) for p in payloads), (
            "Dynamic routing is not enabled, but some payloads have recipient IDs"
        )

        return self.recipient_ids

    async def post_message(self, message: AgentMessage[OutT, StateT]) -> None:
        if self._dynamic_routing:
            self._validate_dynamic_routing(message.payloads)
        else:
            self._validate_static_routing(message.payloads)

        await self._message_pool.post(message)

    @abstractmethod
    async def run(
        self,
        inp_items: Any | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        rcv_message: AgentMessage[InT, StateT] | None = None,
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
            return self._exit_impl(
                output_message=output_message, agent_state=self.state, ctx=ctx
            )

        return False

    async def _message_handler(
        self,
        message: AgentMessage[AgentPayload, AgentState],
        ctx: RunContextWrapper[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        rcv_message = cast("AgentMessage[InT, StateT]", message)
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
        self, tool_name: str, tool_description: str, tool_strict: bool = True
    ) -> BaseTool[BaseModel, BaseModel, CtxT]:
        # assert self.state.batch_size == 1, (
        #     "Using agents as tools is only supported for batch size 1"
        # )

        agent_instance = self

        class AgentTool(BaseTool[BaseModel, BaseModel, Any]):
            name: str = tool_name
            description: str = tool_description
            in_schema: type[BaseModel] = agent_instance.rcv_args_schema
            out_schema: type[BaseModel] = agent_instance.out_schema

            strict: bool | None = tool_strict

            async def run(
                self,
                inp: BaseModel,
                ctx: RunContextWrapper[CtxT] | None = None,
            ) -> OutT:
                rcv_args = agent_instance.rcv_args_schema.model_validate(inp)
                rcv_message = AgentMessage(  # type: ignore[arg-type]
                    payloads=[rcv_args],
                    sender_id="<tool_user>",
                    recipient_ids=[agent_instance.agent_id],
                )

                agent_result = await agent_instance.run(
                    rcv_message=rcv_message,  # type: ignore[arg-type]
                    entry_point=False,
                    forbid_state_change=True,
                    ctx=ctx,
                )

                return agent_result.payloads[0]

        return AgentTool()
