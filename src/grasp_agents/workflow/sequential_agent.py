from collections.abc import Sequence
from typing import Any, ClassVar, Generic, cast, final

from ..agent_message_pool import AgentMessage, AgentMessagePool
from ..comm_agent import CommunicatingAgent
from ..run_context import CtxT, RunContextWrapper
from ..typing.io import AgentID, InT, OutT
from .workflow_agent import WorkflowAgent


class SequentialWorkflowAgent(WorkflowAgent[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        agent_id: AgentID,
        subagents: Sequence[CommunicatingAgent[Any, Any, Any, CtxT]],
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
        dynamic_routing: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        super().__init__(
            subagents=subagents,
            start_agent=subagents[0],
            end_agent=subagents[-1],
            agent_id=agent_id,
            message_pool=message_pool,
            recipient_ids=recipient_ids,
            dynamic_routing=dynamic_routing,
        )

    @final
    async def run(
        self,
        inp_items: Any | None = None,
        *,
        rcv_message: AgentMessage[InT, Any] | None = None,
        ctx: RunContextWrapper[CtxT] | None = None,
        entry_point: bool = False,
        forbid_state_change: bool = False,
        **kwargs: Any,
    ) -> AgentMessage[OutT, Any]:
        agent_message = rcv_message
        for subagent in self.subagents:
            agent_message = await subagent.run(
                inp_items=inp_items,
                rcv_message=agent_message,
                entry_point=entry_point,
                forbid_state_change=forbid_state_change,
                ctx=ctx,
                **kwargs,
            )
            inp_items = None
            entry_point = False

        return cast("AgentMessage[OutT, Any]", agent_message)
