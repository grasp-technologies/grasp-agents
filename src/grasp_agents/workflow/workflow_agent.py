from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from ..agent_message_pool import AgentMessage, AgentMessagePool
from ..comm_agent import CommunicatingAgent
from ..run_context import CtxT, RunContextWrapper
from ..typing.io import AgentID, AgentState, InT, OutT


class WorkflowAgent(
    CommunicatingAgent[InT, OutT, AgentState, CtxT],
    ABC,
    Generic[InT, OutT, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        agent_id: AgentID,
        subagents: Sequence[CommunicatingAgent[Any, Any, AgentState, CtxT]],
        start_agent: CommunicatingAgent[InT, Any, AgentState, CtxT],
        end_agent: CommunicatingAgent[Any, OutT, AgentState, CtxT],
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
        dynamic_routing: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if not subagents:
            raise ValueError("At least one step is required")
        if start_agent not in subagents:
            raise ValueError("Start agent must be in the subagents list")
        if end_agent not in subagents:
            raise ValueError("End agent must be in the subagents list")

        self.subagents = subagents

        self._start_agent = start_agent
        self._end_agent = end_agent

        super().__init__(
            agent_id=agent_id,
            message_pool=message_pool,
            recipient_ids=recipient_ids,
            dynamic_routing=dynamic_routing,
        )
        for subagent in subagents:
            assert not subagent.recipient_ids, (
                "Subagents must not have recipient_ids set."
            )

    @property
    def start_agent(self) -> CommunicatingAgent[InT, Any, AgentState, CtxT]:
        return self._start_agent

    @property
    def end_agent(self) -> CommunicatingAgent[Any, OutT, AgentState, CtxT]:
        return self._end_agent

    @abstractmethod
    async def run(
        self,
        inp_items: Any | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        rcv_message: AgentMessage[InT, AgentState] | None = None,
        entry_point: bool = False,
        forbid_state_change: bool = False,
        **generation_kwargs: Any,
    ) -> AgentMessage[OutT, AgentState]:
        pass
