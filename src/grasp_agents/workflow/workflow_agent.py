from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic

from ..agent_message_pool import AgentMessage, AgentMessagePool
from ..comm_agent import CommunicatingAgent
from ..run_context import CtxT, RunContextWrapper
from ..typing.io import AgentID, AgentPayload, AgentState, InT, OutT


class WorkflowAgent(
    CommunicatingAgent[InT, OutT, AgentState, CtxT],
    ABC,
    Generic[InT, OutT, CtxT],
):
    def __init__(
        self,
        agent_id: AgentID,
        subagents: Sequence[
            CommunicatingAgent[AgentPayload, AgentPayload, AgentState, CtxT]
        ],
        start_agent: CommunicatingAgent[InT, AgentPayload, AgentState, CtxT],
        end_agent: CommunicatingAgent[AgentPayload, OutT, AgentState, CtxT],
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
        dynamic_routing: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if not subagents:
            raise ValueError("At least one step is required")

        self.subagents = subagents

        self._start_agent = start_agent
        self._end_agent = end_agent

        super().__init__(
            agent_id=agent_id,
            out_schema=end_agent.out_schema,
            rcv_args_schema=start_agent.rcv_args_schema,
            message_pool=message_pool,
            recipient_ids=recipient_ids,
            dynamic_routing=dynamic_routing,
        )
        for subagent in subagents:
            assert not subagent.recipient_ids, (
                "Subagents must not have recipient_ids set."
            )

    @property
    def start_agent(self) -> CommunicatingAgent[InT, AgentPayload, AgentState, CtxT]:
        return self._start_agent

    @property
    def end_agent(self) -> CommunicatingAgent[AgentPayload, OutT, AgentState, CtxT]:
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
