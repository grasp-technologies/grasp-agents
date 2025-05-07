from collections.abc import Sequence
from logging import getLogger
from typing import Any, Generic, Protocol, TypeVar, cast, final

from ..agent_message_pool import AgentMessage, AgentMessagePool
from ..comm_agent import CommunicatingAgent
from ..run_context import CtxT, RunContextWrapper
from ..typing.io import AgentID, AgentPayload, AgentState, InT, OutT
from .workflow_agent import WorkflowAgent

logger = getLogger(__name__)

_EH_OutT = TypeVar("_EH_OutT", bound=AgentPayload, contravariant=True)  # noqa: PLC0105


class WorkflowLoopExitHandler(Protocol[_EH_OutT, CtxT]):
    def __call__(
        self,
        output_message: AgentMessage[_EH_OutT, AgentState],
        ctx: RunContextWrapper[CtxT] | None,
        **kwargs: Any,
    ) -> bool: ...


class LoopedWorkflowAgent(WorkflowAgent[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    def __init__(
        self,
        agent_id: AgentID,
        subagents: Sequence[
            CommunicatingAgent[AgentPayload, AgentPayload, AgentState, CtxT]
        ],
        exit_agent: CommunicatingAgent[AgentPayload, OutT, AgentState, CtxT],
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
        dynamic_routing: bool = False,
        max_iterations: int = 10,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        super().__init__(
            subagents=subagents,
            agent_id=agent_id,
            start_agent=subagents[0],
            end_agent=exit_agent,
            message_pool=message_pool,
            recipient_ids=recipient_ids,
            dynamic_routing=dynamic_routing,
        )

        self._max_iterations = max_iterations

        self._workflow_loop_exit_impl: WorkflowLoopExitHandler[OutT, CtxT] | None = None

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def workflow_loop_exit_handler(
        self, func: WorkflowLoopExitHandler[OutT, CtxT]
    ) -> WorkflowLoopExitHandler[OutT, CtxT]:
        self._workflow_loop_exit_impl = func

        return func

    def _workflow_loop_exit(
        self,
        output_message: AgentMessage[OutT, AgentState],
        ctx: RunContextWrapper[CtxT] | None,
        **kwargs: Any,
    ) -> bool:
        if self._workflow_loop_exit_impl:
            return self._workflow_loop_exit_impl(output_message, ctx=ctx, **kwargs)

        return False

    @final
    async def run(
        self,
        inp_items: Any | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        rcv_message: AgentMessage[InT, AgentState] | None = None,
        entry_point: bool = False,
        forbid_state_change: bool = False,
        **kwargs: Any,
    ) -> AgentMessage[OutT, AgentState]:
        agent_message = rcv_message
        num_iterations = 0
        exit_message: AgentMessage[OutT, AgentState] | None = None

        while True:
            for subagent in self.subagents:
                agent_message = await subagent.run(
                    inp_items=inp_items,
                    rcv_message=agent_message,
                    entry_point=entry_point,
                    forbid_state_change=forbid_state_change,
                    ctx=ctx,
                    **kwargs,
                )

                if subagent is self._end_agent:
                    num_iterations += 1
                    exit_message = cast("AgentMessage[OutT, AgentState]", agent_message)
                    if self._workflow_loop_exit(exit_message, ctx=ctx):
                        return exit_message
                    if num_iterations >= self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). Exiting loop."
                        )
                        return exit_message

                inp_items = None
                entry_point = False
