from collections.abc import Sequence
from logging import getLogger
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final

from ..agent_message_pool import AgentMessage, AgentMessagePool
from ..comm_agent import CommunicatingAgent
from ..run_context import CtxT, RunContextWrapper
from ..typing.io import AgentID, AgentState, InT, OutT
from .workflow_agent import WorkflowAgent

logger = getLogger(__name__)

_EH_OutT = TypeVar("_EH_OutT", contravariant=True)  # noqa: PLC0105


class ExitWorkflowLoopHandler(Protocol[_EH_OutT, CtxT]):
    def __call__(
        self,
        output_message: AgentMessage[_EH_OutT, Any],
        ctx: RunContextWrapper[CtxT] | None,
        **kwargs: Any,
    ) -> bool: ...


class LoopedWorkflowAgent(WorkflowAgent[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        agent_id: AgentID,
        subagents: Sequence[CommunicatingAgent[Any, Any, Any, CtxT]],
        exit_agent: CommunicatingAgent[Any, OutT, Any, CtxT],
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

        self._exit_workflow_loop_impl: ExitWorkflowLoopHandler[OutT, CtxT] | None = None

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def exit_workflow_loop_handler(
        self, func: ExitWorkflowLoopHandler[OutT, CtxT]
    ) -> ExitWorkflowLoopHandler[OutT, CtxT]:
        self._exit_workflow_loop_impl = func

        return func

    def _exit_workflow_loop(
        self,
        output_message: AgentMessage[OutT, Any],
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        if self._exit_workflow_loop_impl:
            return self._exit_workflow_loop_impl(output_message, ctx=ctx, **kwargs)

        return False

    @final
    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: InT | Sequence[InT] | None = None,
        in_message: AgentMessage[InT, Any] | None = None,
        ctx: RunContextWrapper[CtxT] | None = None,
        entry_point: bool = False,
        forbid_state_change: bool = False,
        **kwargs: Any,
    ) -> AgentMessage[OutT, AgentState]:
        agent_message = in_message
        num_iterations = 0
        exit_message: AgentMessage[OutT, Any] | None = None

        while True:
            for subagent in self.subagents:
                agent_message = await subagent.run(
                    chat_inputs=chat_inputs,
                    in_args=in_args,
                    in_message=agent_message,
                    entry_point=entry_point,
                    forbid_state_change=forbid_state_change,
                    ctx=ctx,
                    **kwargs,
                )

                if subagent is self._end_agent:
                    num_iterations += 1
                    exit_message = cast("AgentMessage[OutT, AgentState]", agent_message)
                    if self._exit_workflow_loop(exit_message, ctx=ctx):
                        return exit_message
                    if num_iterations >= self._max_iterations:
                        logger.info(
                            f"Max iterations reached ({self._max_iterations}). Exiting loop."
                        )
                        return exit_message

                chat_inputs = None
                in_args = None
                entry_point = False
