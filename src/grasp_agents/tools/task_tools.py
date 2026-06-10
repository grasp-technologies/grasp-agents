"""
``KillTask``: stop *any* backgrounded tool call by its ``task_id`` — a long
shell command, a background sub-agent, whatever.

There is no polling tool: the agent is notified with the result when a
background task finishes, and a running task's streamed output is mirrored to
its ``.grasp`` log (``Read`` / ``Grep`` it to inspect one mid-flight), so
``KillTask`` only stops work the agent no longer wants.

Stateless: it resolves the loop's
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` from the
call's :class:`AgentContext` (an explicitly-constructed manager wins, for
standalone use outside a loop), so a single instance is safe across agents,
sub-agents, and parallel replicas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..agent.background_tasks import BackgroundTaskManager, KillTaskResult
from ..types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from ..agent.agent_context import AgentContext
    from ..run_context import RunContext


class TaskIdInput(BaseModel):
    """Input schema for ``KillTask``."""

    task_id: str = Field(
        description="The id returned when a tool call was moved to the background."
    )


def _resolve_manager(
    explicit: BackgroundTaskManager[Any] | None, agent_ctx: AgentContext | None
) -> BackgroundTaskManager[Any]:
    manager = explicit if explicit is not None else getattr(agent_ctx, "bg_tasks", None)
    if manager is None:
        raise ValueError(
            "No background task manager: run under an agent loop (which wires "
            "one onto the call's AgentContext), or construct the tool with an "
            "explicit manager."
        )
    return manager


class KillTask(BaseTool[TaskIdInput, KillTaskResult, Any]):
    """
    Stop a backgrounded tool call by its ``task_id``.

    For tasks you can see should not continue — its output log shows it is
    misbehaving, or your plan changed and you no longer need it. Not a timeout:
    the framework already bounds runtime (per-tool ``timeout``); kill on what
    the task is *doing*, not on how long it has taken. Returns an excerpt of the
    output it produced before being stopped.
    """

    name = "KillTask"
    description = (
        "Stop a backgrounded tool call by its task_id — when its output shows "
        "it is misbehaving (e.g. a runaway command) or you no longer need its result. "
        "Returns an excerpt of the recent output produced before it was stopped."
    )
    untrusted_output = True

    def __init__(self, manager: BackgroundTaskManager[Any] | None = None) -> None:
        super().__init__()
        self._manager = manager

    async def _run(
        self,
        inp: TaskIdInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> KillTaskResult:
        del exec_id, progress_callback, path
        manager = _resolve_manager(self._manager, agent_ctx)
        return await manager.kill_task(inp.task_id, ctx=ctx)
