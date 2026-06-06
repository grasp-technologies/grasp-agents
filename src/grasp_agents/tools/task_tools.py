"""
Generic background-task companions: ``TaskOutput`` (poll) and ``KillTask``
(stop) for *any* tool call the agent loop moved to the background, addressed by
its ``task_id`` — a long shell command, a background sub-agent, whatever.

Stateless: both resolve the loop's
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` from the
call's :class:`AgentContext` (an explicitly-constructed manager wins, for
standalone use outside a loop), so a single instance is safe across agents,
sub-agents, and parallel replicas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..agent.background_tasks import BackgroundTaskManager, TaskOutputResult
from ..types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from ..agent.agent_context import AgentContext
    from ..run_context import RunContext


class TaskIdInput(BaseModel):
    """Input schema for ``TaskOutput`` / ``KillTask``."""

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


class TaskOutput(BaseTool[TaskIdInput, TaskOutputResult, Any]):
    """
    Poll a backgrounded tool call for new output by its ``task_id``.

    Returns the output produced since the previous poll, plus the final result
    once the task has finished (after which the task is dropped). You are
    notified automatically when a background task finishes, so do not call this
    in a loop just to wait for completion — use it to inspect progress (e.g. to
    decide whether to ``KillTask``) or to read the output after a completion
    notice.
    """

    name = "TaskOutput"
    description = (
        "Get new output from a backgrounded tool call (e.g. a long shell "
        "command) by its task_id: anything produced since your last check, "
        "plus the final result once it has finished. You are notified "
        "automatically when a background task finishes, so do not call this in "
        "a loop just to wait for completion — use it to inspect progress (e.g. "
        "to decide whether to KillTask) or to read output after a completion "
        "notice."
    )

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
    ) -> TaskOutputResult:
        del ctx, exec_id, progress_callback, path
        return _resolve_manager(self._manager, agent_ctx).read_output(inp.task_id)


class KillTask(BaseTool[TaskIdInput, TaskOutputResult, Any]):
    """Stop a backgrounded tool call by its ``task_id`` (e.g. a runaway shell)."""

    name = "KillTask"
    description = (
        "Stop a backgrounded tool call by its task_id (e.g. terminate a "
        "runaway shell command). Returns any output produced since the last "
        "check."
    )

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
    ) -> TaskOutputResult:
        del ctx, exec_id, progress_callback, path
        return await _resolve_manager(self._manager, agent_ctx).kill_task(inp.task_id)
