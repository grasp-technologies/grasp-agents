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
    Read a backgrounded tool call's output by its ``task_id``.

    You are notified automatically — with the result — when a background task
    finishes, so this is **not** for waiting on completion. Its two real uses:
    (1) inspect a *still-running* task's output so far (e.g. to judge whether it
    is misbehaving and should be ``KillTask``-ed); (2) read the *full* result of
    a finished task whose completion notification said its output was truncated.
    Returns the output since your last read, plus the terminal result once
    finished (after which the task is dropped).
    """

    name = "TaskOutput"
    description = (
        "Read a backgrounded tool call's output by its task_id. You are "
        "notified automatically (with the result) when a task finishes, so do "
        "NOT call this to wait for completion. Use it to (1) inspect a "
        "still-running task's output so far — e.g. to decide whether to "
        "KillTask — or (2) read the full result of a finished task whose "
        "completion notification said the output was truncated. Returns output "
        "since your last check, plus the terminal result once finished."
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
        del exec_id, progress_callback, path
        manager = _resolve_manager(self._manager, agent_ctx)
        return await manager.read_output(inp.task_id, ctx=ctx)


class KillTask(BaseTool[TaskIdInput, TaskOutputResult, Any]):
    """
    Stop a backgrounded tool call by its ``task_id``.

    For tasks you can see should not continue — output (via ``TaskOutput``)
    shows it is misbehaving, or your plan changed and you no longer need it.
    Not a timeout: the framework already bounds runtime (per-tool ``timeout``);
    kill on what the task is *doing*, not on how long it has taken. Returns any
    output produced since the last check.
    """

    name = "KillTask"
    description = (
        "Stop a backgrounded tool call by its task_id — when its output shows "
        "it is misbehaving (e.g. a runaway command) or you no longer need it. "
        "This is not a timeout (the framework bounds runtime itself); decide "
        "from what the task is doing, not from elapsed time. Returns any output "
        "produced since the last check."
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
        del exec_id, progress_callback, path
        manager = _resolve_manager(self._manager, agent_ctx)
        return await manager.kill_task(inp.task_id, ctx=ctx)
