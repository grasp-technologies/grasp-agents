"""
Shared helpers for exercising the backgrounding stack — the
``BackgroundTaskManager`` + ``Bash`` + generic ``TaskOutput`` / ``KillTask`` —
against a live ``ExecBackend``. Used by the seatbelt / srt / e2b sandbox tests
so each backend gets the same end-to-end coverage (background a command, poll
incremental output, kill it) without duplicating the wiring.

A minimal :class:`AgentContext` + :class:`BackgroundTaskManager` stands in for a
full ``AgentLoop`` (the manager owns the deadline race; the bash tool reads its
``shell_state`` off the context).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager, TaskOutputResult
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.tools.bash import Bash
from grasp_agents.tools.bash_common import BashInput, ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit.session_state import FileEditSessionState
from grasp_agents.tools.notebook_exec import KernelHolder
from grasp_agents.tools.task_tools import KillTask, TaskIdInput, TaskOutput
from grasp_agents.types.items import FunctionToolCallItem

if TYPE_CHECKING:
    from grasp_agents.run_context import RunContext
    from grasp_agents.sandbox.environment import ExecutionEnvironment


def make_stack() -> tuple[AgentContext, BackgroundTaskManager[Any]]:
    """A minimal AgentContext + BackgroundTaskManager (no full AgentLoop)."""
    transcript = LLMAgentTranscript()
    mgr: BackgroundTaskManager[Any] = BackgroundTaskManager(
        agent_name="t", transcript=transcript, tools={}
    )
    agent_ctx = AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=FileEditSessionState(),
        bg_tasks=mgr,
        session_holder=BashSessionHolder(),
        kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )
    return agent_ctx, mgr


async def background(
    mgr: BackgroundTaskManager[Any],
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    command: str,
    *,
    abg: float = 0.3,
    timeout: float = 60,
) -> tuple[Any, str | None]:
    """
    Background ``command`` via the manager's deadline path (a fresh single-use
    ``Bash``). Returns ``(note, task_id)``: ``task_id`` is the sidelined task's
    id once it outlives ``abg`` (else ``None`` — it finished in the foreground).
    """
    bash = Bash(auto_background_at=abg)
    call = FunctionToolCallItem(call_id="c1", name=bash.name, arguments="{}")
    note, _launched = await mgr.run_with_deadline(
        call,
        bash,
        BashInput(command=command, timeout=timeout),
        ctx=ctx,
        exec_id="t",
        agent_ctx=agent_ctx,
    )
    task_id = next(iter(mgr._tasks), None)  # pyright: ignore[reportPrivateUsage]
    return note, task_id


async def poll_until_done(
    mgr: BackgroundTaskManager[Any],
    task_id: str,
    *,
    tries: int = 80,
    delay: float = 0.25,
) -> tuple[str, TaskOutputResult]:
    """Poll ``TaskOutput`` until the task finishes; return (output, final)."""
    poll = TaskOutput(mgr)
    collected = ""
    for _ in range(tries):
        out = await poll._run(TaskIdInput(task_id=task_id))
        collected += out.output
        if out.status in {"completed", "failed"}:
            return collected, out
        await asyncio.sleep(delay)
    raise AssertionError("backgrounded command never completed")


async def kill(mgr: BackgroundTaskManager[Any], task_id: str) -> TaskOutputResult:
    """Stop a backgrounded task via ``KillTask``."""
    return await KillTask(mgr)._run(TaskIdInput(task_id=task_id))


async def marker_size(env: ExecutionEnvironment, marker: str) -> int:
    """Size of ``marker`` via the backend's file API (0 if absent)."""
    try:
        return (await env.file_backend.stat(Path(marker))).size
    except Exception:
        return 0
