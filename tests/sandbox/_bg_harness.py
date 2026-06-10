"""
Shared helpers for exercising the backgrounding stack — the
``BackgroundTaskManager`` + ``Bash`` + ``KillTask`` — against a live
``ExecBackend``. Used by the seatbelt / srt / e2b sandbox tests so each backend
gets the same end-to-end coverage (background a command, observe its streamed
output, kill it) without duplicating the wiring.

A minimal :class:`AgentContext` + :class:`BackgroundTaskManager` stands in for a
full ``AgentLoop`` (the manager owns the deadline race; the bash tool reads its
``shell_state`` off the context).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager, KillTaskResult
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.tools.bash import Bash
from grasp_agents.tools.bash_common import BashInput, ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit.session_state import FileEditSessionState
from grasp_agents.tools.notebook_exec import KernelHolder
from grasp_agents.tools.task_tools import KillTask, TaskIdInput
from grasp_agents.types.events import ToolErrorEvent, ToolOutputEvent, ToolStreamEvent
from grasp_agents.types.items import FunctionToolCallItem

if TYPE_CHECKING:
    from grasp_agents.run_context import RunContext
    from grasp_agents.sandbox.environment import ExecutionEnvironment


def make_stack() -> tuple[AgentContext, BackgroundTaskManager[Any]]:
    """
    A minimal AgentContext + BackgroundTaskManager (no full AgentLoop).

    ``path=[]`` so a store-backed ctx persists task records + progress logs
    (``make_tool_call_path(None, ...)`` would otherwise yield ``None``).
    """
    transcript = LLMAgentTranscript()
    mgr: BackgroundTaskManager[Any] = BackgroundTaskManager(
        agent_name="t", transcript=transcript, tools={}, path=[]
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
    max_inline_result_chars: int | None = None,
) -> tuple[Any, str | None]:
    """
    Background ``command`` via the manager's deadline path (a fresh single-use
    ``Bash``). Returns ``(note, task_id)``: ``task_id`` is the sidelined task's
    id once it outlives ``abg`` (else ``None`` — it finished in the foreground).

    ``max_inline_result_chars`` overrides the ``Bash`` default (``None`` keeps
    it), so a test can force the cap-and-defer path with a small result.
    """
    bash = (
        Bash(auto_background_at=abg)
        if max_inline_result_chars is None
        else Bash(
            auto_background_at=abg, max_inline_result_chars=max_inline_result_chars
        )
    )
    call = FunctionToolCallItem(call_id="c1", name=bash.name, arguments="{}")
    note, _launched = await mgr.run_backgroundable(
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
) -> tuple[str, KillTaskResult]:
    """
    Wait until the backgrounded task finishes; return (streamed_output, final).

    With the polling tool gone, a finished task's output is read straight off
    its buffered events (the same source ``drain`` and the ``.grasp`` log use).
    Leaves the task in place so a later ``drain`` still delivers its note.
    """
    for _ in range(tries):
        pt = mgr.get(task_id)
        if pt.task.done():
            text = "".join(
                str(e.data) for e in pt.events if isinstance(e, ToolStreamEvent)
            )
            result: Any = None
            failed = False
            for e in pt.events:
                if isinstance(e, ToolErrorEvent):
                    result, failed = e.data, True
                elif isinstance(e, ToolOutputEvent):
                    result = e.data
            return text, KillTaskResult(
                task_id=task_id,
                tool_name=pt.tool_name,
                status="failed" if failed else "completed",
                output=text,
                result=result,
            )
        await asyncio.sleep(delay)
    raise AssertionError("backgrounded command never completed")


async def kill(mgr: BackgroundTaskManager[Any], task_id: str) -> KillTaskResult:
    """Stop a backgrounded task via ``KillTask``."""
    return await KillTask(mgr)._run(TaskIdInput(task_id=task_id))


async def flush(mgr: BackgroundTaskManager[Any], ctx: RunContext[Any]) -> None:
    """
    Drive one ``drain`` pass for its log-mirroring side effect, discarding the
    bubbled events. ``drain`` now owns flushing (there is no standalone
    ``flush_progress``); used by tests that just want a running task's output
    written to its ``.grasp`` log.
    """
    async for _ in mgr.drain(exec_id="t", ctx=ctx):
        pass


async def drain_notes(
    mgr: BackgroundTaskManager[Any], ctx: RunContext[Any]
) -> list[str]:
    """
    The completion notes a single turn-boundary ``drain`` injects. ``drain``
    also mirrors progress to the ``.grasp`` logs, so a truncated note points at
    the task's log.
    """
    from grasp_agents.types.events import UserMessageEvent

    return [
        e.data.text  # the rendered note text, as the model sees it (not a repr)
        async for e in mgr.drain(exec_id="t", ctx=ctx)
        if isinstance(e, UserMessageEvent)
    ]


async def marker_size(env: ExecutionEnvironment, marker: str) -> int:
    """Size of ``marker`` via the backend's file API (0 if absent)."""
    try:
        return (await env.file_backend.stat(Path(marker))).size
    except Exception:
        return 0
