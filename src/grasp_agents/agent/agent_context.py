"""
Per-:class:`AgentLoop` agent-scope state, passed explicitly to tools.

Where :class:`~grasp_agents.run_context.RunContext` is the *run*-scoped DI
container shared by every processor in a run, :class:`AgentContext` is the
*agent*-scoped counterpart: one per :class:`AgentLoop`, carrying the mutable
state a single agent's tools operate against — the file-edit ledger, the
persistent shell session, the background-task manager, and the agent's own
transcript / sibling tools (read by sub-agent tools as their *parent's*).

The loop owns one and passes it (as ``agent_ctx``) to every tool call, so tools
stay stateless: a single tool instance can be shared across agents without the
state of one clobbering another, and there is no async-local ``ContextVar`` to
set / reset around a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.bash_common import ShellState
    from ..tools.bash_session import BashSessionHolder
    from ..tools.file_edit.session_state import FileEditSessionState
    from ..tools.notebook_exec import KernelHolder
    from ..types.tool import BaseTool
    from .background_tasks import BackgroundTaskManager
    from .llm_agent_transcript import LLMAgentTranscript


@dataclass
class AgentContext:
    """The agent-scope state one :class:`AgentLoop` exposes to its tools."""

    transcript: LLMAgentTranscript
    tools: dict[str, BaseTool[Any, Any, Any]]
    file_edit_state: FileEditSessionState
    bg_tasks: BackgroundTaskManager[Any]
    session_holder: BashSessionHolder
    # Persistent Jupyter kernel for ``RunCell`` (the notebook's kernel) — one per
    # loop, like the shell.
    kernel_holder: KernelHolder
    # Fresh-``Bash`` cwd carried across calls (``cd`` sticks between turns).
    shell_state: ShellState
    # The owning agent's name. ``BaseTool.run_stream`` stamps it as the
    # ``destination`` on a tool's stream events, so a UI routes their (possibly
    # backgrounded / bubbled) output to the right agent's pane.
    agent_name: str = ""
    # Separate persistent kernel for ``RunPython`` — its own Python session, not
    # shared with the notebook kernel. ``None`` (the default) → each call opens a
    # throwaway kernel; ``AgentLoop`` wires a persistent one per loop.
    code_kernel_holder: KernelHolder | None = None

    @classmethod
    def create(
        cls,
        *,
        transcript: LLMAgentTranscript,
        tools: dict[str, BaseTool[Any, Any, Any]],
        bg_tasks: BackgroundTaskManager[Any],
        agent_name: str = "",
        file_edit_state: FileEditSessionState | None = None,
        exec_context_id: str | None = None,
    ) -> AgentContext:
        """
        Build an ``AgentContext`` with fresh per-loop state.

        Creates the per-loop holders (the Bash session, the ``RunCell`` and
        ``RunPython`` kernels, the shell cwd) and, unless one is supplied, an
        empty file-edit ledger — so callers pass only the agent-specific pieces
        (transcript, tool map, background-task manager) instead of hand-building
        each holder. The tool-module imports are local to keep them off the
        agent core's import path.

        Pass ``exec_context_id`` when resuming a session to re-attach the
        ``RunPython`` kernel to its persisted code-execution context (E2B)
        instead of starting fresh — read it from ``code_kernel_holder.context_id``
        before the sandbox was paused/snapshotted. See :class:`KernelHolder`.
        """
        from ..tools.bash_common import ShellState  # noqa: PLC0415
        from ..tools.bash_session import BashSessionHolder  # noqa: PLC0415
        from ..tools.file_edit.session_state import (  # noqa: PLC0415
            FileEditSessionState as _FileEditSessionState,
        )
        from ..tools.notebook_exec import KernelHolder  # noqa: PLC0415

        return cls(
            transcript=transcript,
            tools=tools,
            file_edit_state=file_edit_state or _FileEditSessionState(),
            bg_tasks=bg_tasks,
            agent_name=agent_name,
            session_holder=BashSessionHolder(),
            kernel_holder=KernelHolder(),
            code_kernel_holder=KernelHolder(context_id=exec_context_id),
            shell_state=ShellState(),
        )
