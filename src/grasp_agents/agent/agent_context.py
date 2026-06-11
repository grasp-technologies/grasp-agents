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
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.bash_common import ShellState
    from ..tools.bash_session import BashSessionHolder
    from ..tools.file_edit.session_state import FileEditSessionState
    from ..tools.notebook_exec import KernelHolder
    from ..types.tool import BaseTool
    from .background_tasks import BackgroundTaskManager
    from .llm_agent_transcript import LLMAgentTranscript

logger = getLogger(__name__)


@dataclass(frozen=True)
class AgentContextState:
    """
    Snapshot of the agent-context state that is paired with the transcript.

    Taken at run start so a failed run rolls back both together: the
    file-read ledger and shell cwd describe what the *transcript* knows, and
    a deferred task-record flip must not outlive its rolled-back completion
    note (see :meth:`AgentContext.restore_state`).
    """

    read_file_state: dict[str, float]
    dotfile_overrides: list[str]
    shell_cwd: str | None
    pending_delivered: dict[str, dict[str, Any]]


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
    nb_kernel_holder: KernelHolder
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
            nb_kernel_holder=KernelHolder(),
            code_kernel_holder=KernelHolder(context_id=exec_context_id),
            shell_state=ShellState(),
        )

    def snapshot_state(self) -> AgentContextState:
        """Capture the transcript-paired state for a failed-run rollback."""
        read_file_state, dotfile_overrides = self.file_edit_state.export_state()
        return AgentContextState(
            read_file_state=read_file_state,
            dotfile_overrides=dotfile_overrides,
            shell_cwd=self.shell_state.cwd,
            pending_delivered=self.bg_tasks.export_pending_delivered(),
        )

    def restore_state(self, state: AgentContextState) -> None:
        """
        Restore a :meth:`snapshot_state` snapshot after a failed run.

        Keeps the context consistent with the rolled-back transcript: the
        file-read ledger forgets Reads whose results were rolled back, the
        fresh-``Bash`` cwd matches the history the model still sees, and the
        failed run's deferred task-record flips are discarded (their
        completion notes were rolled back, so the records must stay
        COMPLETED for a later resume to re-inject). The process holders are
        deliberately not part of the snapshot — they are closed at the end
        of every run and reopen lazily, surfacing a reset notice.
        """
        self.file_edit_state.import_state(
            state.read_file_state, state.dotfile_overrides
        )
        self.shell_state.cwd = state.shell_cwd
        self.bg_tasks.restore_pending_delivered(state.pending_delivered)

    async def close(self) -> None:
        """
        Close the per-loop process holders (the shell session and both
        kernels). Each closes independently so one failed close cannot leak
        the others' processes.
        """
        holders: tuple[BashSessionHolder | KernelHolder | None, ...] = (
            self.session_holder,
            self.nb_kernel_holder,
            self.code_kernel_holder,
        )
        for holder in holders:
            if holder is None:
                continue
            try:
                await holder.close()
            except Exception:
                logger.warning(
                    "Failed to close a session/kernel holder during "
                    "agent-context cleanup",
                    exc_info=True,
                )
