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
    from grasp_agents.run_context import RunContext
    from grasp_agents.skills.types import SkillFilter
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.tools.bash_common import ShellState
    from grasp_agents.tools.bash_session import BashSessionHolder
    from grasp_agents.tools.file_edit.session_state import FileEditSessionState
    from grasp_agents.tools.notebook_exec import KernelHolder

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
    # Names of the tools the owning agent was *explicitly* given (its ``tools=``
    # arg), as opposed to framework-auto-attached capability tools (skills /
    # memory / MCP / final-answer). A sub-agent (``AgentTool``) inherits only
    # these — capability tools come from the child's own feature flags.
    explicit_tool_names: frozenset[str] = frozenset()
    # Separate persistent kernel for ``RunPython`` — its own Python session, not
    # shared with the notebook kernel. ``None`` (the default) → each call opens a
    # throwaway kernel; ``AgentLoop`` wires a persistent one per loop.
    ipy_kernel_holder: KernelHolder | None = None
    # Per-agent view over the session-shared skill catalog (``ctx.skills``). The
    # skills prompt section and the ``load_skill`` / ``list_skills`` tools read it
    # so this agent sees only its allowed skills. ``None`` = the full catalog.
    skill_filter: SkillFilter | None = None

    @classmethod
    def create(
        cls,
        *,
        transcript: LLMAgentTranscript,
        tools: dict[str, BaseTool[Any, Any, Any]],
        bg_tasks: BackgroundTaskManager[Any] | None = None,
        agent_name: str = "",
        file_edit_state: FileEditSessionState | None = None,
        ipy_exec_context_id: str | None = None,
        nb_exec_context_id: str | None = None,
        path: list[str] | None = None,
        max_background: int = 16,
        explicit_tool_names: frozenset[str] = frozenset(),
        skill_filter: SkillFilter | None = None,
    ) -> AgentContext:
        """
        Build an ``AgentContext`` with fresh agent-scope state.

        Creates the session holders (the Bash session, the ``RunCell`` and
        ``RunPython`` kernels, the shell cwd), the background-task manager
        (unless one is supplied — ``path`` / ``max_background`` configure the
        built-in one), and an empty file-edit ledger — so callers pass only
        the agent-specific pieces (transcript, tool map) instead of
        hand-building each part. The tool-module imports are local to keep
        them off the agent core's import path.

        Pass ``ipy_exec_context_id`` / ``nb_exec_context_id`` when resuming a
        session to re-attach the ``RunPython`` / ``RunCell`` kernel to its
        persisted code-execution context (E2B) instead of starting fresh —
        read them from the holders' ``context_id`` before the sandbox was
        paused/snapshotted. See :class:`KernelHolder`.
        """
        from grasp_agents.tools.bash_common import ShellState  # noqa: PLC0415
        from grasp_agents.tools.bash_session import BashSessionHolder  # noqa: PLC0415
        from grasp_agents.tools.file_edit.session_state import (  # noqa: PLC0415
            FileEditSessionState as _FileEditSessionState,
        )
        from grasp_agents.tools.notebook_exec import KernelHolder  # noqa: PLC0415

        from .background_tasks import (  # noqa: PLC0415
            BackgroundTaskManager as _BackgroundTaskManager,
        )

        if bg_tasks is None:
            bg_tasks = _BackgroundTaskManager[Any](
                agent_name=agent_name,
                transcript=transcript,
                tools=tools,
                path=path,
                max_background=max_background,
            )

        return cls(
            transcript=transcript,
            tools=tools,
            file_edit_state=file_edit_state or _FileEditSessionState(),
            bg_tasks=bg_tasks,
            agent_name=agent_name,
            explicit_tool_names=explicit_tool_names,
            session_holder=BashSessionHolder(),
            nb_kernel_holder=KernelHolder(context_id=nb_exec_context_id),
            ipy_kernel_holder=KernelHolder(context_id=ipy_exec_context_id),
            shell_state=ShellState(),
            skill_filter=skill_filter,
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
        deliberately not part of the snapshot — live processes are
        non-transactional (a failed run's kernel/shell side effects can't
        be unwound), and the kernel-reset notice covers genuine restarts.
        """
        self.file_edit_state.import_state(
            state.read_file_state, state.dotfile_overrides
        )
        self.shell_state.cwd = state.shell_cwd
        self.bg_tasks.restore_pending_delivered(state.pending_delivered)

    async def close(self, *, ctx: RunContext[Any] | None = None) -> None:
        """
        Release everything this agent-scope context owns: cancel background
        tasks, close the process holders (shell + both kernels), and cascade
        teardown to tools that wrap sub-processors.

        Session teardown — reached via ``LLMAgent.aclose()``, never at run end
        (this state is session-scoped and survives run boundaries). ``ctx`` is
        the run context; it lets :meth:`BackgroundTaskManager.cancel_all`
        persist CANCELLED task records (the tool cascade needs no ctx — each
        wrapped processor closes with its own). Steps run in order —
        background tasks first, since one may be mid-command holding a
        session/kernel lock that would otherwise stall the holder closes for
        the full command timeout — and each is isolated so one failure can't
        leak the rest.
        """
        try:
            await self.bg_tasks.cancel_all(ctx=ctx)
        finally:
            try:
                await self._close_holders()
            finally:
                await self._close_tools()

    async def _close_holders(self) -> None:
        holders: tuple[BashSessionHolder | KernelHolder | None, ...] = (
            self.session_holder,
            self.nb_kernel_holder,
            self.ipy_kernel_holder,
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

    async def _close_tools(self) -> None:
        # Cascade teardown to tools that wrap sub-processors (AgentTool /
        # ProcessorTool close their template's session); plain tools no-op.
        for tool in self.tools.values():
            try:
                await tool.aclose()
            except Exception:
                logger.warning(
                    "Failed to close tool %r during agent-context cleanup",
                    tool.name,
                    exc_info=True,
                )
