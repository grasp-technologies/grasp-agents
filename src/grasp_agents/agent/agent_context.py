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
    # Persistent Jupyter kernel for ``RunCell`` (one per loop, like the shell).
    kernel_holder: KernelHolder
    # Fresh-``Bash`` cwd carried across calls (``cd`` sticks between turns).
    shell_state: ShellState
