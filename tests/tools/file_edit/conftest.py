"""
Shared fixtures for the file-edit tool tests.

The file-edit tools read their per-agent :class:`FileEditSessionState` from
the :class:`AgentContext` passed on each call (no ContextVar). These fixtures
provide a fresh state and a minimal ``AgentContext`` wrapping it, so a test
can exercise read-before-write bookkeeping by passing ``agent_ctx=`` to a
tool call (and inspecting ``state`` directly).
"""

from __future__ import annotations

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.tools.bash_common import ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit import FileEditSessionState
from grasp_agents.tools.notebook_exec import KernelHolder


@pytest.fixture
def state() -> FileEditSessionState:
    """A fresh per-test file-edit ledger (the tool reads it via ``agent_ctx``)."""
    return FileEditSessionState()


@pytest.fixture
def agent_ctx(state: FileEditSessionState) -> AgentContext:
    """A minimal :class:`AgentContext` carrying ``state`` for tool calls."""
    transcript = LLMAgentTranscript()
    return AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=state,
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
        session_holder=BashSessionHolder(),
        kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )
