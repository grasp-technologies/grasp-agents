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
from grasp_agents.tools.file_edit import FileEditSessionState
from tests._helpers import _make_agent_ctx


@pytest.fixture
def state() -> FileEditSessionState:
    """A fresh per-test file-edit ledger (the tool reads it via ``agent_ctx``)."""
    return FileEditSessionState()


@pytest.fixture
def agent_ctx(state: FileEditSessionState) -> AgentContext:
    """A minimal :class:`AgentContext` carrying ``state`` for tool calls."""
    return _make_agent_ctx(file_edit_state=state)
