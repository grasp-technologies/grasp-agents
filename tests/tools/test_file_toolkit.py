"""
Tests for :class:`FileToolkit` — the unified file toolkit bundling the
edit tools (Read/Write/Edit/Delete) and the search tools (Glob/Grep).

The toolkit is **stateless**: backend + allowed_roots live on the
:class:`FileBackend` wired onto :attr:`SessionContext.file_backend`, and
read-before-write bookkeeping lives on the active :class:`AgentLoop`
via :class:`FileEditSessionState`. These tests cover what the toolkit
owns (tool selection + per-tool configuration) and the end-to-end
Read → Write flow via ctx + the agent's :class:`AgentContext`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.bash_common import ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit import (
    DeleteTool,
    EditTool,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    ReadTool,
    WriteInput,
    WriteTool,
)
from grasp_agents.tools.file_search import GlobTool, GrepTool
from grasp_agents.tools.notebook_exec import KernelHolder

if TYPE_CHECKING:
    from pathlib import Path


def test_toolkit_returns_all_six_tools() -> None:
    tools = FileToolkit().tools()
    assert [type(t) for t in tools] == [
        ReadTool,
        WriteTool,
        EditTool,
        DeleteTool,
        GlobTool,
        GrepTool,
    ]


def test_read_only_tools_excludes_mutators() -> None:
    tools = FileToolkit().read_only_tools()
    # Read + the two search tools; no Write/Edit/Delete.
    assert [type(t) for t in tools] == [ReadTool, GlobTool, GrepTool]


def test_per_tool_accessors() -> None:
    tk = FileToolkit()
    assert isinstance(tk.read, ReadTool)
    assert isinstance(tk.write, WriteTool)
    assert isinstance(tk.edit, EditTool)
    assert isinstance(tk.delete, DeleteTool)
    assert isinstance(tk.glob, GlobTool)
    assert isinstance(tk.grep, GrepTool)


def test_passes_redactor_to_read() -> None:
    redactor = NullRedactor()
    tk = FileToolkit(redactor=redactor)
    assert tk.read._redactor is redactor  # pyright: ignore[reportPrivateUsage]


def test_passes_new_file_mode_to_write() -> None:
    tk = FileToolkit(new_file_mode=0o640)
    assert tk.write._new_file_mode == 0o640  # pyright: ignore[reportPrivateUsage]


def test_passes_max_file_bytes_to_read() -> None:
    tk = FileToolkit(max_file_bytes=123)
    assert tk.read._max_file_bytes == 123  # pyright: ignore[reportPrivateUsage]


def test_timeout_propagates_to_tools() -> None:
    tk = FileToolkit(tool_timeout=5.0)
    assert tk.glob.timeout == 5.0
    assert tk.grep.timeout == 5.0
    assert tk.read.timeout == 5.0


@pytest.mark.parametrize("include_hidden", [True, False])
def test_glob_hidden_flag_propagates(include_hidden: bool) -> None:
    tk = FileToolkit(glob_include_hidden=include_hidden)
    assert tk.glob._include_hidden is include_hidden  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_read_then_write_composes(tmp_path: Path) -> None:
    """End-to-end: ctx-wired backend + the agent's AgentContext."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(
        file_backend=backend, session_key="default"
    )
    tk = FileToolkit(redactor=NullRedactor())
    f = tmp_path / "a.txt"
    f.write_text("original")

    transcript = LLMAgentTranscript()
    agent_ctx = AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=FileEditSessionState(),
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
        session_holder=BashSessionHolder(),
        nb_kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )
    await tk.read.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    await tk.write.run(
        WriteInput(path=str(f), content="updated"), ctx=ctx, agent_ctx=agent_ctx
    )

    assert f.read_text() == "updated"
