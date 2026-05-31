"""
Tests for :class:`FileEditToolkit`.

The toolkit is **stateless**: backend + allowed_roots live on the
:class:`FileBackend` wired onto :attr:`RunContext.file_backend`, and
read-before-write bookkeeping lives on the active :class:`AgentLoop`
via :class:`FileEditSessionState`. These tests cover what the toolkit
itself owns (tool selection + per-tool configuration) and the
end-to-end Read → Write flow via ctx + the agent-state ContextVar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit import (
    DeleteTool,
    EditTool,
    FileEditSessionState,
    FileEditToolkit,
    LocalFileBackend,
    NullRedactor,
    ReadInput,
    ReadTool,
    WriteInput,
    WriteTool,
    reset_current_file_edit_state,
    set_current_file_edit_state,
)


def test_toolkit_returns_four_tools() -> None:
    tk = FileEditToolkit()
    tools = tk.tools()
    assert len(tools) == 4
    assert isinstance(tools[0], ReadTool)
    assert isinstance(tools[1], WriteTool)
    assert isinstance(tools[2], EditTool)
    assert isinstance(tools[3], DeleteTool)


def test_toolkit_per_tool_accessors() -> None:
    tk = FileEditToolkit()
    assert isinstance(tk.read, ReadTool)
    assert isinstance(tk.write, WriteTool)
    assert isinstance(tk.edit, EditTool)
    assert isinstance(tk.delete, DeleteTool)


def test_toolkit_passes_redactor_to_read() -> None:
    redactor = NullRedactor()
    tk = FileEditToolkit(redactor=redactor)
    # Private member access — verifying the configuration flow-through.
    assert tk.read._redactor is redactor  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_toolkit_passes_new_file_mode_to_write() -> None:
    tk = FileEditToolkit(new_file_mode=0o640)
    assert tk.write._new_file_mode == 0o640  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_read_then_write_composes(tmp_path: Path) -> None:
    """End-to-end: ctx-wired backend + agent-state ContextVar."""
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(
        file_backend=backend, session_key="default"
    )
    tk = FileEditToolkit(redactor=NullRedactor())
    f = tmp_path / "a.txt"
    f.write_text("original")

    token = set_current_file_edit_state(FileEditSessionState())
    try:
        await tk.read.run(ReadInput(path=str(f)), ctx=ctx)
        await tk.write.run(WriteInput(path=str(f), content="updated"), ctx=ctx)
    finally:
        reset_current_file_edit_state(token)

    assert f.read_text() == "updated"
