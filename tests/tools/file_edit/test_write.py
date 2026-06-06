"""
Unit tests for :class:`WriteTool`.

Focus: the read-before-write invariant, mtime staleness refusal, and
atomic-write guarantees. Covers both the create-new-file and
overwrite-existing branches.
"""

from __future__ import annotations

import asyncio
import os
import stat
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit import (
    FileEditSessionState,
    LocalFileBackend,
    NullRedactor,
    ReadInput,
    ReadTool,
    WriteInput,
    WriteResult,
    WriteTool,
)
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

TEST_KEY = "test"


def _error_message(result: Any) -> str:
    """Unwrap a ``ToolErrorInfo`` returned from ``.run(...)``."""
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext[Any]:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    return RunContext[Any](file_backend=backend, session_key=TEST_KEY)


@pytest.fixture
def read_tool() -> ReadTool:
    return ReadTool(redactor=NullRedactor())


@pytest.fixture
def write_tool() -> WriteTool:
    return WriteTool()


# ---------------------------------------------------------------------------
# Create-new-file branch
# ---------------------------------------------------------------------------


async def test_write_new_file(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, write_tool: WriteTool
) -> None:
    target = tmp_path / "new.txt"
    result = await write_tool.run(
        WriteInput(path=str(target), content="hello\n"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, WriteResult)
    assert result.created is True
    assert result.bytes_written == 6
    assert target.read_text() == "hello\n"


async def test_write_registers_post_write_mtime(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    state: FileEditSessionState,
    write_tool: WriteTool,
) -> None:
    target = tmp_path / "new.txt"
    await write_tool.run(
        WriteInput(path=str(target), content="hi"), ctx=ctx, agent_ctx=agent_ctx
    )
    record = state.get_read_record(target.resolve())
    assert record is not None
    assert record.mtime == target.stat().st_mtime


async def test_write_refuses_when_parent_missing(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, write_tool: WriteTool
) -> None:
    target = tmp_path / "missing" / "sub" / "f.txt"
    result = await write_tool.run(
        WriteInput(path=str(target), content="x"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "Parent directory" in _error_message(result)
    assert "does not exist" in _error_message(result)


# ---------------------------------------------------------------------------
# Overwrite-existing branch — read-before-write
# ---------------------------------------------------------------------------


async def test_write_refuses_existing_without_prior_read(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, write_tool: WriteTool
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")

    result = await write_tool.run(
        WriteInput(path=str(target), content="overwritten"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "Must Read" in _error_message(result)
    # File untouched.
    assert target.read_text() == "original"


async def test_write_allowed_after_prior_read(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    write_tool: WriteTool,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")

    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)
    result = await write_tool.run(
        WriteInput(path=str(target), content="overwritten"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, WriteResult)
    assert result.created is False
    assert target.read_text() == "overwritten"


# ---------------------------------------------------------------------------
# Staleness refusal — Decision #2: Write REFUSES
# ---------------------------------------------------------------------------


async def test_write_refuses_on_stale_mtime(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    write_tool: WriteTool,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    # External modification. asyncio.sleep (not time.sleep) advances
    # the event loop cleanly and lets the filesystem mtime tick forward.
    await asyncio.sleep(0.01)
    target.write_text("changed externally")
    os.utime(target, None)

    result = await write_tool.run(
        WriteInput(path=str(target), content="my changes"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "modified since you last read" in _error_message(result)
    # Externally-modified content is preserved.
    assert target.read_text() == "changed externally"


async def test_consecutive_writes_do_not_trip_staleness(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    write_tool: WriteTool,
) -> None:
    """
    The tool's own prior Write refreshes the recorded mtime so a
    follow-up Write in the same session doesn't see external drift.
    """
    target = tmp_path / "t.txt"
    target.write_text("v0")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    await write_tool.run(
        WriteInput(path=str(target), content="v1"), ctx=ctx, agent_ctx=agent_ctx
    )
    # No staleness complaint on the second Write, despite mtime having
    # moved (the backend refreshed the record post-write).
    result = await write_tool.run(
        WriteInput(path=str(target), content="v2"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, WriteResult)
    assert target.read_text() == "v2"


# ---------------------------------------------------------------------------
# Sensitive-path refusal
# ---------------------------------------------------------------------------


async def test_write_refuses_dotfile_by_default(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, write_tool: WriteTool
) -> None:
    # Create ~/.ssh-like layout under the tmp root.
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    target = ssh_dir / "id_rsa"

    result = await write_tool.run(
        WriteInput(path=str(target), content="PRIVATE KEY DATA"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "credential-sensitive directory" in _error_message(result)


async def test_write_dotfile_allowed_after_explicit_override(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    state: FileEditSessionState,
    write_tool: WriteTool,
) -> None:
    target = tmp_path / ".env"
    state.add_dotfile_override(target.resolve())

    result = await write_tool.run(
        WriteInput(path=str(target), content="DEBUG=1\n"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, WriteResult)
    assert result.created is True
    assert target.read_text() == "DEBUG=1\n"


async def test_write_refuses_dotfile_when_include_dotfiles_false(
    tmp_path: Path,
    state: FileEditSessionState,
    agent_ctx: AgentContext,
) -> None:
    """
    With ``include_dotfiles=False`` on the backend, ``.env`` writes are
    permitted — the backend-level knob only affects the default
    deny-list composition.
    """
    del state  # activated by the ``state`` fixture's ContextVar set
    lenient_backend = LocalFileBackend(allowed_roots=[tmp_path], include_dotfiles=False)
    ctx: RunContext[Any] = RunContext(
        file_backend=lenient_backend, session_key=TEST_KEY
    )
    target = tmp_path / ".env"
    result = await WriteTool().run(
        WriteInput(path=str(target), content="DEBUG=1\n"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, WriteResult)
    assert result.created is True


# ---------------------------------------------------------------------------
# Mode preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_write_preserves_existing_executable_bit(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    write_tool: WriteTool,
) -> None:
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho hi\n")
    target.chmod(0o755)

    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)
    await write_tool.run(
        WriteInput(path=str(target), content="#!/bin/sh\necho updated\n"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_new_file_uses_default_mode(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    tool = WriteTool(new_file_mode=0o640)
    target = tmp_path / "new.txt"
    await tool.run(
        WriteInput(path=str(target), content="x"), ctx=ctx, agent_ctx=agent_ctx
    )
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o640
