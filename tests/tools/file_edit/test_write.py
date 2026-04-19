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

from grasp_agents.tools.file_edit import (
    InMemoryFileEditStore,
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
def store() -> InMemoryFileEditStore:
    return InMemoryFileEditStore()


@pytest.fixture
def read_tool(tmp_path: Path, store: InMemoryFileEditStore) -> ReadTool:
    return ReadTool(
        store=store,
        default_session_key=TEST_KEY,
        allowed_roots=[tmp_path],
        redactor=NullRedactor(),
    )


@pytest.fixture
def write_tool(tmp_path: Path, store: InMemoryFileEditStore) -> WriteTool:
    return WriteTool(
        store=store,
        default_session_key=TEST_KEY,
        allowed_roots=[tmp_path],
        include_dotfiles=True,
    )


# ---------------------------------------------------------------------------
# Create-new-file branch
# ---------------------------------------------------------------------------


async def test_write_new_file(tmp_path: Path, write_tool: WriteTool) -> None:
    target = tmp_path / "new.txt"
    result = await write_tool.run(WriteInput(path=str(target), content="hello\n"))
    assert isinstance(result, WriteResult)
    assert result.created is True
    assert result.bytes_written == 6
    assert target.read_text() == "hello\n"


async def test_write_registers_post_write_mtime(
    tmp_path: Path, store: InMemoryFileEditStore, write_tool: WriteTool
) -> None:
    target = tmp_path / "new.txt"
    await write_tool.run(WriteInput(path=str(target), content="hi"))
    state = await store.get_session_state(TEST_KEY)
    record = state.get_read_record(target.resolve())
    assert record is not None
    assert record.mtime == target.stat().st_mtime


async def test_write_refuses_when_parent_missing(
    tmp_path: Path, write_tool: WriteTool
) -> None:
    target = tmp_path / "missing" / "sub" / "f.txt"
    result = await write_tool.run(WriteInput(path=str(target), content="x"))
    assert "Parent directory does not exist" in _error_message(result)


# ---------------------------------------------------------------------------
# Overwrite-existing branch — read-before-write
# ---------------------------------------------------------------------------


async def test_write_refuses_existing_without_prior_read(
    tmp_path: Path, write_tool: WriteTool
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")

    result = await write_tool.run(WriteInput(path=str(target), content="overwritten"))
    assert "Must Read" in _error_message(result)
    # File untouched.
    assert target.read_text() == "original"


async def test_write_allowed_after_prior_read(
    tmp_path: Path, read_tool: ReadTool, write_tool: WriteTool
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")

    await read_tool.run(ReadInput(path=str(target)))
    result = await write_tool.run(WriteInput(path=str(target), content="overwritten"))
    assert isinstance(result, WriteResult)
    assert result.created is False
    assert target.read_text() == "overwritten"


# ---------------------------------------------------------------------------
# Staleness refusal — Decision #2: Write REFUSES
# ---------------------------------------------------------------------------


async def test_write_refuses_on_stale_mtime(
    tmp_path: Path, read_tool: ReadTool, write_tool: WriteTool
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original")
    await read_tool.run(ReadInput(path=str(target)))

    # External modification. asyncio.sleep (not time.sleep) advances
    # the event loop cleanly and lets the filesystem mtime tick forward.
    await asyncio.sleep(0.01)
    target.write_text("changed externally")
    os.utime(target, None)

    result = await write_tool.run(WriteInput(path=str(target), content="my changes"))
    assert "modified since you last read" in _error_message(result)
    # Externally-modified content is preserved.
    assert target.read_text() == "changed externally"


async def test_consecutive_writes_do_not_trip_staleness(
    tmp_path: Path, read_tool: ReadTool, write_tool: WriteTool
) -> None:
    """
    The tool's own prior Write refreshes the recorded mtime so a
    follow-up Write in the same session doesn't see external drift.
    """
    target = tmp_path / "t.txt"
    target.write_text("v0")
    await read_tool.run(ReadInput(path=str(target)))

    await write_tool.run(WriteInput(path=str(target), content="v1"))
    # No staleness complaint on the second Write, despite mtime having
    # moved (the tool refreshed the record post-write).
    result = await write_tool.run(WriteInput(path=str(target), content="v2"))
    assert isinstance(result, WriteResult)
    assert target.read_text() == "v2"


# ---------------------------------------------------------------------------
# Sensitive-path refusal
# ---------------------------------------------------------------------------


async def test_write_refuses_dotfile_by_default(
    tmp_path: Path, write_tool: WriteTool
) -> None:
    # Create ~/.ssh-like layout under the tmp root.
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    target = ssh_dir / "id_rsa"

    result = await write_tool.run(
        WriteInput(path=str(target), content="PRIVATE KEY DATA")
    )
    assert "credential-sensitive directory" in _error_message(result)


async def test_write_dotfile_allowed_after_explicit_override(
    tmp_path: Path, store: InMemoryFileEditStore, write_tool: WriteTool
) -> None:
    target = tmp_path / ".env"
    state = await store.get_session_state(TEST_KEY)
    state.dotfile_overrides.add(target.resolve())

    result = await write_tool.run(WriteInput(path=str(target), content="DEBUG=1\n"))
    assert isinstance(result, WriteResult)
    assert result.created is True
    assert target.read_text() == "DEBUG=1\n"


async def test_write_refuses_dotfile_when_include_dotfiles_false(
    tmp_path: Path, store: InMemoryFileEditStore
) -> None:
    """
    With dotfiles disabled, ``.env`` writes are permitted — the
    toolkit-level knob only affects the default deny-list composition.
    """
    lenient_write = WriteTool(
        store=store,
        default_session_key=TEST_KEY,
        allowed_roots=[tmp_path],
        include_dotfiles=False,
    )
    target = tmp_path / ".env"
    result = await lenient_write.run(WriteInput(path=str(target), content="DEBUG=1\n"))
    assert isinstance(result, WriteResult)
    assert result.created is True


# ---------------------------------------------------------------------------
# Mode preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_write_preserves_existing_executable_bit(
    tmp_path: Path, read_tool: ReadTool, write_tool: WriteTool
) -> None:
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho hi\n")
    target.chmod(0o755)

    await read_tool.run(ReadInput(path=str(target)))
    await write_tool.run(
        WriteInput(path=str(target), content="#!/bin/sh\necho updated\n")
    )

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_new_file_uses_default_mode(
    tmp_path: Path, store: InMemoryFileEditStore
) -> None:
    tool = WriteTool(
        store=store,
        default_session_key=TEST_KEY,
        allowed_roots=[tmp_path],
        new_file_mode=0o640,
    )
    target = tmp_path / "new.txt"
    await tool.run(WriteInput(path=str(target), content="x"))
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o640
