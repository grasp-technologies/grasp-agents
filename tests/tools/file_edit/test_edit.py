"""
Unit tests for :class:`EditTool`.

Focus: the read-before-edit invariant, staleness refusal, fuzzy chain
integration, quote preservation, mode preservation, and atomic-write
safety. End-to-end — :class:`EditTool` is built on top of
:mod:`fuzzy_match`, so the matching strategies themselves are covered
separately in ``test_fuzzy_match.py``; this file checks the wiring.
"""

from __future__ import annotations

import asyncio
import os
import stat
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.file_edit import (
    EditInput,
    EditResult,
    EditTool,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    ReadTool,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> SessionContext[Any]:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    return SessionContext[Any](file_backend=backend, session_key=TEST_KEY)


@pytest.fixture
def read_tool() -> ReadTool:
    return ReadTool(redactor=NullRedactor())


@pytest.fixture
def edit_tool() -> EditTool:
    return EditTool()


# ---------------------------------------------------------------------------
# Happy path — exact match
# ---------------------------------------------------------------------------


async def test_edit_exact_match(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hello world\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="hello", new_string="goodbye"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert result.edits_applied == 1
    assert result.strategy == "exact"
    assert target.read_text() == "goodbye world\n"


async def test_edit_refuses_when_old_equals_new(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("abc\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="abc", new_string="abc"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    # No-op: Edit explicitly refuses so the model gets a clear signal.
    assert "identical" in _error_message(result)


async def test_edit_empty_old_string_rejected_by_schema() -> None:
    # Pydantic ``min_length=1`` rejects before the tool runs. This test is
    # synchronous logic but declared ``async`` because the module-level
    # pytestmark enforces the asyncio decorator on every test.
    with pytest.raises(ValidationError):
        EditInput(path="/tmp/x", old_string="", new_string="y")


# ---------------------------------------------------------------------------
# Fuzzy fallthrough
# ---------------------------------------------------------------------------


async def test_edit_falls_through_to_line_trimmed(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.py"
    # Trailing whitespace in file, model sent clean pattern.
    target.write_text("def foo():   \n    return 1\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(
            path=str(target),
            old_string="def foo():\n    return 1",
            new_string="def bar():\n    return 2",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert result.strategy == "line_trimmed"
    assert "def bar():" in target.read_text()


# ---------------------------------------------------------------------------
# Uniqueness / replace_all
# ---------------------------------------------------------------------------


async def test_edit_refuses_ambiguous_match(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x = 1\ny = 1\nz = 1\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="= 1", new_string="= 2"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "3 matches" in _error_message(result)
    # File untouched.
    assert target.read_text() == "x = 1\ny = 1\nz = 1\n"


async def test_edit_replace_all(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x = 1\ny = 1\nz = 1\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(
            path=str(target), old_string="= 1", new_string="= 2", replace_all=True
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert result.edits_applied == 3
    assert target.read_text() == "x = 2\ny = 2\nz = 2\n"


async def test_edit_no_match(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="missing-text", new_string="new"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "Could not find a match" in _error_message(result)


# ---------------------------------------------------------------------------
# Read-before-edit + staleness
# ---------------------------------------------------------------------------


async def test_edit_refuses_without_prior_read(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="hello", new_string="bye"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "Must Read" in _error_message(result)
    # File untouched.
    assert target.read_text() == "hello\n"


async def test_edit_refuses_on_stale_mtime(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("original\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    await asyncio.sleep(0.01)
    target.write_text("external change\n")
    os.utime(target, None)

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="external", new_string="tampered"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "modified since you last read" in _error_message(result)
    # External edit preserved.
    assert target.read_text() == "external change\n"


async def test_consecutive_edits_do_not_trip_staleness(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    """
    The tool's own edit refreshes the session-state mtime, so a follow-up
    Edit in the same session doesn't see its own write as drift.
    """
    target = tmp_path / "f.txt"
    target.write_text("a b c\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    await edit_tool.run(
        EditInput(path=str(target), old_string="a", new_string="A"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    result = await edit_tool.run(
        EditInput(path=str(target), old_string="b", new_string="B"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert target.read_text() == "A B c\n"


async def test_edit_refuses_missing_file(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "does-not-exist.txt"
    result = await edit_tool.run(
        EditInput(path=str(target), old_string="x", new_string="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    # resolve_safe(must_exist=True) → PathAccessError → ToolError.
    err = _error_message(result)
    assert "does not exist" in err or "not found" in err.lower()


# ---------------------------------------------------------------------------
# Sensitive path / dotfile
# ---------------------------------------------------------------------------


async def test_edit_refuses_dotfile_by_default(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / ".env"
    target.write_text("SECRET=x\n")
    # No Read registered — but the dotfile check runs before read-before-edit,
    # so we get the sensitivity error first.
    result = await edit_tool.run(
        EditInput(path=str(target), old_string="SECRET=x", new_string="SECRET=y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "credential-like" in _error_message(result)


async def test_edit_dotfile_allowed_after_override(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    state: FileEditSessionState,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / ".env"
    target.write_text("DEBUG=0\n")
    state.add_dotfile_override(target.resolve())

    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)
    result = await edit_tool.run(
        EditInput(path=str(target), old_string="DEBUG=0", new_string="DEBUG=1"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert target.read_text() == "DEBUG=1\n"


# ---------------------------------------------------------------------------
# Quote-convention preservation (CC-style)
# ---------------------------------------------------------------------------


async def test_edit_preserves_curly_quote_convention(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    """
    File uses curly quotes; model's new_string has straight quotes.
    EditTool rewrites the replacement to keep the file's convention.
    """
    target = tmp_path / "doc.md"
    target.write_text("She said “hello” to the crowd.\n")
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    # Model uses straight quotes in both old_string and new_string.
    result = await edit_tool.run(
        EditInput(
            path=str(target),
            old_string='She said "hello"',
            new_string='She shouted "goodbye"',
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert result.strategy == "unicode_normalized"
    # The file keeps curly convention — straight " should not appear.
    new_content = target.read_text()
    assert "“goodbye”" in new_content
    assert '"' not in new_content


async def test_edit_unicode_em_dash_match(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "doc.md"
    target.write_text("flag — enabled\n")  # real em-dash
    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)

    result = await edit_tool.run(
        EditInput(
            path=str(target),
            old_string="flag -- enabled",
            new_string="flag -- disabled",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, EditResult)
    assert result.strategy == "unicode_normalized"
    assert "disabled" in target.read_text()


# ---------------------------------------------------------------------------
# Mode preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_edit_preserves_executable_bit(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    read_tool: ReadTool,
    edit_tool: EditTool,
) -> None:
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho hi\n")
    target.chmod(0o755)

    await read_tool.run(ReadInput(path=str(target)), ctx=ctx, agent_ctx=agent_ctx)
    await edit_tool.run(
        EditInput(path=str(target), old_string="echo hi", new_string="echo bye"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o755
    assert "echo bye" in target.read_text()


# ---------------------------------------------------------------------------
# Binary / device refusal
# ---------------------------------------------------------------------------


async def test_edit_refuses_binary_extension(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    edit_tool: EditTool,
) -> None:
    # Even with a prior Read we couldn't have performed (binary guard on Read
    # too), Edit's own binary guard refuses up-front.
    target = tmp_path / "img.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await edit_tool.run(
        EditInput(path=str(target), old_string="x", new_string="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "binary" in _error_message(result).lower()


async def test_edit_refuses_device_path(
    tmp_path: Path,
    ctx: SessionContext[Any],
    agent_ctx: AgentContext,
    edit_tool: EditTool,
) -> None:
    # Device guard runs via validate_path against the literal path, before
    # resolve_safe — doesn't matter that /dev/stdin isn't under tmp_path.
    del tmp_path
    result = await edit_tool.run(
        EditInput(path="/dev/stdin", old_string="x", new_string="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "device" in _error_message(result).lower()
