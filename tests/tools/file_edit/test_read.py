"""
Unit tests for :class:`ReadTool`.

Uses the public ``.run(...)`` API with a ctx-driven backend: errors
return a ``ToolErrorInfo`` value (not an exception), matching how the
agent loop sees tool calls. The :func:`_error_message` helper unwraps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit import (
    DefaultSecretRedactor,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    ReadResult,
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


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext[Any]:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    return RunContext[Any](file_backend=backend, session_key=TEST_KEY)


@pytest.fixture
def read_tool() -> ReadTool:
    return ReadTool(redactor=NullRedactor(), max_read_chars=100_000)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_returns_line_numbered_content(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    f = tmp_path / "a.py"
    f.write_text("line one\nline two\nline three\n")

    result = await read_tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)

    assert isinstance(result, ReadResult)
    assert "     1\tline one" in result.content
    assert "     2\tline two" in result.content
    assert "     3\tline three" in result.content
    assert result.total_lines == 3


async def test_read_respects_offset_and_limit(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")

    result = await read_tool.run(
        ReadInput(path=str(f), offset=3, limit=2), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, ReadResult)
    # Lines 3 and 4 only.
    assert "     3\tline 3" in result.content
    assert "     4\tline 4" in result.content
    assert "line 5" not in result.content
    assert "line 1" not in result.content
    assert result.total_lines == 10


async def test_read_records_state_for_read_before_write(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    state: FileEditSessionState,
    read_tool: ReadTool,
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi\n")

    await read_tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)

    record = state.get_read_record(f.resolve())
    assert record is not None
    assert record.mtime == f.stat().st_mtime


async def test_repeat_read_returns_fresh_content(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    """
    Re-reading the same region returns the content again — no dedup
    stub. The model may want to refresh attention recency.
    """
    f = tmp_path / "a.txt"
    f.write_text("hello\n")

    first = await read_tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    second = await read_tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)

    assert isinstance(first, ReadResult)
    assert isinstance(second, ReadResult)
    assert first.content == second.content


# ---------------------------------------------------------------------------
# Guards — errors returned as ToolErrorInfo via ``.run(...)``
# ---------------------------------------------------------------------------


async def test_device_path_refused(
    ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    result = await read_tool.run(
        ReadInput(path="/dev/stdin"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "device path" in _error_message(result)


async def test_binary_extension_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    result = await read_tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    assert "binary" in _error_message(result)


async def test_path_outside_root_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    outside = tmp_path.parent / "escape.txt"
    outside.write_text("secret")
    try:
        result = await read_tool.run(
            ReadInput(path=str(outside)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert "outside allowed roots" in _error_message(result)
    finally:
        outside.unlink(missing_ok=True)


async def test_nonexistent_file_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext, read_tool: ReadTool
) -> None:
    result = await read_tool.run(
        ReadInput(path=str(tmp_path / "nope.txt")), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "does not exist" in _error_message(result)


async def test_char_cap_truncates_oversized_read(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    """A window past the char cap is truncated with a notice, not refused."""
    tool = ReadTool(
        redactor=NullRedactor(),
        max_read_chars=50,  # tiny cap to force truncation
    )
    f = tmp_path / "big.txt"
    f.write_text("\n".join("x" * 20 for _ in range(20)) + "\n")
    result = await tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    assert isinstance(result, ReadResult)
    assert result.truncated
    assert result.total_lines == 20
    assert "Read truncated" in result.content
    # The content before the notice stays within the cap.
    body = result.content.split("\n\n[Read truncated")[0]
    assert len(body) <= 50


async def test_size_gate_refuses_huge_file(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    """A file past ``max_file_bytes`` is refused outright (too large to open)."""
    tool = ReadTool(redactor=NullRedactor(), max_file_bytes=100)
    f = tmp_path / "huge.txt"
    f.write_text("x" * 500)
    result = await tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    assert "too large to open" in _error_message(result)


async def test_read_without_ctx_refused() -> None:
    """Stateless tools refuse to run without a wired backend."""
    tool = ReadTool(redactor=NullRedactor())
    result = await tool.run(ReadInput(path="/tmp/x.txt"))
    assert "ctx.file_backend" in _error_message(result)


# ---------------------------------------------------------------------------
# Redaction integration
# ---------------------------------------------------------------------------


async def test_default_redactor_masks_aws_key(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    tool = ReadTool(redactor=DefaultSecretRedactor())
    f = tmp_path / "secret.py"
    f.write_text("AKIAIOSFODNN7EXAMPLE\n")
    result = await tool.run(ReadInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    assert isinstance(result, ReadResult)
    assert "AKIAIOSFODNN7EXAMPLE" not in result.content
    assert "<REDACTED:AWS_ACCESS_KEY>" in result.content
