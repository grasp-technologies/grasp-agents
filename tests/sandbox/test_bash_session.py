"""
Tests for the ``BashSession`` tool: a persistent shell session where ``cd`` /
environment / shell variables carry across calls (backed by an
``ExecSession`` from a ``SessionCapable`` backend). The session lives on the
agent loop's ``AgentContext``; ``BashSession`` reads it from there per call,
so a tool instance holds no state of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.bash_common import BashInput, ShellState
from grasp_agents.tools.bash_session import BashSession, BashSessionHolder
from grasp_agents.tools.file_edit.session_state import FileEditSessionState

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _ctx(tmp_path: Path) -> RunContext[None]:
    env = local_environment(allowed_roots=[tmp_path])
    return RunContext(environment=env)


def _agent_ctx(holder: BashSessionHolder) -> AgentContext:
    """A minimal AgentContext carrying ``holder`` — the field BashSession reads."""
    transcript = LLMAgentTranscript()
    return AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=FileEditSessionState(),
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
        session_holder=holder,
        shell_state=ShellState(),
    )


async def test_shell_keeps_state_across_calls(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    holder = BashSessionHolder()
    tool = BashSession()
    agent_ctx = _agent_ctx(holder)
    ctx = _ctx(tmp_path)
    try:
        r1 = await tool._run(
            BashInput(command="cd sub && export FOO=bar"),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
        assert r1.returncode == 0
        # cd and the env var both persist into the next call (same shell).
        r2 = await tool._run(
            BashInput(command="pwd && echo $FOO"), ctx=ctx, agent_ctx=agent_ctx
        )
        assert r2.returncode == 0
        assert "sub" in r2.stdout
        assert "bar" in r2.stdout
    finally:
        await holder.close()


async def test_shell_without_holder_is_ephemeral(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    tool = BashSession()  # no AgentContext in scope → a throwaway session per call
    ctx = _ctx(tmp_path)
    await tool._run(BashInput(command="cd sub"), ctx=ctx)
    r = await tool._run(BashInput(command="pwd"), ctx=ctx)
    # No persistence without a holder: the earlier cd did not carry over.
    assert not r.stdout.strip().endswith("sub")


async def test_shell_cwd_is_one_off(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    holder = BashSessionHolder()
    tool = BashSession()
    agent_ctx = _agent_ctx(holder)
    ctx = _ctx(tmp_path)
    try:
        # `cwd` runs this call in a subshell — the session's own cwd is untouched.
        r1 = await tool._run(
            BashInput(command="pwd", cwd=str(tmp_path / "sub")),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
        assert r1.stdout.strip().endswith("sub")
        r2 = await tool._run(BashInput(command="pwd"), ctx=ctx, agent_ctx=agent_ctx)
        assert not r2.stdout.strip().endswith("sub")
    finally:
        await holder.close()


async def test_shell_stdout_and_stderr_separate(tmp_path: Path) -> None:
    holder = BashSessionHolder()
    tool = BashSession()
    agent_ctx = _agent_ctx(holder)
    try:
        res = await tool._run(
            BashInput(command="echo OUT; echo ERR 1>&2"),
            ctx=_ctx(tmp_path),
            agent_ctx=agent_ctx,
        )
        assert res.returncode == 0
        assert "OUT" in res.stdout
        assert "ERR" in res.stderr
    finally:
        await holder.close()


async def test_shell_blocks_leading_sleep(tmp_path: Path) -> None:
    tool = BashSession()
    with pytest.raises(ValueError, match="leading `sleep`"):
        await tool._run(BashInput(command="sleep 5"), ctx=_ctx(tmp_path))
