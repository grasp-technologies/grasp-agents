"""
Tests for the ``Shell`` tool: a persistent shell session where ``cd`` /
environment / shell variables carry across calls (backed by an
``ExecSession`` from a ``SessionCapable`` backend).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.bash_common import BashInput
from grasp_agents.tools.bash_session import BashSession, BashSessionHolder

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _ctx(tmp_path: Path) -> RunContext[None]:
    env = local_environment(allowed_roots=[tmp_path])
    return RunContext(environment=env)


async def test_shell_keeps_state_across_calls(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    holder = BashSessionHolder()
    tool = BashSession(holder=holder)
    ctx = _ctx(tmp_path)
    try:
        r1 = await tool._run(BashInput(command="cd sub && export FOO=bar"), ctx=ctx)
        assert r1.returncode == 0
        # cd and the env var both persist into the next call (same shell).
        r2 = await tool._run(BashInput(command="pwd && echo $FOO"), ctx=ctx)
        assert r2.returncode == 0
        assert "sub" in r2.stdout
        assert "bar" in r2.stdout
    finally:
        await holder.close()


async def test_shell_without_holder_is_ephemeral(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    tool = BashSession()  # no holder in scope → a throwaway session per call
    ctx = _ctx(tmp_path)
    await tool._run(BashInput(command="cd sub"), ctx=ctx)
    r = await tool._run(BashInput(command="pwd"), ctx=ctx)
    # No persistence without a holder: the earlier cd did not carry over.
    assert not r.stdout.strip().endswith("sub")


async def test_shell_cwd_is_one_off(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    holder = BashSessionHolder()
    tool = BashSession(holder=holder)
    ctx = _ctx(tmp_path)
    try:
        # `cwd` runs this call in a subshell — the session's own cwd is untouched.
        r1 = await tool._run(
            BashInput(command="pwd", cwd=str(tmp_path / "sub")), ctx=ctx
        )
        assert r1.stdout.strip().endswith("sub")
        r2 = await tool._run(BashInput(command="pwd"), ctx=ctx)
        assert not r2.stdout.strip().endswith("sub")
    finally:
        await holder.close()


async def test_shell_stdout_and_stderr_separate(tmp_path: Path) -> None:
    holder = BashSessionHolder()
    tool = BashSession(holder=holder)
    try:
        res = await tool._run(
            BashInput(command="echo OUT; echo ERR 1>&2"), ctx=_ctx(tmp_path)
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
