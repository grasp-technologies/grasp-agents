"""
Tests for ``LocalExecSession`` — the persistent (stateful) shell behind
``ExecSession`` / ``SessionCapable``: state persists across commands, stdout and
stderr stay separate, exit codes propagate, and a timeout closes the session.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from grasp_agents.sandbox import local_environment
from grasp_agents.sandbox.exec_backend import (
    ExecResult,
    ExecSession,
    SessionCapable,
)

pytestmark = pytest.mark.asyncio


def _seatbelt_can_apply() -> bool:
    """True only on a real (non-nested) macOS host where sandbox-exec applies."""
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return False
    proc = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


_CAN_APPLY = _seatbelt_can_apply()


async def _open(tmp_path: Path) -> ExecSession:
    env = local_environment(allowed_roots=[tmp_path])
    backend = env.exec_backend
    assert isinstance(backend, SessionCapable)
    return await backend.open_session(cwd=tmp_path)


async def _collect(
    session: ExecSession, command: str, **kw: float
) -> tuple[str, str, ExecResult]:
    out: list[str] = []
    err: list[str] = []
    terminal: ExecResult | None = None
    async for item in session.run(command, **kw):
        if isinstance(item, ExecResult):
            terminal = item
        else:
            (out if item.stream == "stdout" else err).append(item.data)
    assert terminal is not None
    return "".join(out), "".join(err), terminal


async def test_local_backend_is_session_capable(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    assert isinstance(env.exec_backend, SessionCapable)


async def test_cwd_persists_across_commands(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    session = await _open(tmp_path)
    try:
        await _collect(session, "cd sub")
        out, _, result = await _collect(session, "pwd")
        assert result.returncode == 0
        assert out.strip().endswith("sub")
    finally:
        await session.close()


async def test_env_and_shell_vars_persist(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    try:
        await _collect(session, "export FOO=bar")
        await _collect(session, "MYVAR=42")
        out, _, result = await _collect(session, "echo $FOO $MYVAR")
        assert result.returncode == 0
        assert out.strip() == "bar 42"
    finally:
        await session.close()


async def test_stdout_and_stderr_separate(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    try:
        out, err, result = await _collect(session, "echo OUT; echo ERR 1>&2")
        assert result.returncode == 0
        assert out.strip() == "OUT"
        assert err.strip() == "ERR"
    finally:
        await session.close()


async def test_exit_code_propagates(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    try:
        _, _, ok = await _collect(session, "true")
        assert ok.returncode == 0
        _, _, fail = await _collect(session, "false")
        assert fail.returncode == 1
        # A non-zero exit does not break the session.
        out, _, after = await _collect(session, "echo still-here")
        assert after.returncode == 0
        assert out.strip() == "still-here"
    finally:
        await session.close()


async def test_no_marker_leakage_in_output(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    try:
        out, _, result = await _collect(session, "printf 'no-newline-tail'")
        assert result.returncode == 0
        assert out == "no-newline-tail"
        assert "__GRASP_" not in out
    finally:
        await session.close()


async def test_timeout_interrupts_command_but_keeps_session(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    try:
        _, _, result = await _collect(session, "sleep 5", timeout=0.3)
        assert result.timed_out
        assert result.reason == "overall_timeout"
        # SIGINT terminated the command; the trapped shell survived.
        assert not session.closed
        out, _, after = await _collect(session, "echo alive")
        assert after.returncode == 0
        assert out.strip() == "alive"
    finally:
        await session.close()


async def test_uninterruptible_command_closes_session(tmp_path: Path) -> None:
    from grasp_agents.sandbox.local_exec import LocalExecBackend
    from grasp_agents.sandbox.policy import SandboxPolicy
    from grasp_agents.sandbox.supervisor import ProcessSupervisor, SupervisorLimits

    # A short grace so the close fallback fires quickly.
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,)),
        supervisor=ProcessSupervisor(SupervisorLimits(kill_grace_period=0.3)),
    )
    session = await backend.open_session(cwd=tmp_path)
    try:
        # The command makes the shell ignore SIGINT, so SIGINT cannot stop it;
        # once the grace window overruns, the whole session is force-closed.
        _, _, result = await _collect(session, "trap '' INT; sleep 5", timeout=0.3)
        assert result.timed_out
        assert session.closed
        with pytest.raises(RuntimeError, match="closed"):
            await _collect(session, "echo nope")
    finally:
        await session.close()


async def test_close_is_idempotent_and_marks_closed(tmp_path: Path) -> None:
    session = await _open(tmp_path)
    await _collect(session, "echo hi")
    await session.close()
    assert session.closed
    await session.close()  # no error on second close


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_session_persists_state_under_confinement(
    tmp_path: Path,
) -> None:
    (tmp_path / "sub").mkdir()
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    backend = env.exec_backend
    assert isinstance(backend, SessionCapable)
    assert backend.name == "seatbelt"
    # The persistent shell runs *inside* sandbox-exec and keeps state across
    # commands (the confinement wrapper is applied to the shell itself).
    session = await backend.open_session(cwd=tmp_path)
    try:
        await _collect(session, "cd sub")
        out, _, result = await _collect(session, "pwd")
        assert result.returncode == 0
        assert result.backend == "seatbelt"
        assert out.strip().endswith("sub")
    finally:
        await session.close()
