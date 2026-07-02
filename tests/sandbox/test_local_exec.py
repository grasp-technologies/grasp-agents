"""
Unit tests for backend #1 (pure local): :class:`ProcessSupervisor`,
:class:`LocalExecBackend`, the :func:`local_environment` factory, and the
``Bash`` tool.

These spawn real ``/bin/sh`` subprocesses (no network) and exercise the
supervisor's timeout / output-cap / kill paths plus the ``SessionContext``
co-location guarantee (exec only via an environment).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.file_backend import LocalFileBackend, PathAccessError
from grasp_agents.sandbox import (
    ExecChunk,
    ExecResult,
    LocalEnvironment,
    ProcessSupervisor,
    SandboxPolicy,
    SupervisorLimits,
    TerminationReason,
    local_environment,
)
from grasp_agents.sandbox.local.exec import LocalExecBackend
from grasp_agents.sandbox.local.supervisor import ExecSpec
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.bash import Bash, BashInput, BashResult
from grasp_agents.types.events import ToolErrorInfo

pytestmark = pytest.mark.asyncio


def _backend(tmp_path: Path, **kw: Any) -> LocalExecBackend:
    policy = SandboxPolicy(allowed_roots=(tmp_path,))
    return LocalExecBackend(policy=policy, **kw)


async def _collect(
    stream: Any,
) -> tuple[list[ExecChunk], ExecResult]:
    chunks: list[ExecChunk] = []
    terminal: ExecResult | None = None
    async for item in stream:
        if isinstance(item, ExecChunk):
            chunks.append(item)
        else:
            terminal = item
    assert terminal is not None
    return chunks, terminal


# --- factory + co-location -------------------------------------------------


async def test_local_environment_co_location(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    assert isinstance(env, LocalEnvironment)
    assert env.file_backend.name == "local"
    assert env.exec_backend is not None
    assert env.exec_backend.name == "local"
    # Both surfaces share the one policy's roots.
    assert env.policy.allowed_roots == (tmp_path,)
    assert tmp_path in env.file_backend.allowed_roots


async def test_environment_sources_file_backend(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(environment=env)
    assert ctx.file_backend is env.file_backend
    assert ctx.exec_backend is env.exec_backend


async def test_file_only_ctx_has_no_exec(tmp_path: Path) -> None:
    ctx: SessionContext[Any] = SessionContext(
        file_backend=LocalFileBackend(allowed_roots=[tmp_path])
    )
    assert ctx.exec_backend is None


async def test_divergent_file_backend_and_environment_raises(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    with pytest.raises(ValueError, match="divergent standalone file_backend"):
        SessionContext(
            environment=env,
            file_backend=LocalFileBackend(allowed_roots=[tmp_path]),
        )


async def test_readonly_roots_are_readable_but_write_denied(tmp_path: Path) -> None:
    rw = tmp_path / "rw"
    ro = tmp_path / "ro"
    rw.mkdir()
    ro.mkdir()
    (ro / "ref.txt").write_text("reference")
    env = local_environment(allowed_roots=[rw], readonly_roots=[ro])
    backend = env.file_backend
    assert ro in backend.allowed_roots

    # Readable on the tool plane...
    resolved = await backend.validate_path(
        ro / "ref.txt", must_exist=True, access="read"
    )
    assert resolved == (ro / "ref.txt").resolve()
    # ...but write-denied (same rule Seatbelt applies on the exec plane).
    with pytest.raises(PathAccessError):
        await backend.validate_path(ro / "new.txt", must_exist=False, access="write")
    # The rw root is unaffected.
    await backend.validate_path(rw / "new.txt", must_exist=False, access="write")


async def test_backend_readonly_roots_param(tmp_path: Path) -> None:
    rw = tmp_path / "rw"
    ro = tmp_path / "ro"
    rw.mkdir()
    ro.mkdir()
    backend = LocalFileBackend(allowed_roots=[rw], readonly_roots=[ro])
    with pytest.raises(PathAccessError):
        await backend.validate_path(ro / "x.txt", must_exist=False, access="write")
    await backend.validate_path(ro, must_exist=True, access="read")


async def test_environment_is_async_context_manager(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    async with env as entered:
        assert entered is env
        assert entered.exec_backend is not None


# --- LocalExecBackend.execute ----------------------------------------------


async def test_execute_stdout(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("echo hello")
    assert res.stdout.strip() == "hello"
    assert res.stderr == ""
    assert res.returncode == 0
    assert res.reason is TerminationReason.EXIT
    assert res.backend == "local"
    assert not res.timed_out
    assert not res.truncated


async def test_execute_stderr(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("echo oops 1>&2")
    assert res.stdout == ""
    assert res.stderr.strip() == "oops"
    assert res.returncode == 0


async def test_execute_nonzero_exit(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("exit 3")
    assert res.returncode == 3
    assert res.reason is TerminationReason.EXIT


async def test_execute_default_cwd_is_first_root(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("pwd")
    assert Path(res.stdout.strip()).resolve() == tmp_path.resolve()


async def test_execute_cwd_override(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    res = await _backend(tmp_path).execute("pwd", cwd=sub)
    assert Path(res.stdout.strip()).resolve() == sub.resolve()


async def test_execute_cwd_outside_roots_raises(tmp_path: Path) -> None:
    with pytest.raises(PathAccessError):
        await _backend(tmp_path).execute("pwd", cwd=Path("/"))


async def test_execute_stdin(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("cat", stdin=b"piped-in\n")
    assert res.stdout.strip() == "piped-in"


async def test_execute_per_call_env(tmp_path: Path) -> None:
    res = await _backend(tmp_path).execute("echo $FOO", env={"FOO": "bar"})
    assert res.stdout.strip() == "bar"


async def test_policy_env_exposed(tmp_path: Path) -> None:
    policy = SandboxPolicy(allowed_roots=(tmp_path,), env={"GREETING": "hi"})
    backend = LocalExecBackend(policy=policy)
    res = await backend.execute("echo $GREETING")
    assert res.stdout.strip() == "hi"


async def test_no_inherit_host_env(tmp_path: Path) -> None:
    backend = _backend(tmp_path, inherit_host_env=False)
    # $HOME is in the host env but must not leak when inheritance is off; the
    # explicit policy/call env still applies.
    res = await backend.execute("echo [$HOME][$MARK]", env={"MARK": "x"})
    assert res.stdout.strip() == "[][x]"


# --- timeouts + caps + kill ------------------------------------------------


async def test_overall_timeout_kills(tmp_path: Path) -> None:
    start = time.monotonic()
    res = await _backend(tmp_path).execute("sleep 5", timeout=0.3)
    elapsed = time.monotonic() - start
    assert res.reason is TerminationReason.OVERALL_TIMEOUT
    assert res.timed_out
    assert elapsed < 3.0  # killed promptly, not after 5s


async def test_idle_timeout(tmp_path: Path) -> None:
    sup = ProcessSupervisor(SupervisorLimits(overall_timeout=5.0, idle_timeout=0.3))
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,)), supervisor=sup
    )
    res = await backend.execute("sleep 5")
    assert res.reason is TerminationReason.NO_OUTPUT_TIMEOUT
    assert res.timed_out


async def test_output_cap_truncates(tmp_path: Path) -> None:
    sup = ProcessSupervisor(SupervisorLimits(max_output_chars=5))
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,)), supervisor=sup
    )
    res = await backend.execute("echo abcdefghij")
    assert res.truncated
    assert len(res.stdout) <= 5


# --- resource limits + env scrub --------------------------------------------


async def test_cpu_timeout_kills_busy_loop(tmp_path: Path) -> None:
    sup = ProcessSupervisor(SupervisorLimits(overall_timeout=15.0, cpu_timeout=1))
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,)), supervisor=sup
    )
    start = time.monotonic()
    res = await backend.execute("python3 -c 'while True: pass'")
    # RLIMIT_CPU delivers SIGXCPU well before the 15s wall-clock ceiling.
    assert res.reason is TerminationReason.SIGNAL
    assert res.returncode < 0
    assert time.monotonic() - start < 10.0


async def test_max_file_size_limits_writes(tmp_path: Path) -> None:
    sup = ProcessSupervisor(SupervisorLimits(max_file_size_mb=1))
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,)), supervisor=sup
    )
    res = await backend.execute("head -c 5000000 /dev/zero > big.bin")
    assert res.returncode != 0  # SIGXFSZ / write error past the 1MB ceiling


async def test_env_scrub_removes_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "leaked")
    backend = LocalExecBackend(
        policy=SandboxPolicy(allowed_roots=(tmp_path,), env_scrub=("SECRET_*",))
    )
    res = await backend.execute("echo [$SECRET_TOKEN]")
    assert res.stdout.strip() == "[]"


async def test_env_scrub_keeps_explicit_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "leaked")
    backend = LocalExecBackend(
        policy=SandboxPolicy(
            allowed_roots=(tmp_path,),
            env_scrub=("SECRET_*",),
            env={"SECRET_OK": "kept"},  # deliberately set → never scrubbed
        )
    )
    res = await backend.execute("echo [$SECRET_TOKEN][$SECRET_OK]")
    assert res.stdout.strip() == "[][kept]"


# --- tool-plane carve-outs (FileBackend.validate_path) ----------------------


async def test_tool_plane_deny_read(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.mkdir()
    backend = LocalFileBackend(allowed_roots=[tmp_path], deny_read=[secret])
    with pytest.raises(PathAccessError, match="denied-read"):
        await backend.validate_path(secret / "x.txt", must_exist=False, access="read")
    # outside the denied region: still readable
    await backend.validate_path(tmp_path / "ok.txt", must_exist=False, access="read")


async def test_tool_plane_allow_read_overrides_deny_read(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    shared = secret / "shared"
    shared.mkdir(parents=True)
    backend = LocalFileBackend(
        allowed_roots=[tmp_path], deny_read=[secret], allow_read=[shared]
    )
    with pytest.raises(PathAccessError):
        await backend.validate_path(secret / "z.txt", must_exist=False, access="read")
    resolved = await backend.validate_path(
        shared / "y.txt", must_exist=False, access="read"
    )
    assert resolved == (shared / "y.txt").resolve()


async def test_tool_plane_deny_write_still_readable(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    backend = LocalFileBackend(allowed_roots=[tmp_path], deny_write=[protected])
    with pytest.raises(PathAccessError, match="write-protected"):
        await backend.validate_path(
            protected / "f.txt", must_exist=False, access="write"
        )
    # read of the same region is unaffected
    resolved = await backend.validate_path(protected, must_exist=True, access="read")
    assert resolved == protected.resolve()


# --- streaming -------------------------------------------------------------


async def test_stream_matches_execute(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    chunks, terminal = await _collect(backend.stream("echo streamed"))
    assembled = "".join(c.data for c in chunks if c.stream == "stdout")
    assert assembled.strip() == "streamed"
    assert terminal.returncode == 0
    assert terminal.reason is TerminationReason.EXIT
    # The terminal result carries metadata only; bytes came as chunks.
    assert terminal.stdout == ""


# --- supervisor spawn error -------------------------------------------------


async def test_spawn_error(tmp_path: Path) -> None:
    sup = ProcessSupervisor()
    spec = ExecSpec(
        argv=("/nonexistent/binary-xyz",),
        cwd=tmp_path,
        env={},
        backend="local",
    )
    _, terminal = await _collect(sup.run(spec))
    assert terminal.reason is TerminationReason.SPAWN_ERROR
    assert terminal.returncode == -1


# --- Bash tool -------------------------------------------------------------


async def test_bash_tool_happy_path(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(environment=env)
    result = await Bash().run(BashInput(command="echo hi"), ctx=ctx)
    assert isinstance(result, BashResult)
    assert result.stdout.strip() == "hi"
    assert result.returncode == 0
    assert result.reason == "exit"
    assert result.backend == "local"


async def test_bash_tool_requires_exec_backend() -> None:
    ctx: SessionContext[Any] = SessionContext()
    result = await Bash().run(BashInput(command="echo hi"), ctx=ctx)
    assert isinstance(result, ToolErrorInfo)
    assert "exec_backend" in result.error


async def test_bash_tool_timeout_clamped(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: SessionContext[Any] = SessionContext(environment=env)
    # `echo` first: a leading `sleep` is rejected by the blocked-pattern guard.
    result = await Bash(max_timeout=0.3).run(
        BashInput(command="echo go && sleep 5", timeout=100.0), ctx=ctx
    )
    assert isinstance(result, BashResult)
    assert result.timed_out
    assert result.reason == "overall_timeout"
