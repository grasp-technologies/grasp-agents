"""
Integration tests for ``LocalKernel`` (the ``KernelSession`` over a local
``jupyter_client`` kernel) and ``LocalExecBackend.open_kernel`` (``KernelCapable``).

These spawn a real kernel — its ZMQ channels need loopback networking, which the
default command sandbox blocks — so they are gated behind the ``integration``
marker and run unsandboxed via ``uv run pytest -m integration``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from grasp_agents.sandbox import local_environment
from grasp_agents.sandbox.kernel import (
    CellOutput,
    CellResult,
    KernelCapable,
    KernelSession,
)
from grasp_agents.sandbox.policy import NetworkPolicy

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# A 1x1 transparent PNG — lets us assert image capture without matplotlib.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def _open(tmp_path: Path) -> KernelSession:
    env = local_environment(allowed_roots=[tmp_path])
    backend = env.exec_backend
    assert isinstance(backend, KernelCapable)
    return await backend.open_kernel(cwd=tmp_path)


async def _collect(
    kernel: KernelSession, code: str, **kw: float
) -> tuple[list[CellOutput], CellResult]:
    outputs: list[CellOutput] = []
    result: CellResult | None = None
    async for item in kernel.execute(code, **kw):
        if isinstance(item, CellResult):
            result = item
        else:
            outputs.append(item)
    assert result is not None
    return outputs, result


async def test_backend_is_kernel_capable(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    assert isinstance(env.exec_backend, KernelCapable)


async def test_execute_stdout(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        outputs, result = await _collect(kernel, "print('hello world')")
        streams = [o.text for o in outputs if o.output_type == "stream"]
        assert any("hello world" in (t or "") for t in streams)
        assert result.status == "ok"
        assert result.execution_count == 1
    finally:
        await kernel.close()


async def test_execute_result_value(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        outputs, result = await _collect(kernel, "21 * 2")
        results = [o for o in outputs if o.output_type == "execute_result"]
        assert results
        assert "42" in str(results[0].data and results[0].data.get("text/plain"))
        assert result.status == "ok"
    finally:
        await kernel.close()


async def test_state_persists_across_executes(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        await _collect(kernel, "x = 41")
        outputs, result = await _collect(kernel, "x + 1")
        results = [o for o in outputs if o.output_type == "execute_result"]
        assert "42" in str(results[0].data and results[0].data.get("text/plain"))
        assert result.status == "ok"
    finally:
        await kernel.close()


async def test_image_output_captured(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        code = (
            "import base64\n"
            "from IPython.display import Image, display\n"
            f"display(Image(data=base64.b64decode('{_PNG_B64}'), format='png'))\n"
        )
        outputs, result = await _collect(kernel, code)
        images = [
            o
            for o in outputs
            if o.output_type == "display_data" and o.data and "image/png" in o.data
        ]
        assert images, f"expected an image/png output, got {outputs}"
        assert result.status == "ok"
    finally:
        await kernel.close()


async def test_error_output(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        outputs, result = await _collect(kernel, "1 / 0")
        errors = [o for o in outputs if o.output_type == "error"]
        assert errors
        assert errors[0].ename == "ZeroDivisionError"
        assert result.status == "error"
    finally:
        await kernel.close()


async def test_restart_clears_state(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        await _collect(kernel, "y = 123")
        await kernel.restart()
        outputs, result = await _collect(kernel, "y")
        errors = [o for o in outputs if o.output_type == "error"]
        assert errors
        assert errors[0].ename == "NameError"
        assert result.status == "error"
    finally:
        await kernel.close()


async def test_timeout_interrupts(tmp_path: Path) -> None:
    kernel = await _open(tmp_path)
    try:
        _, result = await _collect(
            kernel, "import time\nwhile True:\n    time.sleep(0.05)", timeout=3.0
        )
        # Interrupt lands as KeyboardInterrupt (error) or, if ignored, an abort.
        assert result.timed_out or result.status in {"error", "abort"}
    finally:
        await kernel.close()


async def test_crash_restart_flags_reset(tmp_path: Path) -> None:
    # A kernel killed between cells is replaced in place on the next execute;
    # the replacement must work AND report the reset (REPL state was lost).
    kernel = await _open(tmp_path)
    try:
        await _collect(kernel, "x = 1")
        proc = kernel._proc
        assert proc is not None
        proc.kill()
        await proc.wait()

        outputs, result = await _collect(kernel, "print('alive')")
        assert result.status == "ok"
        assert any(o.text and "alive" in o.text for o in outputs)
        assert kernel.take_reset() is True
        assert kernel.take_reset() is False  # cleared after taking
    finally:
        await kernel.close()


# ---------------------------------------------------------------------------
# Confinement — the kernel launches under the backend's sandbox wrapper. The
# kernel's loopback ZMQ must be permitted by the profile, so these document
# which (confinement, network) combinations actually allow a kernel.
# ---------------------------------------------------------------------------


def _seatbelt_can_apply() -> bool:
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return False
    proc = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


@pytest.mark.skipif(not _seatbelt_can_apply(), reason="seatbelt unavailable / nested")
async def test_kernel_under_seatbelt(tmp_path: Path) -> None:
    # Seatbelt needs loopback for the kernel's ZMQ; network=ALL permits it.
    env = local_environment(
        allowed_roots=[tmp_path], confinement="seatbelt", network=NetworkPolicy.ALL
    )
    backend = env.exec_backend
    assert isinstance(backend, KernelCapable)
    kernel = await backend.open_kernel(cwd=tmp_path)
    try:
        outputs, result = await _collect(kernel, "print('confined hello')")
        assert result.status == "ok"
        assert backend.name == "seatbelt"
        streams = [o.text for o in outputs if o.output_type == "stream"]
        assert any("confined hello" in (t or "") for t in streams)
    finally:
        await kernel.close()


@pytest.mark.skipif(shutil.which("srt") is None, reason="srt not installed")
async def test_kernel_under_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # srt confines the kernel: allowLocalBinding (set in build_srt_settings) lets
    # it bind its loopback ZMQ ports while external egress stays allowlist-gated.
    # srt forces TMPDIR for the sandboxed process to $CLAUDE_CODE_TMPDIR ||
    # /tmp/claude (auto-allowing /tmp/claude). This suite runs inside Claude Code,
    # which sets CLAUDE_CODE_TMPDIR to a path outside the sandbox; drop it so srt
    # falls back to its own writable temp. (Not needed in a normal shell.)
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)
    env = local_environment(
        allowed_roots=[tmp_path],
        confinement="srt",
        network=NetworkPolicy.ALLOWLIST,
    )
    backend = env.exec_backend
    assert isinstance(backend, KernelCapable)
    kernel = await backend.open_kernel(cwd=tmp_path)
    try:
        outputs, result = await _collect(kernel, "print('srt hello')")
        assert result.status == "ok"
        assert backend.name == "srt"
        streams = [o.text for o in outputs if o.output_type == "stream"]
        assert any("srt hello" in (t or "") for t in streams)
    finally:
        await kernel.close()
