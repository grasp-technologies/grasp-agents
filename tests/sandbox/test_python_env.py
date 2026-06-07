"""
Configuring the code-interpreter Python interpreter (the ``python=`` knob).

Kernel-free: asserts the launch argv + PATH wiring rather than spawning a
kernel, so these run under the default sandbox.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.sandbox import local_environment
from grasp_agents.sandbox.config import EnvironmentConfig
from grasp_agents.sandbox.kernel import CellOutput, KernelCapable
from grasp_agents.sandbox.local.environment import _dist_name
from grasp_agents.sandbox.local.exec import LocalExecBackend

if TYPE_CHECKING:
    from collections.abc import Mapping


def _backend(tmp_path: Path, **kw: Any) -> LocalExecBackend:
    backend = local_environment(allowed_roots=[tmp_path], **kw).exec_backend
    assert isinstance(backend, LocalExecBackend)
    return backend


def test_default_kernel_uses_sys_executable(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend._kernel_launch_argv("cf")[0] == sys.executable
    assert backend._python_path_prepend is None  # no synthetic PATH prefix


def test_configured_kernel_interpreter(tmp_path: Path) -> None:
    backend = _backend(tmp_path, python=sys.executable)
    argv = backend._kernel_launch_argv("cf")
    assert argv[0] == sys.executable  # kept as-is (no symlink resolution)
    assert list(argv[1:]) == ["-m", "ipykernel_launcher", "-f", "cf"]


def test_configured_interpreter_leads_path(tmp_path: Path) -> None:
    backend = _backend(tmp_path, python=sys.executable)
    merged: Mapping[str, str] = backend._merged_env(None)
    bin_dir = str(Path(sys.executable).parent)
    assert merged["PATH"].split(os.pathsep)[0] == bin_dir


def test_missing_interpreter_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        local_environment(
            allowed_roots=[tmp_path], python="not-a-real-interpreter-xyz123"
        )


def test_config_python_field(tmp_path: Path) -> None:
    cfg = EnvironmentConfig.model_validate(
        {
            "filesystem": {"allowed_roots": [str(tmp_path)]},
            "exec": {"python": sys.executable},
        }
    )
    backend = cfg.build().exec_backend
    assert isinstance(backend, LocalExecBackend)
    assert backend._kernel_launch_argv("cf")[0] == sys.executable


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("srt") is None, reason="srt not installed")
async def test_configured_interpreter_runs_under_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured interpreter is the one the (srt-confined) kernel runs as."""
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)
    backend = _backend(tmp_path, confinement="srt", python=sys.executable)
    assert isinstance(backend, KernelCapable)
    kernel = await backend.open_kernel(cwd=tmp_path)
    try:
        text = ""
        async for item in kernel.execute("import sys; print(sys.executable)"):
            if isinstance(item, CellOutput) and item.output_type == "stream":
                text += item.text or ""
    finally:
        await kernel.close()
    printed = text.strip().splitlines()
    assert printed, "kernel printed nothing"
    assert Path(printed[-1]).resolve() == Path(sys.executable).resolve()


# ---------------------------------------------------------------------------
# Package verification (local = verify, not install)
# ---------------------------------------------------------------------------


def test_dist_name_strips_specifiers() -> None:
    assert _dist_name("torch>=2.0") == "torch"
    assert _dist_name("datasets[audio]") == "datasets"
    assert _dist_name("scikit-learn") == "scikit-learn"
    assert _dist_name("pkg ; python_version < '3.12'") == "pkg"


def test_packages_present_passes(tmp_path: Path) -> None:
    # `pytest` is installed in the dev env → verification passes silently.
    local_environment(allowed_roots=[tmp_path], packages=["pytest"])


def test_packages_missing_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required packages"):
        local_environment(
            allowed_roots=[tmp_path], packages=["grasp-not-a-real-pkg-xyz"]
        )
