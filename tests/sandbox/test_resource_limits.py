"""
local_environment(limits=...) — convenient per-command resource ceilings
without having to construct a ProcessSupervisor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.sandbox.local.environment import local_environment
from grasp_agents.sandbox.local.supervisor import ProcessSupervisor


def test_limits_build_a_supervisor(tmp_path: Path) -> None:
    env = local_environment(
        allowed_roots=[tmp_path],
        limits={"max_memory_mb": 256, "cpu_timeout": 5.0, "max_file_size_mb": 10},
    )
    backend = env.exec_backend
    assert backend is not None
    limits = backend._supervisor.limits
    assert limits.max_memory_mb == 256
    assert limits.cpu_timeout == 5.0
    assert limits.max_file_size_mb == 10
    # non-resource ceilings keep their defaults
    assert limits.overall_timeout == 600.0


def test_no_limits_means_no_resource_ceilings(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    backend = env.exec_backend
    assert backend is not None
    assert backend._supervisor.limits.max_memory_mb is None
    assert backend._supervisor.limits.cpu_timeout is None


def test_limits_and_supervisor_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        local_environment(
            allowed_roots=[tmp_path],
            limits={"max_memory_mb": 256},
            supervisor=ProcessSupervisor(),
        )
