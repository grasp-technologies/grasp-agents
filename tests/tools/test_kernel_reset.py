"""
Kernel reset detection (RunCell / RunPython).

Mirrors the persistent-shell-session reset: when the KernelHolder reopens a
kernel that was killed (e.g. a cell timeout), the REPL namespace (variables /
imports / definitions) is lost. The holder flags that so the kernel tool can
tell the model its kernel was restarted, instead of silently continuing in a
fresh one.
"""

from __future__ import annotations

import pytest

from grasp_agents.tools.notebook_exec import KernelHolder


class _FakeKernel:
    def __init__(self) -> None:
        self.closed = False
        self.context_id: str | None = None
        self.was_reset = False

    def take_reset(self) -> bool:
        was = self.was_reset
        self.was_reset = False
        return was

    async def close(self) -> None:
        self.closed = True


class _FakeKernelBackend:
    name = "fake"

    def __init__(self) -> None:
        self.opened: list[_FakeKernel] = []

    async def open_kernel(self, context_id: str | None = None) -> _FakeKernel:
        k = _FakeKernel()
        self.opened.append(k)
        return k


@pytest.mark.asyncio
async def test_first_open_is_not_a_reset() -> None:
    holder = KernelHolder()
    backend = _FakeKernelBackend()
    await holder.get(backend)
    assert holder.take_reset() is False


@pytest.mark.asyncio
async def test_reopen_after_close_flags_reset() -> None:
    holder = KernelHolder()
    backend = _FakeKernelBackend()
    k1 = await holder.get(backend)
    k1.closed = True
    k2 = await holder.get(backend)
    assert k2 is not k1
    assert holder.take_reset() is True
    assert holder.take_reset() is False  # cleared after taking


@pytest.mark.asyncio
async def test_alive_kernel_reused_without_reset() -> None:
    holder = KernelHolder()
    backend = _FakeKernelBackend()
    k1 = await holder.get(backend)
    k2 = await holder.get(backend)
    assert k2 is k1
    assert holder.take_reset() is False
    assert len(backend.opened) == 1


@pytest.mark.asyncio
async def test_in_place_kernel_restart_surfaces_through_holder() -> None:
    # A kernel that crashed between calls replaces its own process in place
    # (the holder never sees it as closed); the holder must pick up the
    # kernel's own reset flag.
    holder = KernelHolder()
    backend = _FakeKernelBackend()
    k1 = await holder.get(backend)
    k1.was_reset = True
    assert holder.take_reset() is True
    assert holder.take_reset() is False
    assert len(backend.opened) == 1
