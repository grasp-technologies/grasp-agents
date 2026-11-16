"""
Live E2B checks of the sandbox lifecycle that the fakes in ``_e2b_fakes.py``
cannot prove: what E2B itself does on a pause, a snapshot, a restore, a resume
and a kill.

Needs ``e2b`` installed, ``E2B_API_KEY`` and network; run unsandboxed:
``uv run pytest -m integration tests/sandbox/test_e2b_live_lifecycle.py``.

Every sandbox a test creates carries a metadata tag of its own (key
:data:`TAG_KEY`), and the test kills every sandbox with that tag, and deletes
every snapshot it took, in a ``finally``. Sandboxes also get a short keep-alive
(:data:`KEEP_ALIVE`), so one a failed cleanup leaves behind dies soon; find
leftovers by listing sandboxes with that metadata key.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.sandbox import e2b_environment

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from grasp_agents.sandbox.e2b.environment import E2BEnvironment

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_HAS_E2B = importlib.util.find_spec("e2b") is not None
# bool, not the key itself — a raw key in a module global can surface in logs.
_HAS_E2B_KEY = bool(os.getenv("E2B_API_KEY"))
_live = pytest.mark.skipif(
    not (_HAS_E2B and _HAS_E2B_KEY), reason="needs e2b installed + E2B_API_KEY"
)

WS = "/home/user/workspace"
TAG_KEY = "grasp-agents-live-test"
KEEP_ALIVE = 180  # seconds; a leftover sandbox dies on its own after this


class LiveRun:
    """The sandboxes and snapshots one live test creates, and their cleanup."""

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex
        self.snapshots: list[str] = []

    def env(self, **kwargs: Any) -> E2BEnvironment:
        return e2b_environment(
            allowed_roots=[WS],
            metadata={TAG_KEY: self.tag},
            sandbox_timeout=KEEP_ALIVE,
            **kwargs,
        )

    async def create_sandbox(self) -> Any:
        """A sandbox outside any environment, tagged like the environment's."""
        from e2b import AsyncSandbox

        return await AsyncSandbox.create(
            metadata={TAG_KEY: self.tag}, timeout=KEEP_ALIVE
        )

    async def tagged(self) -> set[str]:
        """Ids of this test's sandboxes E2B still has, running or paused."""
        from e2b import (
            AsyncSandbox,
            SandboxQuery,
        )

        pages = AsyncSandbox.list(query=SandboxQuery(metadata={TAG_KEY: self.tag}))
        ids: set[str] = set()
        while pages.has_next:
            ids.update(info.sandbox_id for info in await pages.next_items())
        return ids

    async def cleanup(self) -> None:
        from e2b import AsyncSandbox

        for sandbox_id in await self.tagged():
            with contextlib.suppress(Exception):
                await AsyncSandbox.kill(sandbox_id)
        for snapshot_id in self.snapshots:
            with contextlib.suppress(Exception):
                await AsyncSandbox.delete_snapshot(snapshot_id)


async def until(check: Callable[[], Awaitable[bool]], *, timeout: float = 30.0) -> None:
    """Poll ``check`` until it holds: E2B's listing lags behind its kills."""
    deadline = time.monotonic() + timeout
    while not await check():
        if time.monotonic() > deadline:
            raise AssertionError(f"still false after {timeout:g}s")
        await asyncio.sleep(1.0)


def live_id(env: E2BEnvironment) -> str:
    sandbox = env._holder.sandbox
    assert sandbox is not None
    return str(sandbox.sandbox_id)


# --- resume ------------------------------------------------------------------


@_live
async def test_resuming_another_sandbox_kills_the_previous_one() -> None:
    run = LiveRun()
    try:
        other = await run.create_sandbox()
        async with run.env(lazy=True) as env:
            await env.exec_backend.execute("true")
            first = live_id(env)

            await env.resume(str(other.sandbox_id))

            assert (await env.exec_backend.execute("true")).returncode == 0
            assert live_id(env) == other.sandbox_id

            async def first_is_gone() -> bool:
                return first not in await run.tagged()

            await until(first_is_gone)
    finally:
        await run.cleanup()
