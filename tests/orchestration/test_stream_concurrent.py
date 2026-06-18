"""
stream_concurrent: the ``max_concurrency=1`` serial mode (drains each
generator fully before the next, with the same per-stream error isolation).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from grasp_agents.utils.streaming import stream_concurrent


async def _gen(items: list[str]) -> AsyncIterator[str]:
    for it in items:
        await asyncio.sleep(0)  # yield control so interleaving is *possible*
        yield it


@pytest.mark.asyncio
async def test_max_concurrency_1_does_not_interleave() -> None:
    gens = [_gen(["0a", "0b", "0c"]), _gen(["1a", "1b", "1c"])]
    order = [idx async for idx, _ in stream_concurrent(gens, max_concurrency=1)]
    # One generator is fully drained before the other starts (no interleaving).
    assert order in ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0])


@pytest.mark.asyncio
async def test_serial_mode_isolates_per_stream_errors() -> None:
    async def boom() -> AsyncIterator[str]:
        yield "x"
        raise RuntimeError("boom")

    merged = stream_concurrent([boom(), _gen(["1a"])], max_concurrency=1)
    items = [item async for _, item in merged]
    assert "1a" in items  # the other stream still runs
    assert [e.index for e in merged.errors] == [0]
