"""Tests for the memory system-prompt section + render_memory_block."""

from __future__ import annotations

import inspect

import pytest

from grasp_agents.memory import (
    MemorySnapshot,
    memory_system_prompt_section,
    render_memory_block,
)


class TestRenderMemoryBlock:
    def test_empty_snapshot_returns_none(self) -> None:
        assert render_memory_block(MemorySnapshot()) is None

    def test_blank_index_returns_none(self) -> None:
        assert render_memory_block(MemorySnapshot(index="   \n  ")) is None

    def test_index_renders(self) -> None:
        snap = MemorySnapshot(index="# header\nbody\n")
        out = render_memory_block(snap)
        assert out is not None
        assert "# memory" in out
        assert "header" in out
        assert "body" in out

    def test_freshness_warning_prepended(self) -> None:
        snap = MemorySnapshot(
            index="# idx",
            index_freshness_warning="<system-reminder>This memory is 14 days old"
            " — verify before acting.</system-reminder>",
        )
        out = render_memory_block(snap)
        assert out is not None
        assert "<system-reminder>" in out
        assert out.index("<system-reminder>") < out.index("# idx")


class TestMemorySection:
    def test_section_name_is_memory(self) -> None:
        assert memory_system_prompt_section.name == "memory"

    @pytest.mark.anyio
    async def test_compute_no_ctx_returns_none(self) -> None:
        result = memory_system_prompt_section.compute(ctx=None, exec_id="e")
        if inspect.isawaitable(result):
            result = await result
        assert result is None
