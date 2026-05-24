"""Tests for the memory system-prompt section + render_memory_block."""

from __future__ import annotations

import inspect

import pytest

from grasp_agents.memory import (
    memory_system_prompt_section,
    render_auto_memory_instructions,
    render_memory_block,
)


class TestRenderMemoryBlock:
    def test_none_index_returns_none(self) -> None:
        assert render_memory_block(None) is None

    def test_blank_index_returns_none(self) -> None:
        assert render_memory_block("   \n  ") is None

    def test_index_renders(self) -> None:
        out = render_memory_block("# header\nbody\n")
        assert out is not None
        # Wrapped in <memory-index> so the user's internal headings
        # don't clash with the surrounding section hierarchy.
        assert out.startswith("<memory-index>")
        assert out.rstrip().endswith("</memory-index>")
        assert "# header" in out
        assert "body" in out

    def test_freshness_warning_prepended(self) -> None:
        out = render_memory_block(
            "# idx",
            freshness_warning=(
                "<system-reminder>This memory is 14 days old"
                " — verify before acting.</system-reminder>"
            ),
        )
        assert out is not None
        assert "<system-reminder>" in out
        # Freshness warning lands BEFORE the <memory-index> block so the
        # block content stays pure file content — important because the
        # agent may copy that content verbatim when calling Edit.
        assert out.index("<system-reminder>") < out.index("<memory-index>")
        assert out.index("<memory-index>") < out.index("# idx")


class TestMemorySection:
    def test_section_name_is_memory(self) -> None:
        assert memory_system_prompt_section.name == "memory"

    @pytest.mark.anyio
    async def test_compute_no_ctx_returns_none(self) -> None:
        result = memory_system_prompt_section.compute(ctx=None, exec_id="e")
        if inspect.isawaitable(result):
            result = await result
        assert result is None


class TestRenderAutoMemoryInstructions:
    def test_emits_taxonomy_unconditionally(self) -> None:
        # The instructions describe the substrate (taxonomy, format
        # rule, edit loop) — they do NOT depend on which tools the
        # agent has wired. The system prompt stays cache-stable.
        out = render_auto_memory_instructions()
        assert "# Memory" in out
        assert "`user`" in out
        assert "`feedback`" in out
        assert "`project`" in out
        assert "`reference`" in out

    def test_explains_memory_md_format(self) -> None:
        out = render_auto_memory_instructions()
        assert "MEMORY.md" in out
        assert "[name](file.md)" in out

    def test_uses_generic_file_tools(self) -> None:
        # CC-aligned: no specialized save_memory / load_memory / etc.
        # Authoring uses Read / Write / Edit on the memdir directly.
        out = render_auto_memory_instructions()
        assert "Read" in out
        assert "Write" in out
        assert "Edit" in out
        # Old specialized-tool names must NOT leak into the prompt —
        # they don't exist anymore.
        assert "save_memory" not in out
        assert "load_memory" not in out
        assert "list_memories" not in out
        assert "update_memory_index" not in out
        assert "delete_memory" not in out

    def test_memdir_path_substitution(self) -> None:
        # When the caller passes the memdir path, it lands in the lead
        # paragraph so the agent knows where to author.
        out = render_auto_memory_instructions(memdir="/tmp/memdir")
        assert "`/tmp/memdir`" in out
        # And the placeholder phrasing is replaced.
        assert "rooted at the memdir." not in out

    def test_memdir_unset_keeps_placeholder(self) -> None:
        out = render_auto_memory_instructions()
        # No path supplied → the lead paragraph keeps the generic
        # "rooted at the memdir" phrasing so the prompt still reads.
        assert "rooted at the memdir." in out

    def test_selector_adds_per_turn_note(self) -> None:
        out = render_auto_memory_instructions(has_selector=True)
        assert "surfaced into each turn" in out

    def test_no_selector_omits_per_turn_note(self) -> None:
        out = render_auto_memory_instructions(has_selector=False)
        assert "surfaced into each turn" not in out

    def test_verify_warning_present(self) -> None:
        out = render_auto_memory_instructions()
        # Per-turn verification reminder — keep before-acting cue.
        assert "verify before asserting" in out

    def test_save_discipline_present(self) -> None:
        # Inspired by CC's ``## What NOT to save`` — the substrate spells
        # out the failure modes we hit before (saved fabricated bodies,
        # duplicate near-identical topics).
        out = render_auto_memory_instructions()
        assert "Don't save" in out
        assert "Guesswork" in out
        assert "near-duplicate" in out

    def test_index_discipline_present(self) -> None:
        # The index is a *map* (links), not a store: one-line entries
        # capped at ~150 chars, no topic bodies inlined, watch length.
        # The "never inline ..." phrasing may wrap across a newline, so
        # normalize whitespace before matching.
        out = " ".join(render_auto_memory_instructions().split())
        assert "never inline a topic body" in out.lower()
        assert "~150" in out
        assert "truncated" in out

    def test_search_fallback_present(self) -> None:
        out = render_auto_memory_instructions()
        assert "search the memdir directly" in out
