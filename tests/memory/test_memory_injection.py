"""Tests for the memory system-prompt section + render helpers."""

from __future__ import annotations

import inspect

import pytest

from grasp_agents.memory import (
    memory_system_prompt_section,
    render_memory_index,
    render_memory_instructions,
)


class TestRenderMemoryIndex:
    def test_none_index_returns_none(self) -> None:
        assert render_memory_index(None) is None

    def test_blank_index_returns_none(self) -> None:
        assert render_memory_index("   \n  ") is None

    def test_index_renders(self) -> None:
        out = render_memory_index("# header\nbody\n")
        assert out is not None
        # Wrapped in <memory-index> so the user's internal headings
        # don't clash with the surrounding section hierarchy.
        assert out.startswith("<memory-index>")
        assert out.rstrip().endswith("</memory-index>")
        assert "# header" in out
        assert "body" in out

    def test_freshness_warning_prepended(self) -> None:
        out = render_memory_index(
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

    def test_truncation_marker_surfaced(self) -> None:
        # When the index is truncated by line / byte caps, the renderer
        # appends a `[truncated]` marker so the model knows it's only
        # seeing a partial map and can act accordingly.
        out = render_memory_index("# idx\n- a\n- b\n", truncated=True)
        assert out is not None
        assert "truncated" in out.lower()
        # Marker is inside the block so it travels with the index text.
        assert out.index("truncated") > out.index("<memory-index>")


class TestMemorySection:
    def test_section_name_is_memory(self) -> None:
        assert memory_system_prompt_section.name == "memory"

    @pytest.mark.asyncio
    async def test_compute_no_ctx_returns_none(self) -> None:
        result = memory_system_prompt_section.compute(ctx=None, exec_id="e")
        if inspect.isawaitable(result):
            result = await result
        assert result is None


class TestRenderMemoryInstructions:
    def test_emits_taxonomy_unconditionally(self) -> None:
        # The instructions describe the substrate (taxonomy, format,
        # save/use loop) — independent of which tools the agent has wired.
        # The system prompt stays cache-stable.
        out = render_memory_instructions()
        assert "# Memory" in out
        assert "user" in out
        assert "feedback" in out
        assert "project" in out
        assert "reference" in out

    def test_explains_memory_md_format(self) -> None:
        out = render_memory_instructions()
        assert "MEMORY.md" in out
        assert "[name](file.md)" in out

    def test_no_specialized_memory_tools_referenced(self) -> None:
        # Authoring uses generic file tools on the memdir directly —
        # there are no save_memory / load_memory / etc. shortcuts.
        out = render_memory_instructions()
        assert "save_memory" not in out
        assert "load_memory" not in out
        assert "list_memories" not in out
        assert "delete_memory" not in out

    def test_memdir_path_substitution(self) -> None:
        # Caller-passed memdir lands in the lead paragraph so the agent
        # knows where to author.
        out = render_memory_instructions(memdir="/tmp/memdir")
        assert "`/tmp/memdir`" in out
        # Placeholder phrasing is replaced.
        assert "rooted at the memdir." not in out

    def test_memdir_unset_keeps_placeholder(self) -> None:
        out = render_memory_instructions()
        # No path supplied → keep the generic "rooted at the memdir"
        # phrasing so the prompt still reads.
        assert "rooted at the memdir." in out

    def test_selector_adds_per_turn_note(self) -> None:
        out = render_memory_instructions(has_selector=True)
        assert "surfaced into each turn" in out

    def test_no_selector_omits_per_turn_note(self) -> None:
        out = render_memory_instructions(has_selector=False)
        assert "surfaced into each turn" not in out

    def test_verify_warning_present(self) -> None:
        out = render_memory_instructions()
        # Per-turn verification reminder — recall isn't proof.
        assert "verify" in out.lower()

    def test_index_discipline_present(self) -> None:
        # The index is a map (links), not a store: concise entries
        # capped by line/byte limits, no topic bodies inlined, mention
        # of the truncation cap.
        out = " ".join(render_memory_instructions().split())
        assert "never inline topic bodies into it" in out.lower()
        assert "keep entries concise" in out.lower()
        assert "truncated" in out

    def test_no_coding_agent_assumptions(self) -> None:
        # The framework is general-purpose. Memory instructions must
        # not assume the agent is a coding agent (grep, git log,
        # CLAUDE.md / GRASP.md are not universal).
        out = render_memory_instructions().lower()
        assert "grep" not in out
        assert "git log" not in out
        assert "claude.md" not in out
        assert "grasp.md" not in out

    def test_template_variables_substituted(self) -> None:
        # No raw placeholder strings should leak through — every
        # template var the prompt references must be filled in.
        out = render_memory_instructions(memdir="/tmp/memdir")
        for token in ("{memdir}", "{index_file}", "{index_path}",
                      "{memory_types}", "{max_lines}", "{max_bytes}",
                      "{selector_instructions}", "${MAX_ENTRYPOINT_LINES}",
                      "${MEMORY_TYPES.join"):
            assert token not in out, f"leaked placeholder: {token!r}"
