"""
Regression tests for the P2 misc fixes
(consolidated audit 2026-06-11, §3 items 37 and 40).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from grasp_agents.printer import sanitize_terminal_text
from grasp_agents.telemetry import traced
from grasp_agents.ui._event_render import truncate, truncate_lines

# ---------- Item 40: terminal escape-sequence sanitization ----------


class TestTerminalSanitization:
    def test_csi_clear_screen_neutralized(self) -> None:
        out = sanitize_terminal_text("before\x1b[2Jafter")
        assert "\x1b" not in out
        assert "before" in out
        assert "after" in out

    def test_osc_title_spoof_neutralized(self) -> None:
        out = sanitize_terminal_text("\x1b]0;you-have-been-pwned\x07rest")
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "rest" in out

    def test_carriage_return_overwrite_neutralized(self) -> None:
        # "\r" rewinds the line — classic approval-prompt spoof.
        out = sanitize_terminal_text("rm -rf /\rls -la    ")
        assert "\r" not in out
        assert "rm -rf /" in out

    def test_newlines_and_tabs_kept(self) -> None:
        assert sanitize_terminal_text("a\n\tb\r\nc") == "a\n\tb\nc"

    def test_render_truncate_helpers_sanitize(self) -> None:
        assert "\x1b" not in truncate("x\x1b[2Jy", 100)
        assert "\x1b" not in truncate_lines("x\x1b[2Jy\nz", 10)


# ---------- Item 37: @traced generators stream through ----------


class TestTracedGeneratorPassThrough:
    @pytest.mark.asyncio
    async def test_async_gen_yields_all_items(self) -> None:
        @traced(name="gen")
        async def gen() -> AsyncIterator[int]:
            for i in range(5):
                yield i

        assert [i async for i in gen()] == [0, 1, 2, 3, 4]

    def test_sync_gen_yields_all_items(self) -> None:
        @traced(name="gen")
        def gen():
            yield from range(5)

        assert list(gen()) == [0, 1, 2, 3, 4]
