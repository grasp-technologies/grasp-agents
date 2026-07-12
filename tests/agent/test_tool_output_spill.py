"""
Foreground tool-output spilling.

When a foreground tool call's serialized result exceeds the tool's
``max_inline_result_chars``, the loop spills the full text to a ``.grasp`` file
and inlines only a head+tail excerpt that points at it (``Read`` / ``Grep``
recovers the rest) — bounding transcript growth from one large output without
hiding it unrecoverably. Mirrors the backgrounded-task cap-and-defer path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.agent.agent_loop import AgentLoop
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.agent.task_progress import excerpt_for_inline, spill_if_large
from grasp_agents.sandbox import local_environment
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.events import ToolErrorInfo
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem
from tests._helpers import MockLLM, _make_agent_loop


@function_tool(max_inline_result_chars=100)
async def capped(n: int = 0) -> str:
    del n
    return ""


@function_tool
async def uncapped(n: int = 0) -> str:
    del n
    return ""


@function_tool(max_inline_result_chars=100, untrusted_output=True)
async def capped_untrusted(n: int = 0) -> str:
    del n
    return ""


def _loop(
    tools: list[BaseTool[Any, Any, Any]], *, ctx: SessionContext[None]
) -> AgentLoop[None]:
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    return _make_agent_loop(
        agent_name="test",
        llm=MockLLM(model_name="mock", responses_queue=[]),
        transcript=transcript,
        ctx=ctx,
        tools=tools,
        max_turns=10,
        stream_llm=False,
    )


def _backed_ctx(tmp_path: Path) -> SessionContext[None]:
    return SessionContext[None](environment=local_environment(allowed_roots=[tmp_path]))


# ---------- excerpt_for_inline (pure) ----------


class TestExcerptForInline:
    def test_under_cap_is_unchanged(self) -> None:
        assert excerpt_for_inline("short", 100) == ("short", False)

    def test_none_cap_is_unchanged(self) -> None:
        assert excerpt_for_inline("x" * 1000, None) == ("x" * 1000, False)

    def test_over_cap_keeps_head_and_tail_with_pointer(self) -> None:
        text, truncated = excerpt_for_inline(
            "A" * 1000, 100, log_file="/root/.grasp/tasks/c1.result"
        )
        assert truncated is True
        # Head + tail total the cap; the dropped middle is reported.
        assert text.count("A") == 100
        assert "900 chars omitted" in text
        assert "full output in /root/.grasp/tasks/c1.result" in text

    def test_over_cap_without_file_omits_pointer(self) -> None:
        text, truncated = excerpt_for_inline("A" * 1000, 100)
        assert truncated is True
        assert "chars omitted" in text
        assert "full output in" not in text


# ---------- spill_if_large (orchestrator) ----------


class TestSpillIfLarge:
    @pytest.mark.asyncio
    async def test_under_cap_returns_text_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        ctx = _backed_ctx(tmp_path)
        out = await spill_if_large(ctx.file_backend, name="c1", text="small", cap=100)
        assert out == "small"
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_none_cap_returns_whole_text(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        out = await spill_if_large(
            ctx.file_backend, name="c1", text="X" * 5000, cap=None
        )
        assert out == "X" * 5000
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_over_cap_spills_to_result_file_and_excerpts(
        self, tmp_path: Path
    ) -> None:
        ctx = _backed_ctx(tmp_path)
        out = await spill_if_large(
            ctx.file_backend, name="c1", text="Z" * 5000, cap=100
        )
        spilled = tmp_path / ".grasp" / "tasks" / "c1.result"
        assert spilled.read_text() == "Z" * 5000
        assert "chars omitted" in out
        assert str(spilled) in out

    @pytest.mark.asyncio
    async def test_no_backend_returns_whole_text(self) -> None:
        out = await spill_if_large(None, name="c1", text="X" * 5000, cap=100)
        assert out == "X" * 5000


# ---------- _convert_tool_output spilling ----------


class TestConvertToolOutputSpill:
    @pytest.mark.asyncio
    async def test_large_output_spills_and_inlines_excerpt(
        self, tmp_path: Path
    ) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([capped], ctx=ctx)
        call = FunctionToolCallItem(call_id="c1", name="capped", arguments="{}")

        item = await loop._convert_tool_output("X" * 1000, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert "chars omitted" in item.text
        match = re.search(r"full output in (.+?)\]", item.text)
        assert match is not None
        spilled = Path(match.group(1).strip())
        assert spilled == tmp_path / ".grasp" / "tasks" / "c1.result"
        # The transcript carries only the excerpt; the file holds the full text.
        assert len(item.text) < 1000
        assert spilled.read_text() == "X" * 1000

    @pytest.mark.asyncio
    async def test_small_output_is_not_spilled(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([capped], ctx=ctx)
        call = FunctionToolCallItem(call_id="c1", name="capped", arguments="{}")

        item = await loop._convert_tool_output("small output", call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert item.text == "small output"
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_no_cap_never_spills(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([uncapped], ctx=ctx)
        call = FunctionToolCallItem(call_id="c1", name="uncapped", arguments="{}")

        item = await loop._convert_tool_output("X" * 5000, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert item.text == "X" * 5000
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_no_file_backend_inlines_whole(self) -> None:
        # No place to spill → keep the full text rather than hide it with no
        # recovery path.
        ctx = SessionContext[None](state=None)
        assert ctx.file_backend is None
        loop = _loop([capped], ctx=ctx)
        call = FunctionToolCallItem(call_id="c1", name="capped", arguments="{}")

        item = await loop._convert_tool_output("X" * 1000, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert item.text == "X" * 1000

    @pytest.mark.asyncio
    async def test_error_result_is_not_spilled(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([capped], ctx=ctx)
        call = FunctionToolCallItem(call_id="c1", name="capped", arguments="{}")
        error = ToolErrorInfo(tool_name="capped", error="X" * 1000, timed_out=False)

        item = await loop._convert_tool_output(error, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert item.is_error is True
        assert "chars omitted" not in item.text
        assert item.text.count("X") == 1000
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_custom_converter_bypasses_spill(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([capped], ctx=ctx)

        async def converter(tool_output: object, *, exec_id: str | None) -> str:
            del tool_output, exec_id
            return "Y" * 1000

        loop.tool_output_converters["capped"] = converter
        call = FunctionToolCallItem(call_id="c1", name="capped", arguments="{}")

        item = await loop._convert_tool_output("X" * 1000, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        assert item.text == "Y" * 1000
        assert not (tmp_path / ".grasp" / "tasks").exists()

    @pytest.mark.asyncio
    async def test_untrusted_output_spills_then_wraps(self, tmp_path: Path) -> None:
        ctx = _backed_ctx(tmp_path)
        loop = _loop([capped_untrusted], ctx=ctx)
        call = FunctionToolCallItem(
            call_id="c1", name="capped_untrusted", arguments="{}"
        )

        item = await loop._convert_tool_output("X" * 1000, call, exec_id="t")  # pyright: ignore[reportPrivateUsage]

        # The inlined excerpt is fenced; the spilled file holds the raw text.
        assert item.text.startswith('<untrusted_content source="capped_untrusted">')
        assert "chars omitted" in item.text
        spilled = tmp_path / ".grasp" / "tasks" / "c1.result"
        assert spilled.read_text() == "X" * 1000
