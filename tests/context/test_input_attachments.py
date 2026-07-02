"""Tests for the InputAttachment seam + relevant_memories_attachment."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context.prompt_builder import (
    InputAttachment,
    PromptBuilder,
)
from grasp_agents.memory import (
    RELEVANT_MEMORIES_ATTACHMENT_NAME,
    InMemoryMemoryProvider,
    MemoryEntry,
    MemoryFrontmatter,
    relevant_memories_attachment,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.file_edit.session_state import FileEditSessionState
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import InputMessageItem
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _State(BaseModel):
    pass


def _entry(name: str, description: str, body: str, path: Path) -> MemoryEntry:
    return MemoryEntry(
        frontmatter=MemoryFrontmatter(name=name, description=description),
        body=body,
        path=path,
        mtime_ms=0,
    )


def _builder() -> PromptBuilder[str, _State]:
    return PromptBuilder[str, _State](agent_name="t", sys_prompt=None, in_prompt=None)


# ---------- PromptBuilder.apply_input_attachments ----------


class TestApplyInputAttachments:
    @pytest.mark.asyncio
    async def test_no_attachments_returns_unchanged(self) -> None:
        b = _builder()
        msg = InputMessageItem.from_text("hello")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert out is msg

    @pytest.mark.asyncio
    async def test_text_return_wrapped_in_system_reminder(self) -> None:
        b = _builder()

        def attach(**_: Any) -> str:
            return "Relevant note here."

        b.add_input_attachment(InputAttachment(name="x", compute=attach))
        msg = InputMessageItem.from_text("hello")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert out is not msg
        assert len(out.content_parts) == 2
        attached = out.content_parts[1]
        assert isinstance(attached, InputText)
        assert attached.text.startswith("<system-reminder>")
        assert "Relevant note here." in attached.text
        assert attached.text.endswith("</system-reminder>")

    @pytest.mark.asyncio
    async def test_current_time_attachment_stamps_input(self) -> None:
        from grasp_agents.context import make_current_time_attachment

        b = _builder()
        b.add_input_attachment(make_current_time_attachment())
        msg = InputMessageItem.from_text("hello")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        attached = out.content_parts[1]
        assert isinstance(attached, InputText)
        assert "<system-reminder>" in attached.text
        assert "Current time:" in attached.text

    @pytest.mark.asyncio
    async def test_text_return_unwrapped_when_disabled(self) -> None:
        b = _builder()

        def attach(**_: Any) -> str:
            return "raw text"

        b.add_input_attachment(
            InputAttachment(name="x", compute=attach, wrap_in_system_reminder=False)
        )
        msg = InputMessageItem.from_text("hi")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert isinstance(out.content_parts[1], InputText)
        assert out.content_parts[1].text == "raw text"

    @pytest.mark.asyncio
    async def test_none_return_is_skipped(self) -> None:
        b = _builder()

        def attach(**_: Any) -> None:
            return None

        b.add_input_attachment(InputAttachment(name="x", compute=attach))
        msg = InputMessageItem.from_text("hello")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert out is msg

    @pytest.mark.asyncio
    async def test_input_part_sequence_passes_through(self) -> None:
        b = _builder()

        def attach(**_: Any) -> Sequence[Any]:
            return [
                InputText(text="block"),
                InputImage.from_url("https://example.com/img.jpg"),
            ]

        b.add_input_attachment(InputAttachment(name="x", compute=attach))
        msg = InputMessageItem.from_text("hello")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        # 1 original text part + 2 attached.
        assert len(out.content_parts) == 3
        assert isinstance(out.content_parts[1], InputText)
        assert out.content_parts[1].text == "block"
        assert isinstance(out.content_parts[2], InputImage)

    @pytest.mark.asyncio
    async def test_multiple_attachments_run_in_order(self) -> None:
        b = _builder()

        b.add_input_attachment(
            InputAttachment(
                name="a",
                compute=lambda **_: "first",
                wrap_in_system_reminder=False,
            )
        )
        b.add_input_attachment(
            InputAttachment(
                name="b",
                compute=lambda **_: "second",
                wrap_in_system_reminder=False,
            )
        )
        msg = InputMessageItem.from_text("u")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert len(out.content_parts) == 3
        assert isinstance(out.content_parts[1], InputText)
        assert isinstance(out.content_parts[2], InputText)
        assert out.content_parts[1].text == "first"
        assert out.content_parts[2].text == "second"

    @pytest.mark.asyncio
    async def test_async_attachment(self) -> None:
        b = _builder()

        async def attach(**_: Any) -> str:
            return "async note"

        b.add_input_attachment(
            InputAttachment(name="x", compute=attach, wrap_in_system_reminder=False)
        )
        msg = InputMessageItem.from_text("hi")
        out = await b.apply_input_attachments(
            msg, ctx=SessionContext(state=_State()), exec_id="e"
        )
        assert isinstance(out.content_parts[1], InputText)
        assert out.content_parts[1].text == "async note"

    def test_add_input_attachment_dedupes_by_name(self) -> None:
        b = _builder()
        first = InputAttachment(name="x", compute=lambda **_: "1")
        second = InputAttachment(name="x", compute=lambda **_: "2")
        b.add_input_attachment(first)
        b.add_input_attachment(second)
        assert len(b.input_attachments) == 1
        assert b.input_attachments[0] is second


# ---------- relevant_memories_attachment ----------


class TestMemoryRelevanceAttachment:
    @pytest.mark.asyncio
    async def test_no_provider_returns_none(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State())
        result = await relevant_memories_attachment.compute(
            input_message=InputMessageItem.from_text("q"), ctx=ctx, exec_id="e"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_selector_returns_none(self, tmp_path: Path) -> None:
        provider = InMemoryMemoryProvider(
            entries=[_entry("alpha", "Alpha", "body", tmp_path / "alpha.md")]
        )
        ctx: SessionContext[_State] = SessionContext(state=_State(), memory=provider)
        result = await relevant_memories_attachment.compute(
            input_message=InputMessageItem.from_text("q"), ctx=ctx, exec_id="e"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_selector_renders_selected_bodies(self, tmp_path: Path) -> None:
        entries = [
            _entry("alpha", "Alpha desc", "ALPHA BODY", tmp_path / "alpha.md"),
            _entry("beta", "Beta desc", "BETA BODY", tmp_path / "beta.md"),
        ]
        provider = InMemoryMemoryProvider(entries=entries)

        def keep_alpha(
            *, entries: Sequence[MemoryEntry], **_: Any
        ) -> Sequence[MemoryEntry]:
            return [e for e in entries if e.name == "alpha"]

        provider.set_selector(keep_alpha)
        ctx: SessionContext[_State] = SessionContext(state=_State(), memory=provider)
        result = await relevant_memories_attachment.compute(
            input_message=InputMessageItem.from_text("q"), ctx=ctx, exec_id="e"
        )
        assert result is not None
        assert "## Relevant memories" in result
        assert "### alpha" in result
        assert "Alpha desc" in result
        assert "ALPHA BODY" in result
        # beta wasn't selected.
        assert "BETA BODY" not in result

    @pytest.mark.asyncio
    async def test_seen_memories_are_filtered_out(self, tmp_path: Path) -> None:
        """A memory already in the session read-set is not re-injected."""
        from grasp_agents.memory.injection import (
            _compute_relevant_memories,
        )

        entries = [
            _entry("alpha", "Alpha desc", "ALPHA BODY", tmp_path / "alpha.md"),
            _entry("beta", "Beta desc", "BETA BODY", tmp_path / "beta.md"),
        ]
        provider = InMemoryMemoryProvider(entries=entries)
        provider.set_selector(
            lambda *, entries, **_: entries  # type: ignore[arg-type,return-value]
        )

        # alpha was already read this session — its path is in the read-set,
        # so it must be dropped before selection; beta still surfaces.
        state = FileEditSessionState()
        state.record_read(tmp_path / "alpha.md", 0.0)
        agent_ctx = SimpleNamespace(file_edit_state=state)

        class _Ctx:
            memory = provider

        rendered = await _compute_relevant_memories(
            input_message=InputMessageItem.from_text("q", role="user"),  # type: ignore[arg-type]
            ctx=_Ctx(),  # type: ignore[arg-type]
            exec_id=None,
            messages=[InputMessageItem.from_text("q", role="user")],
            agent_ctx=agent_ctx,  # type: ignore[arg-type]
        )

        assert rendered is not None
        assert "BETA BODY" in rendered
        assert "ALPHA BODY" not in rendered

    @pytest.mark.asyncio
    async def test_selector_returning_empty_returns_none(self, tmp_path: Path) -> None:
        entries = [_entry("alpha", "Alpha", "x", tmp_path / "alpha.md")]
        provider = InMemoryMemoryProvider(entries=entries)

        def empty(**_: Any) -> Sequence[MemoryEntry]:
            return []

        provider.set_selector(empty)
        ctx: SessionContext[_State] = SessionContext(state=_State(), memory=provider)
        result = await relevant_memories_attachment.compute(
            input_message=InputMessageItem.from_text("q"), ctx=ctx, exec_id="e"
        )
        assert result is None


# ---------- LLMAgent auto-registers relevant_memories_attachment ----------


class TestLLMAgentAutoRegisters:
    def test_attachment_registered_when_memory_enabled(self) -> None:
        agent = LLMAgent[str, str, _State](
            name="t",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
            enable_memory=True,
        )
        names = [
            a.name
            for a in agent._prompt_builder.input_attachments  # pyright: ignore[reportPrivateUsage]
        ]
        assert RELEVANT_MEMORIES_ATTACHMENT_NAME in names

    def test_attachment_not_registered_by_default(self) -> None:
        agent = LLMAgent[str, str, _State](
            name="t",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
        )
        names = [
            a.name
            for a in agent._prompt_builder.input_attachments  # pyright: ignore[reportPrivateUsage]
        ]
        assert RELEVANT_MEMORIES_ATTACHMENT_NAME not in names
