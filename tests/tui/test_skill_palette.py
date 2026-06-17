"""
Headless tests for the TUI skill palette (slash commands).

Typing ``/`` opens a filtered picker of the available skills; selecting one
unwraps it (with any typed args) into a user-message turn, rendered as a
command chip. Driven with a fake agent so no LLM is involved.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual")

from collections.abc import AsyncIterator

from grasp_agents.run_context import RunContext
from grasp_agents.skills import SkillRegistry
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    Event,
    OutputMessageItemEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.ui.app import (
    GraspAgentsApp,
    _PromptArea,
    _SkillPalette,
)

if TYPE_CHECKING:
    from pathlib import Path

_SKILLS = [
    ("proofread", "Proofread text for grammar and clarity."),
    ("brainstorm", "Brainstorm fresh ideas for a topic."),
    ("explain-code", "Explain a code snippet step by step."),
]


def _make_skills(root: Path) -> SkillRegistry:
    for name, desc in _SKILLS:
        skill_dir = root / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\nDo: $ARGUMENTS\n",
            encoding="utf-8",
        )
    return SkillRegistry.from_path(root)


def _recording_agent(sink: list[str]):
    async def agent(text: str) -> AsyncIterator[Event[object]]:
        sink.append(text)
        yield UserMessageEvent(
            data=InputMessageItem.from_text(text),
            source="assistant",
            destination="assistant",
        )
        yield OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[OutputMessageText(text="ok")], status="completed"
            ),
            source="assistant",
        )

    return agent


@pytest.mark.asyncio
async def test_palette_opens_on_slash(tmp_path: Path) -> None:
    skills = _make_skills(tmp_path)
    app = GraspAgentsApp(
        on_submit=_recording_agent([]),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        palette = app.query_one("#skill-palette", _SkillPalette)
        assert palette.display is False
        app.query_one("#prompt", _PromptArea).insert("/")
        await pilot.pause()
        assert palette.display is True
        assert palette.option_count == len(_SKILLS)


@pytest.mark.asyncio
async def test_palette_filters_by_name(tmp_path: Path) -> None:
    skills = _make_skills(tmp_path)
    app = GraspAgentsApp(
        on_submit=_recording_agent([]),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", _PromptArea).insert("/bra")
        await pilot.pause()
        palette = app.query_one("#skill-palette", _SkillPalette)
        assert palette.display is True
        assert palette.option_count == 1
        assert palette.highlighted_skill() == "brainstorm"


@pytest.mark.asyncio
async def test_non_command_text_keeps_palette_hidden(tmp_path: Path) -> None:
    skills = _make_skills(tmp_path)
    app = GraspAgentsApp(
        on_submit=_recording_agent([]),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", _PromptArea).insert("hello there")
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is False


@pytest.mark.asyncio
async def test_palette_enter_inserts_command_without_submitting(
    tmp_path: Path,
) -> None:
    # Selecting a skill (Enter on the highlighted option) inserts `/name ` so
    # the user can type arguments — it must NOT submit immediately.
    skills = _make_skills(tmp_path)
    sink: list[str] = []
    app = GraspAgentsApp(
        on_submit=_recording_agent(sink),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", _PromptArea)
        prompt.insert("/proofread")  # no space → palette open, "proofread" highlighted
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is True
        await pilot.press("enter")
        await pilot.pause()
        # inserted (with trailing space for args), nothing submitted, palette gone
        assert prompt.text == "/proofread "
        assert sink == []
        assert app.query_one("#skill-palette", _SkillPalette).display is False


@pytest.mark.asyncio
async def test_palette_hides_once_typing_args(tmp_path: Path) -> None:
    skills = _make_skills(tmp_path)
    app = GraspAgentsApp(
        on_submit=_recording_agent([]),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", _PromptArea)
        prompt.insert("/pro")
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is True
        prompt.insert(" report.md")  # a space starts the args → palette dismisses
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is False


@pytest.mark.asyncio
async def test_submit_unwraps_slash_command(tmp_path: Path) -> None:
    # Submitting a `/name args` line (palette already dismissed by the space)
    # unwraps it into the skill's body — exactly what the agent receives.
    # (How that turn renders is covered in test_event_render.)
    skills = _make_skills(tmp_path)
    sink: list[str] = []
    app = GraspAgentsApp(
        on_submit=_recording_agent(sink),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", _PromptArea)
        prompt.insert("/proofread fix this sentence")
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is False
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(sink) == 1
        assert sink[0].startswith(
            '<system-reminder note="user invoked skill proofread">'
        )
        assert sink[0].rstrip().endswith("</system-reminder>")
        assert "Do: fix this sentence" in sink[0]
        assert prompt.text == ""


@pytest.mark.asyncio
async def test_escape_closes_palette(tmp_path: Path) -> None:
    skills = _make_skills(tmp_path)
    app = GraspAgentsApp(
        on_submit=_recording_agent([]),
        main_agent="assistant",
        ctx=RunContext(state=None, skills=skills),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", _PromptArea).insert("/pro")
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is True
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#skill-palette", _SkillPalette).display is False


@pytest.mark.asyncio
async def test_no_palette_without_skills() -> None:
    app = GraspAgentsApp(on_submit=_recording_agent([]), main_agent="assistant")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query("#skill-palette")


def test_example_skills_load() -> None:
    from grasp_agents.examples.skills_copilot import _SKILLS_ROOT

    names = {s.name for s in SkillRegistry.from_path(_SKILLS_ROOT).all}
    assert {"proofread", "brainstorm", "explain-code"} <= names


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY to construct"
)
def test_example_build_copilot_constructs() -> None:
    from grasp_agents.examples.skills_copilot import build_copilot

    agent, ctx = build_copilot()
    assert agent.name == "assistant"
    assert ctx.skills is not None
    assert {s.name for s in ctx.skills.all} == {
        "proofread",
        "brainstorm",
        "explain-code",
    }
