"""
Interactive skills copilot — a slash-command demo for the TUI.

A single assistant agent with a small library of **skills** (reusable prompt
templates under ``skills/``). In the TUI, type ``/`` to open a picker of the
available skills — name on the left, description on the right, filtered by name
as you type — then press Enter to invoke one. The chosen skill is unwrapped
(its body, with any text you typed after the name substituted for
``$ARGUMENTS``) into a user-message turn, shown as a compact command chip.

Try, for example::

    /brainstorm weekend projects for a spare Raspberry Pi
    /explain-code sorted(items, key=lambda x: -x.score)

Run (needs ``OPENAI_API_KEY`` in ``.env``)::

    python -m grasp_agents.examples.tui.skills_copilot

Requires the ``tui`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.skills import SkillRegistry

DEFAULT_MODEL = "gpt-5.4-nano"

_SKILLS_ROOT = Path(__file__).parent / "skills"

_SYS = "You are a concise, friendly writing-and-coding assistant."


def build_copilot(
    *, model: str = DEFAULT_MODEL
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    """Build the assistant agent (skills enabled) and its session context."""
    llm = OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "medium", "summary": "auto"}},
        ),
    )
    skills = SkillRegistry.from_path(_SKILLS_ROOT)
    ctx = SessionContext[None](state=None, skills=skills)
    agent = LLMAgent[str, str, None](
        name="assistant",
        ctx=ctx,
        llm=llm,
        sys_prompt=_SYS,
        enable_skills=True,
        stream_llm=True,
    )
    return agent, ctx


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    agent, _ = build_copilot()
    # The slash-command palette reads agent.ctx.skills automatically (the same
    # registry the agent uses).
    run_tui_interactive(agent)


if __name__ == "__main__":
    main()
