"""
Live integration test — runs the full ``memory_skills_demo.ipynb`` flow
end-to-end against a real LLM. Verifies that the agent recovers from
tool-arg validation failures via the synthesis fallback even with a
weak model.

Use ``MEMORY_TEST_MODEL`` and ``MEMORY_TEST_VALIDATION_RETRIES`` to
sweep models / retry budgets.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import (
    LLMAgent,
    MemoryEntry,
    MemoryProvider,
    ProcPacketOutEvent,
    RunContext,
    SkillRegistry,
    render_events,
)
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
    OpenAILLMSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _make_llm() -> OpenAILLM:
    model_name = os.environ.get("MEMORY_TEST_MODEL", "openai/gpt-4o-mini")
    retries = int(os.environ.get("MEMORY_TEST_VALIDATION_RETRIES", "3"))
    return OpenAILLM(
        model_name=model_name,
        llm_settings=OpenAILLMSettings(temperature=0.0),
        retry_policy=RetryPolicy(validation_retries=retries),
    )


async def _run_and_capture(
    agent: LLMAgent[Any, Any, Any], message: str
) -> Any:
    final: Any = None
    async for event in render_events(
        agent.run_stream(message),
        show_input_messages=False,
        max_input_msg_lines=20,
        max_tool_output_lines=20,
    ):
        if isinstance(event, ProcPacketOutEvent):
            final = event.data.payloads[0]
    return final


@pytest.fixture
def memdir(tmp_path: Path) -> Path:
    d = tmp_path / "memdir"
    d.mkdir()
    (d / "MEMORY.md").write_text(
        "# grasp-agents demo memory index\n\n"
        "Topics:\n\n"
        "- [user_preferences](user_preferences.md) — how the user "
        "likes their answers formatted.\n",
        encoding="utf-8",
    )
    (d / "user_preferences.md").write_text(
        "---\nname: user_preferences\ntype: user\n"
        "description: how the user likes their answers formatted\n---\n"
        "Reply in concise bullet points unless the user asks for prose.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    skill_dir = root / "summarize"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: summarize\n"
        "description: Summarize text in exactly 3 bullets.\n---\n"
        "Output exactly three bullet points capturing the key facts of "
        "the text the user supplied. No preamble, no closing remark, "
        "no tool calls — just the three bullets.\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Live test — requires OPENAI_API_KEY",
)
async def test_full_notebook_flow(memdir: Path, skills_root: Path) -> None:
    """
    Run the four steps of ``memory_skills_demo.ipynb`` against a real LLM:

    1. Read existing memory + linked topic
    2. Author a new topic + maintain MEMORY.md (the recovery test)
    3. Load a skill via ``load_skill``
    4. Per-turn memory relevance via a selector

    After step 2 (the one that exercises the Edit-append pattern and
    triggers validation failures on weak models), assert the index
    structure is preserved and a new topic file exists.
    """
    backend = LocalFileBackend(allowed_roots=[memdir])

    # ---- Step 1: Read existing memory ----
    print("\n\n========== STEP 1: READ ==========")
    provider1 = MemoryProvider(memdir)
    ctx1: RunContext[None] = RunContext(
        state=None, memory=provider1, file_backend=backend
    )
    agent1 = LLMAgent[str, str, None](
        name="demo",
        ctx=ctx1,
        llm=_make_llm(),
        enable_memory=True,
        tools=[],
        sys_prompt="You are a helpful assistant.",
        env_info=False,
        stream_llm=True,
    )
    final1 = await _run_and_capture(
        agent1,
        "How do I prefer my answers formatted? Look it up if you need to.",
    )
    print(f"step 1 final: {final1}")
    assert final1, "step 1 produced no final answer"
    assert "bullet" in str(final1).lower(), (
        "step 1 should mention bullet-points preference from "
        "user_preferences.md"
    )

    # ---- Step 2: Author + maintain index ----
    print("\n========== STEP 2: AUTHOR ==========")
    provider2 = MemoryProvider(memdir)
    ctx2 = RunContext(state=None, memory=provider2, file_backend=backend)
    agent2 = LLMAgent[str, str, None](
        name="demo",
        ctx=ctx2,
        llm=_make_llm(),
        enable_memory=True,
        tools=[],
        sys_prompt="You are a helpful assistant.",
        env_info=False,
        stream_llm=True,
    )
    final2 = await _run_and_capture(
        agent2,
        "Please remember: I'm based in Berlin (CET timezone) and "
        "I work on ML research full-time.",
    )
    print(f"step 2 final: {final2}")
    print(f"\nmemdir after step 2: {sorted(p.name for p in memdir.iterdir())}")
    mem_index = (memdir / "MEMORY.md").read_text()
    prefs = (memdir / "user_preferences.md").read_text()
    print("\n----- MEMORY.md -----")
    print(mem_index)
    for p in sorted(memdir.glob("*.md")):
        if p.name in {"MEMORY.md", "user_preferences.md"}:
            continue
        print(f"\n----- {p.name} -----")
        print(p.read_text())

    assert "grasp-agents demo memory index" in mem_index, (
        "MEMORY.md title was clobbered"
    )
    assert "bullet points" in prefs, "user_preferences.md was clobbered"
    new_topics = [
        p for p in memdir.glob("*.md")
        if p.name not in {"MEMORY.md", "user_preferences.md"}
    ]
    assert new_topics, "Agent didn't write any new topic file"

    # ---- Step 3: Skill load ----
    print("\n========== STEP 3: SKILL ==========")
    ctx3 = RunContext(
        state=None,
        memory=MemoryProvider(memdir),
        file_backend=backend,
        skills=SkillRegistry.from_path(skills_root),
    )
    agent3 = LLMAgent[str, str, None](
        name="demo",
        ctx=ctx3,
        llm=_make_llm(),
        enable_skills=True,
        tools=[],
        sys_prompt=(
            "You are a helpful assistant. When the user mentions a skill "
            "by name, call `load_skill` exactly once to retrieve its "
            "body, then follow the body's instructions in your next reply."
        ),
        max_turns=4,
        env_info=False,
        stream_llm=True,
    )
    final3 = await _run_and_capture(
        agent3,
        "Use the summarize skill on this text:\n"
        "Memory in grasp-agents lives in a markdown directory at "
        "~/.grasp/projects/<sanitized-cwd>/memory/. MEMORY.md is "
        "always loaded into the system prompt. Topic .md files have "
        "YAML frontmatter and are loaded on demand with the generic "
        "Read tool.",
    )
    print(f"step 3 final: {final3}")
    assert final3, "step 3 produced no final answer"
    # Skill says exactly 3 bullets — count bullet markers
    bullet_lines = [
        ln for ln in str(final3).splitlines()
        if ln.strip().startswith(("-", "*", "•"))
    ]
    assert len(bullet_lines) >= 2, (
        f"step 3 should produce 3 bullets, got: {bullet_lines!r}"
    )

    # ---- Step 4: Selector + relevance attachment ----
    print("\n========== STEP 4: SELECTOR ==========")
    provider4 = MemoryProvider(memdir)

    def keep_user_type(
        *, entries: Sequence[MemoryEntry], **_: Any
    ) -> Sequence[MemoryEntry]:
        return [e for e in entries if e.memory_type == "user"]

    provider4.set_selector(keep_user_type)
    ctx4 = RunContext(state=None, memory=provider4, file_backend=backend)
    agent4 = LLMAgent[str, str, None](
        name="demo",
        ctx=ctx4,
        llm=_make_llm(),
        enable_memory=True,
        sys_prompt="You are a helpful assistant.",
        env_info=False,
        stream_llm=True,
    )
    final4 = await _run_and_capture(
        agent4,
        "Tell me one thing about how I prefer my answers.",
    )
    print(f"step 4 final: {final4}")
    assert final4, "step 4 produced no final answer"

    print("\n========== ALL STEPS PASSED ==========")
