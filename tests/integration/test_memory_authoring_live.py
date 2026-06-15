"""
Live integration test — exercises the memory_skills_demo step-2 flow
end-to-end against a real LLM, to verify the prompt/tool-description
changes drive the agent toward Edit-not-Write and topic-scope discipline.

Run via: ``uv run pytest tests/integration/test_memory_authoring_live.py -s``

Requires OPENAI_API_KEY. Skipped automatically when the env var is absent.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from grasp_agents import (
    LLMAgent,
    MemoryProvider,
    ProcPacketOutEvent,
    RunContext,
    stream_events,
)
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
    OpenAILLMSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Live test — requires OPENAI_API_KEY",
)
async def test_memory_authoring_does_not_clobber(memdir: Path) -> None:
    """
    The agent should add a new topic + index pointer without clobbering
    the existing user_preferences.md or MEMORY.md.

    Asserts the *invariants*, not the agent's exact filename / wording:
    - user_preferences.md still contains the original 'bullet points' line
    - MEMORY.md still contains the title line and the original pointer
    - at least one new topic file or index pointer was added
    """
    model_name = os.environ.get("MEMORY_TEST_MODEL", "openai/gpt-4o-mini")
    retries = int(os.environ.get("MEMORY_TEST_VALIDATION_RETRIES", "3"))
    llm = OpenAILLM(
        model_name=model_name,
        llm_settings=OpenAILLMSettings(temperature=0.0),
        retry_policy=RetryPolicy(validation_retries=retries),
    )
    provider = MemoryProvider(memdir)
    backend = LocalFileBackend(allowed_roots=[memdir])

    ctx: RunContext[None] = RunContext(
        state=None, memory=provider, file_backend=backend
    )
    agent = LLMAgent[str, str, None](
        name="demo",
        ctx=ctx,
        llm=llm,
        enable_memory=True,
        tools=[],
        sys_prompt="You are a helpful assistant.",
        env_info=False,
        stream_llm=True,
    )

    final: str | None = None
    async for event in stream_events(
        agent.run_stream(
            "Please remember: I'm based in Berlin (CET timezone) and "
            "I work on ML research full-time.",
        ),
        show_input_messages=True,
        max_input_msg_lines=80,
        max_tool_output_lines=80,
    ):
        if isinstance(event, ProcPacketOutEvent):
            final = event.data.payloads[0]  # type: ignore[assignment]

    # Surface what the agent left on disk
    print("\n\n========== FINAL STATE ==========")
    print(f"final answer: {final}")
    print(f"\nmemdir entries: {sorted(p.name for p in memdir.iterdir())}")
    print("\n----- MEMORY.md -----")
    print((memdir / "MEMORY.md").read_text())
    for p in sorted(memdir.glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        print(f"\n----- {p.name} -----")
        print(p.read_text())
    print("=================================\n")

    mem_index = (memdir / "MEMORY.md").read_text()
    prefs = (memdir / "user_preferences.md").read_text()

    # Original MEMORY.md content must survive
    assert "grasp-agents demo memory index" in mem_index, (
        "MEMORY.md title clobbered — agent replaced the index with bare "
        "pointer lines instead of editing in place."
    )
    assert (
        "user_preferences.md) — how the user likes their answers formatted"
        in mem_index
    ), "Original user_preferences pointer line lost from MEMORY.md."

    # Original user_preferences.md content must survive (location/role
    # facts are NOT preferences — they belong in a separate topic)
    assert "bullet points" in prefs, (
        "user_preferences.md was clobbered — original 'Reply in bullet "
        "points' content was lost. Agent should have created a new "
        "topic for location/role facts."
    )

    # Something new must have landed (either a new topic file or a new
    # index line)
    other_md = [p for p in memdir.glob("*.md") if p.name not in {
        "MEMORY.md", "user_preferences.md"
    }]
    new_index_line = any(
        kw in mem_index.lower()
        for kw in ("berlin", "location", "role", "work", "ml", "research", "timezone")
    )
    assert other_md or new_index_line, (
        "Agent did not write any new memory — expected at least a new "
        "topic file or an additional index pointer."
    )
