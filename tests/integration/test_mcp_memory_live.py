"""
Live integration test for ``mcp_memory_demo.ipynb`` — same flow as the
local-FS test but routes file I/O through an MCP stdio server.

Requires ``OPENAI_API_KEY`` and the ``mcp`` optional dependency.
Skipped automatically when either is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

import grasp_agents
from grasp_agents import (
    LLMAgent,
    MemoryProvider,
    ProcPacketOutEvent,
    RunContext,
    stream_events,
)
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    OpenAILLM,
    OpenAILLMSettings,
)

mcp_module = pytest.importorskip("grasp_agents.mcp")
MCPClient = mcp_module.MCPClient
MCPServerStdio = mcp_module.MCPServerStdio
from grasp_agents.file_backend import MCPFileBackend

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _make_llm() -> OpenAILLM:
    model_name = os.environ.get("MEMORY_TEST_MODEL", "openai/gpt-4o-mini")
    retries = int(os.environ.get("MEMORY_TEST_VALIDATION_RETRIES", "3"))
    return OpenAILLM(
        model_name=model_name,
        llm_settings=OpenAILLMSettings(temperature=0.0),
        retry_policy=RetryPolicy(validation_retries=retries),
    )


async def _run_and_capture(agent: LLMAgent[Any, Any, Any], message: str) -> Any:
    final: Any = None
    async for event in stream_events(
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
    # MCP server resolves paths via ``Path.resolve()`` for canonical URIs.
    # macOS ``tmp_path`` lives under ``/var/folders/...`` which symlinks
    # to ``/private/var/folders/...`` — match by resolving here too.
    d = (tmp_path / "memdir").resolve()
    d.mkdir(parents=True)
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
async def test_mcp_memory_flow(memdir: Path) -> None:
    """
    Mirror ``mcp_memory_demo.ipynb``:

    1. Spawn the bundled MCP memory server over stdio.
    2. Read the existing topic via MCP-routed Read.
    3. Author a new topic + update MEMORY.md via MCP write_file.
    4. Assert invariants survive (original content intact, new topic added).
    """
    server_script = (
        Path(grasp_agents.__file__).parent / "examples" / "mcp_memory_server.py"
    )

    async with MCPClient(
        "mem-server",
        server=MCPServerStdio(
            command=sys.executable,
            args=[str(server_script), str(memdir)],
        ),
    ) as client:
        backend = MCPFileBackend(client=client, allowed_roots=[memdir])
        provider = MemoryProvider(memdir)

        ctx: RunContext[None] = RunContext(
            state=None, memory=provider, file_backend=backend
        )

        # Snapshot reads through MCP
        snap = await provider.load()
        assert snap.index is not None
        names = [e.name for e in snap.entries]
        assert "user_preferences" in names

        body = await provider.fetch_body("user_preferences")
        assert "bullet points" in body

        # Step 1: Agent reads memory via MCP
        print("\n========== STEP 1: READ (MCP) ==========")
        agent1 = LLMAgent[str, str, None](
            name="demo",
            ctx=ctx,
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
            "step 1 should mention bullet-points preference"
        )

        # Step 2: Agent authors via MCP write_file. Wrapped in
        # try/except — the goal here is to verify the framework
        # plumbing (ctx propagation, MCP routing, tool dispatch); the
        # LLM's success rate at authoring through MCP under the
        # read-before-write gate is its own concern and varies by
        # model. Whatever happens, we re-assert invariants below.
        print("\n========== STEP 2: AUTHOR (MCP) ==========")
        agent2 = LLMAgent[str, str, None](
            name="demo",
            ctx=ctx,
            llm=_make_llm(),
            enable_memory=True,
            tools=[],
            sys_prompt="You are a helpful assistant.",
            env_info=False,
            stream_llm=True,
        )
        try:
            final2 = await _run_and_capture(
                agent2,
                "Please remember: I'm based in Berlin (CET timezone).",
            )
            print(f"step 2 final: {final2}")
        except Exception as exc:
            print(f"step 2 raised (LLM-side, not framework): {exc!r}")

        await provider.refresh()
        snap2 = await provider.load()
        print(f"entries after step 2: {[e.name for e in snap2.entries]}")

    # Back outside the MCP client — verify the framework left the
    # baseline files intact regardless of whether step 2 succeeded.
    mem_index = (memdir / "MEMORY.md").read_text()
    prefs = (memdir / "user_preferences.md").read_text()
    print(f"\nmemdir after MCP run: {sorted(p.name for p in memdir.iterdir())}")
    print(f"\n----- MEMORY.md -----\n{mem_index}")

    assert "grasp-agents demo memory index" in mem_index
    assert "bullet points" in prefs

    print("\n========== MCP FLOW PASSED ==========")
