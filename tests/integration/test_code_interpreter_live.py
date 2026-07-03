"""
Free-form live integration test for ``RunPython``: a real LLM is asked to make
a plot, runs matplotlib code in a **real srt-confined kernel**, and the figure
it displays must come back to it as an image.

Exercises the whole loop end-to-end — model → tool call → srt kernel → rich
display output → image part fed back to the model. Requires ``OPENAI_API_KEY``
and the ``srt`` CLI, and must run unsandboxed:

    uv run pytest -m integration tests/integration/test_code_interpreter_live.py -s
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import LLMAgent, ProcPacketOutEvent, SessionContext
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
)
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.code_interpreter import RunPython
from grasp_agents.types.content import InputImage
from grasp_agents.types.events import ToolOutputItemEvent

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

_SYS_PROMPT = (
    "You are a Python data assistant with a RunPython tool backed by a live "
    "Jupyter kernel. To make a chart, write matplotlib code that draws it and "
    "call `plt.show()` so the figure is "
    "displayed back to you as an image. After you can see the figure, give a "
    "one-sentence description of it as your final answer."
)


def _make_llm() -> OpenAIResponsesLLM:
    # The Responses API (unlike Chat Completions) supports image tool outputs,
    # which is what lets a displayed plot be fed back to the model.
    model_name = os.environ.get("CODE_INTERP_TEST_MODEL", "gpt-4o-mini")
    return OpenAIResponsesLLM(model_name=model_name)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="Live test — requires OPENAI_API_KEY"
)
@pytest.mark.skipif(shutil.which("srt") is None, reason="srt not installed")
async def test_freeform_agent_produces_plot_via_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # srt forces TMPDIR to $CLAUDE_CODE_TMPDIR when set (outside the sandbox);
    # drop it so the kernel uses srt's own writable temp.
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)

    # Keep matplotlib's font cache inside the writable root (srt confines writes).
    env = local_environment(
        allowed_roots=[tmp_path],
        confinement="srt",
        env={"MPLCONFIGDIR": str(tmp_path / ".mpl")},
    )
    ctx: SessionContext[None] = SessionContext(state=None, environment=env)
    agent = LLMAgent[str, str, None](
        name="plotter",
        ctx=ctx,
        llm=_make_llm(),
        tools=[RunPython()],
        sys_prompt=_SYS_PROMPT,
        env_info=False,
        stream_llm=True,
    )

    images: list[InputImage] = []
    tool_text: list[str] = []
    final: Any = None
    async for event in agent.run_stream(
        "Plot y = x**2 for x from 0 to 10 and show me the figure."
    ):
        if isinstance(event, ToolOutputItemEvent):
            parts = event.data.output
            if not isinstance(parts, str):
                images += [p for p in parts if isinstance(p, InputImage)]
                tool_text += [p.text for p in parts if not isinstance(p, InputImage)]
        elif isinstance(event, ProcPacketOutEvent):
            final = event.data.payloads[0]

    # Investigate the output (visible with `-s`).
    print("\n=== RunPython tool text ===")
    for t in tool_text:
        print(t[:500])
    print(f"\n=== images returned to the model: {len(images)} ===")
    for img in images:
        print(f"  - {img.mime_type}")
    print(f"\n=== agent final answer ===\n{final}")

    assert images, (
        "the agent should have displayed a plot that came back as an image; "
        f"tool text was: {tool_text}"
    )
    assert any(img.mime_type == "image/png" for img in images)
    assert final, "agent produced no final answer"
