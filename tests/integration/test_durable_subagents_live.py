"""
True background-task persistence with sub-agents.

Two sub-agents (``AgentTool``, ``resumable=True``) run as background tasks of a
manager. The manager crashes mid-flight; the in-flight sub-agent tasks are
interrupted with their records left PENDING. On resume a fresh manager over the
same session re-spawns them from their checkpoints (``_try_respawn_child``) and
they drive to completion — unlike a Bash command, which is only reported.

Live + slow; ``integration``-gated; needs ``OPENAI_API_KEY``. Run:

    uv run pytest -m integration tests/integration/test_durable_subagents_live.py -s
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from grasp_agents.durability import FileCheckpointStore
from grasp_agents.durability.store_keys import task_prefix
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.run_context import RunContext
from grasp_agents.tools.agent_tool import AgentTool
from grasp_agents.tools.function_tool import function_tool

pytestmark = pytest.mark.integration


def _make_llm() -> Any:
    from grasp_agents.llm_providers.openai_responses.responses_llm import (
        OpenAIResponsesLLM,
    )

    return OpenAIResponsesLLM(
        model_name="gpt-5.4-nano", llm_settings={"max_output_tokens": 400}
    )


# A slow step so each sub-agent is still in flight when the manager crashes
# (→ its task record stays PENDING → resume must re-spawn it).
@function_tool
async def slow_probe(topic: str) -> str:
    """Look up a fact about the topic (slow)."""
    await asyncio.sleep(6)
    return f"probe complete for {topic}"


def _subagent(name: str, topic: str) -> AgentTool[None]:
    return AgentTool[None](
        name=name,
        description=f"Research {topic}. Returns one factual sentence.",
        llm=_make_llm(),
        tools=[slow_probe],
        sys_prompt=(
            f"Research '{topic}'. First call slow_probe with the topic, then "
            f"reply with ONE factual sentence about {topic} mentioning "
            f"'{topic}'."
        ),
        auto_background_at=0,  # background immediately
        max_turns=4,
    )


def _build_manager(ctx: RunContext[None]) -> Any:
    from grasp_agents.agent.llm_agent import LLMAgent

    return LLMAgent[str, str, None](
        name="manager",
        ctx=ctx,
        llm=_make_llm(),
        tools=[
            _subagent("research_ocean", "ocean"),
            _subagent("research_space", "space"),
        ],
        sys_prompt=(
            "You coordinate two researchers. In your FIRST turn, call BOTH "
            "research_ocean AND research_space (they run in the background). "
            "Then wait for both results and reply with a summary that includes "
            "both findings."
        ),
        max_turns=8,
    )


@pytest.mark.asyncio
async def test_two_backgrounded_subagents_respawn_on_resume(
    tmp_path: Any, openai_api_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    store = FileCheckpointStore(tmp_path / "ckpt")

    # --- First run: crash after both sub-agents have been launched ---
    ctx1: RunContext[None] = RunContext(
        state=None, checkpoint_store=store, session_key="subagents"
    )
    manager1 = _build_manager(ctx1)

    hits = {"n": 0}

    @manager1.add_before_llm_hook
    async def crash_after_launch(**kwargs: Any) -> None:
        hits["n"] += 1
        if hits["n"] == 2:  # turn 1 launched both; crash while they run
            raise RuntimeError("simulated crash mid-research")

    with pytest.raises(BaseException):
        await manager1.run("Investigate both topics.")

    # Faithfully interrupt the still-running sub-agents: cancel the asyncio
    # tasks WITHOUT finalizing their records (a real process death), so the
    # records stay PENDING for resume to act on. (aclose() would instead mark
    # them cancelled, and cancelled records are skipped on resume.)
    in_flight = list(manager1._loop.bg_tasks._tasks.values())
    for pt in in_flight:
        pt.task.cancel()
    await asyncio.gather(*(pt.task for pt in in_flight), return_exceptions=True)

    keys = await store.list_keys(task_prefix("subagents"))
    recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
    assert len(recs) >= 2, "both sub-agents should have been launched as bg tasks"
    assert any(r.status == TaskStatus.PENDING for r in recs), (
        "at least one sub-agent should be mid-flight (PENDING) at the crash"
    )

    # --- Resume: a fresh manager over the same session re-spawns the tasks ---
    ctx2: RunContext[None] = RunContext(
        state=None, checkpoint_store=store, session_key="subagents"
    )
    manager2 = _build_manager(ctx2)

    with caplog.at_level(logging.INFO, logger="grasp_agents.agent.background_tasks"):
        async with manager2:
            result = await manager2.run()  # no inputs -> pure resume
    final = str(result.payloads[0])

    # Proof of TRUE persistence: the framework re-spawned the interrupted
    # sub-agent tasks from their checkpoints (a Bash command, being
    # non-resumable, would only be reported — never logged as re-spawned).
    assert any("Re-spawned child task" in r.getMessage() for r in caplog.records), (
        "resume should have re-spawned the interrupted sub-agent task(s)"
    )

    recs = [
        TaskRecord.model_validate_json(await store.load(k))
        for k in await store.list_keys(task_prefix("subagents"))
    ]
    assert recs
    assert not any(r.status == TaskStatus.PENDING for r in recs), (
        f"no sub-agent task should be left pending after resume, got "
        f"{[(r.tool_name, r.status) for r in recs]}"
    )
    # Every re-spawned researcher that was actually delivered carried its OWN
    # final answer — never the raw ``slow_probe`` output. A re-spawned child
    # must not deliver null / the inner tool's raw text (a no-checkpoint child
    # erroring out, or a task-id collision crossing a re-spawned task's result
    # with a fresh call's). Cancelled duplicates
    # (a weak coordinator re-calling) carry no result and are skipped.
    for r in recs:
        if not r.result:
            continue
        assert not r.result.startswith("probe complete for"), (
            f"{r.tool_name} delivered raw tool output, not a final answer: "
            f"{r.result!r}"
        )

    assert final, "manager produced no final answer on resume"
    low = final.lower()
    assert "ocean" in low, f"final answer missing ocean finding: {final!r}"
    assert "space" in low, f"final answer missing space finding: {final!r}"
    # The summary is built from the researchers' real findings, not raw probe
    # text leaking through a broken re-spawn.
    assert "probe complete for" not in low, (
        f"manager's answer contains raw tool output: {final!r}"
    )
