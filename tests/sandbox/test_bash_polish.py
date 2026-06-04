"""
Tests for the ``Bash`` long-running-command polish: heartbeat progress,
blocked leading ``sleep``, auto-background with the ``BashOutput`` /
``KillBash`` companions, and cancellation semantics.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.bash import (
    Bash,
    BashIdInput,
    BashInput,
    BashOutput,
    BashProcessRegistry,
    KillBash,
    bash_tools,
)

pytestmark = pytest.mark.asyncio


def _ctx(tmp_path: Path) -> RunContext[None]:
    env = local_environment(allowed_roots=[tmp_path])
    return RunContext(environment=env)


class _ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def __call__(
        self, progress: float, total: float | None, message: str | None
    ) -> None:
        self.calls.append((progress, total, message))


# --- blocked leading sleep ---------------------------------------------------


async def test_leading_sleep_is_blocked(tmp_path: Path) -> None:
    tool = Bash()
    with pytest.raises(ValueError, match="leading `sleep`"):
        await tool._run(BashInput(command="sleep 5"), ctx=_ctx(tmp_path))


async def test_leading_sleep_blocked_with_whitespace(tmp_path: Path) -> None:
    tool = Bash()
    with pytest.raises(ValueError, match="leading `sleep`"):
        await tool._run(BashInput(command="   sleep 2 && echo hi"), ctx=_ctx(tmp_path))


async def test_non_leading_sleep_is_allowed(tmp_path: Path) -> None:
    tool = Bash()
    res = await tool._run(
        BashInput(command="echo hi && sleep 0.05"), ctx=_ctx(tmp_path)
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "hi"


async def test_block_can_be_disabled(tmp_path: Path) -> None:
    tool = Bash(block_leading_sleep=False)
    res = await tool._run(BashInput(command="sleep 0.05"), ctx=_ctx(tmp_path))
    assert res.returncode == 0


async def test_stdout_and_stderr_captured_separately(tmp_path: Path) -> None:
    tool = Bash()
    res = await tool._run(
        BashInput(command="echo OUT; echo ERR 1>&2"), ctx=_ctx(tmp_path)
    )
    assert res.returncode == 0
    # Streams are kept separate and labeled.
    assert "OUT" in res.stdout
    assert "ERR" in res.stderr


# --- heartbeat ----------------------------------------------------------------


async def test_heartbeat_progress_reported(tmp_path: Path) -> None:
    recorder = _ProgressRecorder()
    tool = Bash(progress_at=0.1, heartbeat_every=0.15)
    res = await tool._run(
        BashInput(command="echo start && sleep 0.6 && echo end"),
        ctx=_ctx(tmp_path),
        progress_callback=recorder,
    )
    assert res.returncode == 0
    assert len(recorder.calls) >= 1
    elapsed, total, message = recorder.calls[0]
    assert elapsed >= 0.1
    assert total is not None  # the effective timeout
    assert message is not None
    assert "running" in message


async def test_fast_command_emits_no_heartbeat(tmp_path: Path) -> None:
    recorder = _ProgressRecorder()
    tool = Bash(progress_at=5.0)
    res = await tool._run(
        BashInput(command="echo quick"), ctx=_ctx(tmp_path), progress_callback=recorder
    )
    assert res.returncode == 0
    assert recorder.calls == []


# --- auto-background ----------------------------------------------------------


async def test_fast_command_completes_in_foreground(tmp_path: Path) -> None:
    registry = BashProcessRegistry()
    tool = Bash(auto_background_at=5.0, registry=registry)
    res = await tool._run(BashInput(command="echo fg"), ctx=_ctx(tmp_path))
    assert res.status == "completed"
    assert res.bash_id is None
    assert res.stdout.strip() == "fg"
    assert registry._sessions == {}


async def test_long_command_backgrounds_and_completes(tmp_path: Path) -> None:
    registry = BashProcessRegistry()
    tool = Bash(auto_background_at=0.3, registry=registry)
    poll = BashOutput(registry)

    res = await tool._run(
        BashInput(command="echo early && sleep 0.8 && echo late"),
        ctx=_ctx(tmp_path),
    )
    assert res.status == "backgrounded"
    assert res.bash_id is not None
    assert res.returncode is None
    assert res.reason == "running"
    # Output produced before backgrounding is included.
    assert "early" in res.stdout

    # Poll until done.
    collected = res.stdout
    for _ in range(40):
        out = await poll._run(BashIdInput(bash_id=res.bash_id))
        collected += out.stdout
        if out.status == "completed":
            assert out.returncode == 0
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("backgrounded command never completed")

    assert "late" in collected
    # Finished sessions are removed once their final result is delivered.
    assert registry._sessions == {}


async def test_kill_bash_terminates_background_command(tmp_path: Path) -> None:
    registry = BashProcessRegistry()
    tool = Bash(auto_background_at=0.2, registry=registry)
    killer = KillBash(registry)

    marker = tmp_path / "ticks.txt"
    res = await tool._run(
        BashInput(
            command=(
                f"i=0; while true; do echo tick >> {marker}; "
                "i=$((i+1)); sleep 0.1; done"
            ),
            timeout=30,
        ),
        ctx=_ctx(tmp_path),
    )
    assert res.status == "backgrounded"
    assert res.bash_id is not None

    killed = await killer._run(BashIdInput(bash_id=res.bash_id))
    assert killed.status == "completed"
    assert killed.reason in {"manual_cancel", "signal", "exit"}

    # The process group is actually dead: the marker stops growing.
    await asyncio.sleep(0.3)
    size_a = marker.stat().st_size if marker.exists() else 0
    await asyncio.sleep(0.4)
    size_b = marker.stat().st_size if marker.exists() else 0
    assert size_a == size_b

    with pytest.raises(ValueError, match="Unknown bash_id"):
        await killer._run(BashIdInput(bash_id=res.bash_id))


async def test_cancelling_foreground_run_kills_process(tmp_path: Path) -> None:
    tool = Bash()
    marker = tmp_path / "fg_ticks.txt"
    run = asyncio.ensure_future(
        tool._run(
            BashInput(
                command=f"while true; do echo tick >> {marker}; sleep 0.1; done",
                timeout=30,
            ),
            ctx=_ctx(tmp_path),
        )
    )
    await asyncio.sleep(0.4)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    await asyncio.sleep(0.3)
    size_a = marker.stat().st_size if marker.exists() else 0
    await asyncio.sleep(0.4)
    size_b = marker.stat().st_size if marker.exists() else 0
    assert size_a == size_b


async def test_registry_caps_sessions(tmp_path: Path) -> None:
    registry = BashProcessRegistry(max_sessions=1)
    tool = Bash(auto_background_at=0.1, registry=registry)
    ctx = _ctx(tmp_path)

    first = await tool._run(
        BashInput(command="echo a && sleep 1", timeout=10), ctx=ctx
    )
    assert first.status == "backgrounded"
    with pytest.raises(RuntimeError, match="Too many background commands"):
        await tool._run(BashInput(command="echo b && sleep 1", timeout=10), ctx=ctx)
    assert first.bash_id is not None
    await registry.kill(first.bash_id)


# --- completion notes ----------------------------------------------------------


async def test_collect_notes_announces_each_finish_once(tmp_path: Path) -> None:
    registry = BashProcessRegistry()
    tool = Bash(auto_background_at=0.1, registry=registry)
    res = await tool._run(
        BashInput(command="echo done && sleep 0.3", timeout=10), ctx=_ctx(tmp_path)
    )
    assert res.status == "backgrounded"
    assert res.bash_id is not None
    # Output produced before backgrounding was already delivered.
    assert "done" in res.stdout

    # Still running — nothing to announce yet.
    assert registry.collect_notes() == []

    session = registry.get(res.bash_id)
    assert session.drain is not None
    await session.drain

    notes = registry.collect_notes()
    assert len(notes) == 1
    assert res.bash_id in notes[0]
    assert "returncode=0" in notes[0]
    assert "BashOutput" in notes[0]
    # Exactly once.
    assert registry.collect_notes() == []
    # The final result is still pollable; the cursor skips what the
    # backgrounded response already delivered.
    out = await BashOutput(registry)._run(BashIdInput(bash_id=res.bash_id))
    assert out.status == "completed"
    assert out.returncode == 0
    assert out.stdout == ""


async def test_idle_wait_primitives_avoid_polling(tmp_path: Path) -> None:
    """
    The pieces the loop composes for its idle wait: a running session exposes
    a drain future to block on (no poll loop), and once it finishes
    ``collect_notes`` yields exactly one note, announced once.
    """
    registry = BashProcessRegistry()
    tool = Bash(auto_background_at=0.1, registry=registry)
    res = await tool._run(
        BashInput(command="echo bg && sleep 0.4", timeout=10), ctx=_ctx(tmp_path)
    )
    assert res.status == "backgrounded"

    # Still running: a future to block on, no note yet.
    pending = registry.pending_drains()
    assert len(pending) == 1
    assert registry.collect_notes() == []

    # Blocking on the future returns when the command completes — no polling.
    await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

    notes = registry.collect_notes()
    assert len(notes) == 1
    assert res.bash_id is not None
    assert res.bash_id in notes[0]
    # Announced once — nothing left to wait on.
    assert registry.pending_drains() == []


async def test_loop_injects_bash_note_after_idle_wait(tmp_path: Path) -> None:
    """
    End-to-end: the model takes a no-op turn while a backgrounded Bash command
    runs, the loop idle-waits on it (no polling), and injects its completion
    note as a user message before the next turn — which then ends the run.
    """
    from collections.abc import AsyncIterator, Mapping, Sequence
    from dataclasses import dataclass, field
    from typing import Any

    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
    )
    from pydantic import BaseModel

    from grasp_agents.agent.agent_loop import AgentLoop
    from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
    from grasp_agents.llm.llm import LLM
    from grasp_agents.types.content import OutputMessageText
    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem, OutputMessageItem
    from grasp_agents.types.response import Response, ResponseUsage
    from grasp_agents.types.tool import BaseTool as _BaseTool

    usage = ResponseUsage(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )

    def _resp(items: list[Any]) -> Response:
        return Response(model="mock", output_items=items, usage_with_cost=usage)

    @dataclass(frozen=True)
    class _QueueLLM(LLM):
        model_name: str = "mock"
        queue: list[Response] = field(default_factory=list)

        async def _generate_response_once(
            self,
            input: Sequence[Any],
            *,
            tools: Mapping[str, _BaseTool[BaseModel, Any, Any]] | None = None,
            output_schema: Any | None = None,
            tool_choice: Any | None = None,
            **extra_llm_settings: Any,
        ) -> Response:
            return self.queue.pop(0)

        async def _generate_response_stream_once(
            self,
            input: Sequence[Any],
            *,
            tools: Mapping[str, _BaseTool[BaseModel, Any, Any]] | None = None,
            output_schema: Any | None = None,
            tool_choice: Any | None = None,
            **extra_llm_settings: Any,
        ) -> AsyncIterator[Any]:
            raise NotImplementedError
            yield  # stream path unused (stream_llm=False)

    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[None] = RunContext(environment=env)

    transcript = LLMAgentTranscript()
    transcript.reset(instructions="sys")
    transcript.update([InputMessageItem.from_text("go", role="user")])

    # Turn 0: empty output → no tool call, no final answer → the loop
    # continues. Turn 1: a final answer ends the run (after the note).
    llm = _QueueLLM(
        queue=[
            _resp([]),
            _resp(
                [
                    OutputMessageItem(
                        content_parts=[OutputMessageText(text="done")],
                        status="completed",
                    )
                ]
            ),
        ]
    )
    loop = AgentLoop[None](
        agent_name="t",
        llm=llm,
        transcript=transcript,
        ctx=ctx,
        tools=bash_tools(),
        max_turns=10,
        stream_llm=False,
    )

    # Background a command into the loop's own registry; it is still running
    # when the loop starts.
    bash = Bash(auto_background_at=0.1, registry=loop.bash_registry)
    res = await bash._run(
        BashInput(command="echo HELLO && sleep 0.3", timeout=10), ctx=ctx
    )
    assert res.status == "backgrounded"
    assert res.bash_id is not None

    events = [e async for e in loop.execute_stream(exec_id="t")]

    # Exactly one completion note for the command, injected at the idle-wait
    # turn rather than by polling.
    notes = [
        e
        for e in events
        if isinstance(e, UserMessageEvent) and res.bash_id in str(e.data)
    ]
    assert len(notes) == 1
    assert "finished" in str(notes[0].data)
    # ...and it landed in the transcript ahead of the final answer.
    assert any(res.bash_id in str(m) for m in transcript.messages)


async def test_llm_agent_auto_wires_bash_notes() -> None:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from dataclasses import dataclass
    from typing import Any

    from pydantic import BaseModel

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.llm.llm import LLM
    from grasp_agents.types.response import Response
    from grasp_agents.types.tool import BaseTool as _BaseTool

    @dataclass(frozen=True)
    class _StubLLM(LLM):
        model_name: str = "stub"

        async def _generate_response_once(
            self,
            input: Sequence[Any],
            *,
            tools: Mapping[str, _BaseTool[BaseModel, Any, Any]] | None = None,
            output_schema: Any | None = None,
            tool_choice: Any | None = None,
            **extra_llm_settings: Any,
        ) -> Response:
            raise NotImplementedError

        async def _generate_response_stream_once(
            self,
            input: Sequence[Any],
            *,
            tools: Mapping[str, _BaseTool[BaseModel, Any, Any]] | None = None,
            output_schema: Any | None = None,
            tool_choice: Any | None = None,
            **extra_llm_settings: Any,
        ) -> AsyncIterator[Any]:
            raise NotImplementedError
            yield  # makes this an async generator

    tools = bash_tools()
    agent = LLMAgent[str, str, None](name="bash_agent", llm=_StubLLM(), tools=tools)
    # The loop owns the registry directly (it injects notes / waits on it in
    # PRE-ACT); the tools resolve it per run through the ContextVar.
    assert agent._loop.bash_registry is not None
    # The loop also owns one persistent-session holder (bound per run).
    assert agent._loop.bash_session_holder is not None
    bash = tools[0]
    assert isinstance(bash, Bash)
    assert bash.registry is None  # no explicit registry — loop-resolved


# --- factory -------------------------------------------------------------------


async def test_bash_tools_factory_resolves_loop_registry(tmp_path: Path) -> None:
    from grasp_agents.tools.bash import (
        reset_current_bash_registry,
        set_current_bash_registry,
    )

    tools = bash_tools(auto_background_at=0.1)
    assert [t.name for t in tools] == ["Bash", "BashOutput", "KillBash"]
    bash, _poll, kill = tools
    assert isinstance(bash, Bash)
    assert isinstance(kill, KillBash)
    assert "background" in bash.description

    loop_registry = BashProcessRegistry()
    token = set_current_bash_registry(loop_registry)
    try:
        res = await bash._run(
            BashInput(command="echo x && sleep 0.4", timeout=10), ctx=_ctx(tmp_path)
        )
        assert res.status == "backgrounded"
        assert res.bash_id is not None
        # The session landed in the loop's registry...
        assert res.bash_id in loop_registry._sessions
        # ...where the companions find it too.
        killed = await kill._run(BashIdInput(bash_id=res.bash_id))
        assert killed.status == "completed"
    finally:
        reset_current_bash_registry(token)


async def test_companions_require_a_registry_in_scope() -> None:
    with pytest.raises(ValueError, match="no bash session registry"):
        await BashOutput()._run(BashIdInput(bash_id="bash_1"))


async def test_explicit_registry_wins_and_isolates(tmp_path: Path) -> None:
    from grasp_agents.tools.bash import (
        reset_current_bash_registry,
        set_current_bash_registry,
    )

    explicit = BashProcessRegistry()
    other = BashProcessRegistry()
    tool = Bash(auto_background_at=0.1, registry=explicit)
    token = set_current_bash_registry(other)
    try:
        res = await tool._run(
            BashInput(command="echo x && sleep 0.3", timeout=10), ctx=_ctx(tmp_path)
        )
        assert res.bash_id is not None
        assert res.bash_id in explicit._sessions
        # The ambient registry never sees — or announces — this session.
        assert other._sessions == {}
        assert other.collect_notes() == []
        await explicit.kill(res.bash_id)
    finally:
        reset_current_bash_registry(token)


async def test_copies_get_fresh_registries() -> None:
    import copy

    trio = bash_tools(registry=BashProcessRegistry())
    clones = copy.deepcopy(trio)
    bash_clone, poll_clone, kill_clone = clones
    assert isinstance(bash_clone, Bash)
    assert isinstance(poll_clone, BashOutput)
    assert isinstance(kill_clone, KillBash)
    original = trio[0]
    assert isinstance(original, Bash)
    # A copied agent is a new context: fresh, empty registry — one agent's
    # sessions are never announced into another's transcript...
    assert bash_clone.registry is not original.registry
    # ...but the copied trio still shares one registry (deepcopy memo).
    assert bash_clone.registry is poll_clone._registry
    assert bash_clone.registry is kill_clone._registry


# --- persistent session (default Bash) -----------------------------------------


async def test_persistent_bash_keeps_state_across_calls(tmp_path: Path) -> None:
    from grasp_agents.tools.bash import (
        BashSessionHolder,
        reset_current_session_holder,
        set_current_session_holder,
    )

    (tmp_path / "sub").mkdir()
    tool = Bash()  # persistent=True by default
    holder = BashSessionHolder()
    token = set_current_session_holder(holder)
    ctx = _ctx(tmp_path)
    try:
        r1 = await tool._run(BashInput(command="cd sub && export FOO=bar"), ctx=ctx)
        assert r1.returncode == 0
        # cd and the env var both persist into the next call (same shell).
        r2 = await tool._run(BashInput(command="pwd && echo $FOO"), ctx=ctx)
        assert r2.returncode == 0
        assert "sub" in r2.stdout
        assert "bar" in r2.stdout
    finally:
        await holder.close()
        reset_current_session_holder(token)


async def test_bash_without_holder_is_ephemeral(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    tool = Bash()  # persistent=True, but no holder in scope → one-shot
    ctx = _ctx(tmp_path)
    await tool._run(BashInput(command="cd sub"), ctx=ctx)
    r = await tool._run(BashInput(command="pwd"), ctx=ctx)
    # Fresh process per call: the earlier cd did not persist.
    assert not r.stdout.strip().endswith("sub")
