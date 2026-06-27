"""
Tests for the ``Bash`` long-running-command polish: heartbeat progress,
blocked leading ``sleep``, auto-background via the manager's
``run_backgroundable`` with the ``KillTask`` companion, and cancellation
semantics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from pathlib import Path

    from pydantic import BaseModel

    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.response import Response

from grasp_agents.agent.agent_loop import AgentLoop
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.bash import Bash, BashInput, bash_tools
from grasp_agents.tools.task_tools import KillTask, TaskIdInput
from grasp_agents.types.events import BackgroundTaskLaunchedEvent, UserMessageEvent
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class _StubLLM(LLM):
    """A never-called LLM, just to construct an AgentLoop in tests."""

    model_name: str = "stub"

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        raise NotImplementedError

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # makes this an async generator


def _loop(ctx: RunContext[None], *, path: list[str] | None = None) -> AgentLoop[None]:
    """
    A minimal AgentLoop whose ``bg_tasks`` the companions read.

    ``path`` is the loop's tool-call lineage; a non-``None`` value (e.g. ``[]``)
    is required for backgrounded tasks to get a persisted ``TaskRecord``.
    """
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    return AgentLoop[None](
        agent_name="t",
        llm=_StubLLM(),
        transcript=transcript,
        ctx=ctx,
        tools=[],
        max_turns=10,
        stream_llm=False,
        path=path,
    )


async def _bg(
    loop: AgentLoop[None], tool: Bash, command: str, *, timeout: float = 10
) -> tuple[Any, str | None]:
    """
    Run ``command`` through the manager's deadline-backgrounding path.

    Returns ``(result, task_id)``: when it finishes in the foreground,
    ``result`` is the ``BashResult`` and ``task_id`` is ``None``; when it is
    moved to the background, ``result`` is the launch note and ``task_id`` is
    the sidelined task's id (found in ``loop.bg_tasks``).
    """
    call = FunctionToolCallItem(call_id="c1", name=tool.name, arguments="{}")
    before = set(loop.bg_tasks._tasks)  # pyright: ignore[reportPrivateUsage]
    result, _launched = await loop.bg_tasks.run_backgroundable(
        call,
        tool,
        BashInput(command=command, timeout=timeout),
        ctx=loop.ctx,
        exec_id="t",
        agent_ctx=loop.agent_ctx,
    )
    new = set(loop.bg_tasks._tasks) - before  # pyright: ignore[reportPrivateUsage]
    return result, (next(iter(new)) if new else None)


def _ctx(tmp_path: Path) -> RunContext[None]:
    env = local_environment(allowed_roots=[tmp_path])
    return RunContext(environment=env)


async def _flush(manager: BackgroundTaskManager[None], ctx: RunContext[None]) -> None:
    """
    Drive one ``drain`` pass for its log-mirroring side effect, discarding the
    bubbled events — ``drain`` owns flushing (there is no ``flush_progress``).
    """
    async for _ in manager.drain(exec_id="t", ctx=ctx):
        pass


async def _drain_notes(
    manager: BackgroundTaskManager[None], ctx: RunContext[None]
) -> list[str]:
    """
    The completion notes a single turn-boundary ``drain`` injects. ``drain``
    also mirrors progress to the ``.grasp`` logs, so a truncated note can point
    at the log.
    """
    return [
        e.data.text  # the rendered note text, as the model sees it (not a repr)
        async for e in manager.drain(exec_id="t", ctx=ctx)
        if isinstance(e, UserMessageEvent)
    ]


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
    # Streams are kept separate and labeled in the terminal result.
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


# --- cwd round-trip -----------------------------------------------------------


async def test_cwd_persists_across_calls(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    bash = Bash(auto_background_at=30)  # high deadline: these run in the foreground

    first, bid = await _bg(loop, bash, "cd sub && echo here")
    assert bid is None  # foreground
    assert first.returncode == 0
    assert loop.shell_state.cwd is not None
    assert loop.shell_state.cwd.endswith("sub")

    # The next command starts where the previous one left off.
    second, _ = await _bg(loop, bash, "pwd")
    assert second.stdout.strip().endswith("sub")


async def test_cwd_isolated_per_loop(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    ctx = _ctx(tmp_path)
    loop_a = _loop(ctx)
    loop_b = _loop(ctx)
    bash = Bash(auto_background_at=30)  # one shared (stateless) instance

    await _bg(loop_a, bash, "cd a")
    assert loop_a.shell_state.cwd is not None
    assert loop_a.shell_state.cwd.endswith("/a")
    # loop_b shares the filesystem but has its own shell_state — untouched.
    assert loop_b.shell_state.cwd is None
    out_b, _ = await _bg(loop_b, bash, "pwd")
    assert not out_b.stdout.strip().endswith("/a")


async def test_explicit_cwd_overrides_shell_state(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    bash = Bash(auto_background_at=30)

    await _bg(loop, bash, "cd x")
    assert loop.shell_state.cwd is not None
    assert loop.shell_state.cwd.endswith("/x")
    # An explicit per-call cwd wins over the round-tripped one for that call...
    call = FunctionToolCallItem(call_id="c1", name=bash.name, arguments="{}")
    out, _launched = await loop.bg_tasks.run_backgroundable(
        call,
        bash,
        BashInput(command="pwd", cwd=str(tmp_path / "y")),
        ctx=ctx,
        exec_id="t",
        agent_ctx=loop.agent_ctx,
    )
    assert out.stdout.strip().endswith("/y")
    # ...and the round-trip then tracks where that call ended up.
    assert loop.shell_state.cwd.endswith("/y")


# --- auto-background ----------------------------------------------------------


async def test_fast_command_completes_in_foreground(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    result, task_id = await _bg(loop, Bash(auto_background_at=5.0), "echo fg")
    assert task_id is None  # finished in the foreground before the deadline
    assert result.returncode == 0
    assert result.stdout.strip() == "fg"
    assert loop.bg_tasks._tasks == {}  # pyright: ignore[reportPrivateUsage]


async def test_long_command_backgrounds_and_completes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)

    result, task_id = await _bg(
        loop, Bash(auto_background_at=0.3), "echo early && sleep 0.8 && echo late"
    )
    assert task_id is not None
    assert "moved to the background" in result  # launch note, not a BashResult

    # Block until it finishes (the loop's idle wait), then drain its completion
    # note — the result is delivered there; there is no polling.
    await loop.bg_tasks.wait_idle()
    notes = await _drain_notes(loop.bg_tasks, ctx)
    assert len(notes) == 1
    assert "completed" in notes[0]
    assert "early" in notes[0]  # output produced before backgrounding
    assert "late" in notes[0]  # and after
    assert loop.bg_tasks._tasks == {}  # pyright: ignore[reportPrivateUsage]


async def test_kill_task_terminates_background_command(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    killer = KillTask(loop.bg_tasks)

    marker = tmp_path / "ticks.txt"
    _result, task_id = await _bg(
        loop,
        Bash(auto_background_at=0.2),
        f"i=0; while true; do echo tick >> {marker}; i=$((i+1)); sleep 0.1; done",
        timeout=30,
    )
    assert task_id is not None

    killed = await killer._run(TaskIdInput(task_id=task_id))
    # Killed mid-run → reported as cancelled, with whatever output streamed
    # before the kill (no terminal BashResult — cancellation pre-empts it).
    assert killed.status == "cancelled"

    # The process group is actually dead: the marker stops growing.
    await asyncio.sleep(0.3)
    size_a = marker.stat().st_size if marker.exists() else 0
    await asyncio.sleep(0.4)
    size_b = marker.stat().st_size if marker.exists() else 0
    assert size_a == size_b

    with pytest.raises(ValueError, match="Unknown background task id"):
        await killer._run(TaskIdInput(task_id=task_id))


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


async def test_caps_background_tasks(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    loop.bg_tasks._max_background = 1  # pyright: ignore[reportPrivateUsage]

    _first, first_id = await _bg(loop, Bash(auto_background_at=0.1), "echo a; sleep 1")
    assert first_id is not None
    # Exceeding the cap doesn't crash the loop: the call surfaces an error note
    # (and its process is killed, not leaked).
    second, second_id = await _bg(loop, Bash(auto_background_at=0.1), "echo b; sleep 1")
    assert second_id is None
    assert "could not be backgrounded" in second
    await loop.bg_tasks.kill_task(first_id)


# --- completion notes ----------------------------------------------------------


async def test_completion_note_inlines_small_result_once(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    mgr = loop.bg_tasks
    _result, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo done && sleep 0.3"
    )
    assert task_id is not None

    # Still running — drain emits no completion note yet.
    assert await _drain_notes(mgr, ctx) == []

    # Block until it finishes (mirrors the loop's idle wait), then drain once.
    await mgr.wait_idle()
    notes = await _drain_notes(mgr, ctx)
    assert len(notes) == 1
    assert task_id in notes[0]
    assert "completed" in notes[0]
    # A small result is inlined directly in the note (not excerpted), same
    # delivery a spawned (answer-blocking) task gets.
    assert "done" in notes[0]
    assert "omitted" not in notes[0]
    # Announced exactly once, and a fully-delivered task is dropped (the result
    # already reached the model; the full output remains in the log).
    assert await _drain_notes(mgr, ctx) == []
    assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]


async def test_large_result_excerpted_and_deferred(tmp_path: Path) -> None:
    """
    Cap-and-defer: a backgrounded command whose result exceeds the tool's
    ``max_inline_result_chars`` gets an *excerpted* completion note pointing at
    the task's ``.grasp`` log, which holds the full output — instead of dumping a
    huge log into the transcript. The finished task is dropped.
    """
    import re as _re
    from pathlib import Path as _Path

    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    mgr = loop.bg_tasks
    _note, task_id = await _bg(
        loop,
        Bash(auto_background_at=0.1, max_inline_result_chars=200),
        "head -c 5000 /dev/zero | tr '\\0' 'A'; echo END; sleep 0.3",
    )
    assert task_id is not None

    await mgr.wait_idle()
    notes = await _drain_notes(mgr, ctx)
    assert len(notes) == 1
    note = notes[0]
    assert "completed" in note
    assert "chars omitted" in note  # excerpted
    # The finished task is dropped — the full output lives on disk, not memory.
    assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]

    # Two on-disk artifacts (structured + log). The excerpt marker points at the
    # .result sidecar (the full structured BashResult)...
    result_match = _re.search(r"full output in (.+?)\]", note)
    assert result_match is not None
    result_path = _Path(result_match.group(1).strip())
    assert result_path.suffix == ".result"
    assert result_path.read_text().count("A") >= 5000

    # ...and <output_file> is the streamed .log, which also holds the full output.
    log_match = _re.search(r"<output_file>(.+?)</output_file>", note, _re.DOTALL)
    assert log_match is not None
    log_text = _Path(log_match.group(1).strip()).read_text()
    assert log_text.count("A") == 5000
    assert "END" in log_text


async def test_progress_log_appends_deltas(tmp_path: Path) -> None:
    """
    ``drain`` appends only the *new* output each time — not a full rewrite
    (would be O(n²)) and not a re-append of already-written text (would
    duplicate). Two drains over a staged command leave each line exactly once.
    """
    from pathlib import Path as _Path

    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    mgr = loop.bg_tasks
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo AAAA; sleep 0.6; echo BBBB; sleep 0.4"
    )
    assert task_id is not None

    await asyncio.sleep(0.3)  # AAAA emitted, BBBB not yet
    await _flush(mgr, ctx)
    log = _Path(mgr.get(task_id).log_path or "")
    first = log.read_text()
    assert "AAAA" in first
    assert "BBBB" not in first

    await asyncio.sleep(0.6)  # BBBB now emitted
    await _flush(mgr, ctx)
    second = log.read_text()
    assert "BBBB" in second
    assert second.count("AAAA") == 1  # delta-appended, not rewritten or duplicated

    await mgr.cancel_all(ctx=ctx)


async def test_nonblocking_task_bubbles_stream_events(tmp_path: Path) -> None:
    """
    A backgrounded (non-blocking) command's stream events bubble to the parent
    stream — live progress is decoupled from ``blocks_final_answer`` (which now
    governs only the JUDGE gate).
    """
    from grasp_agents.types.events import ToolStreamEvent

    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    mgr = loop.bg_tasks
    _note, task_id = await _bg(
        loop,
        Bash(auto_background_at=0.1, blocks_final_answer=False),  # fire-and-forget
        "echo streamed && sleep 0.4",
    )
    assert task_id is not None
    assert mgr.get(task_id).blocks_final_answer is False  # non-blocking opt-out

    await mgr.wait_idle()
    events = [e async for e in mgr.drain(exec_id="t", ctx=ctx)]
    streamed = [e for e in events if isinstance(e, ToolStreamEvent)]
    assert streamed  # bubbled despite blocks_final_answer=False
    assert any("streamed" in str(e.data) for e in streamed)


async def test_deadline_note_points_at_log(tmp_path: Path) -> None:
    """
    The launch note for a deadline-backgrounded command cites its ``.grasp`` log
    (resolved eagerly) so the model can ``Read`` / ``Grep`` it while the command
    runs — the inspect-a-running-task path now that ``TaskOutput`` is gone.
    """
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo hi && sleep 0.6"
    )
    assert task_id is not None

    log_path = loop.bg_tasks.get(task_id).log_path
    assert log_path is not None  # resolved eagerly at sideline, before any flush
    assert ".grasp/tasks" in log_path
    assert log_path in note  # the note hands the model the exact path
    assert "Read or Grep" in note

    await loop.bg_tasks.cancel_all(ctx=ctx)
    """
    The loop's idle wait: a running command is waited on (no poll loop), and
    once it finishes drain yields exactly one completion note, announced once.
    """
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    mgr = loop.bg_tasks
    _result, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo bg && sleep 0.4"
    )
    assert task_id is not None

    # Block until completion — no polling — then drain exactly one note.
    await mgr.wait_idle()
    notes = await _drain_notes(mgr, ctx)
    assert len(notes) == 1
    assert task_id in notes[0]

    # Announced once: nothing left pending, so wait_idle returns at once and a
    # second drain yields nothing.
    await mgr.wait_idle()
    assert await _drain_notes(mgr, ctx) == []


async def test_loop_injects_bash_note_after_idle_wait(tmp_path: Path) -> None:
    """
    End-to-end: the model takes a no-op turn while a backgrounded Bash command
    runs, the loop idle-waits on it (no polling), and injects its completion
    note as a user message before the next turn — which then ends the run.
    """
    from dataclasses import field

    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
    )

    from grasp_agents.tools.base import BaseTool as _BaseTool
    from grasp_agents.types.content import OutputMessageText
    from grasp_agents.types.items import OutputMessageItem
    from grasp_agents.types.response import Response, ResponseUsage

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
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
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

    # Background a command into the loop's own task manager (the manager owns
    # the deadline); it is still running when the loop starts.
    _result, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo HELLO && sleep 0.3"
    )
    assert task_id is not None

    events = [e async for e in loop.execute_stream(exec_id="t")]

    # Exactly one completion note for the command, injected at the idle-wait
    # turn rather than by polling.
    notes = [
        e for e in events if isinstance(e, UserMessageEvent) and task_id in str(e.data)
    ]
    assert len(notes) == 1
    assert "completed" in str(notes[0].data)
    # ...and it landed in the transcript ahead of the final answer.
    assert any(task_id in str(m) for m in transcript.messages)


async def test_llm_agent_auto_wires_bash_notes() -> None:
    from grasp_agents.agent.llm_agent import LLMAgent

    tools = bash_tools()
    agent = LLMAgent[str, str, None](name="bash_agent", llm=_StubLLM(), tools=tools)
    # The loop's background task manager runs both subagent tasks and
    # backgrounded Bash commands; the task tools resolve it per call from the
    # loop's AgentContext.
    assert agent._loop.bg_tasks is not None
    # The loop also owns one persistent-session holder, exposed the same way.
    assert agent._loop.agent_ctx.session_holder is not None
    bash = tools[0]
    assert isinstance(bash, Bash)
    # Tools are stateless and shared — no per-agent copy: the passed instance
    # is the one the loop uses, and the manager lives on the AgentContext, not
    # on the tool.
    assert agent._loop.tools["Bash"] is bash
    assert agent._loop.agent_ctx.bg_tasks is agent._loop.bg_tasks


# --- factory -------------------------------------------------------------------


async def test_bash_tools_factory_builds_pair(tmp_path: Path) -> None:
    tools = bash_tools(auto_background_at=0.1)
    assert [t.name for t in tools] == ["Bash", "KillTask"]
    bash, kill = tools
    assert isinstance(bash, Bash)
    assert isinstance(kill, KillTask)
    assert bash.auto_background_at == 0.1
    assert "background" in bash.description

    # The manager moves a long command to the background; KillTask (resolving
    # that same manager from the AgentContext) finds it.
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    _result, task_id = await _bg(loop, bash, "echo x && sleep 0.4")
    assert task_id is not None
    assert task_id in loop.bg_tasks._tasks  # pyright: ignore[reportPrivateUsage]
    killed = await kill._run(TaskIdInput(task_id=task_id), agent_ctx=loop.agent_ctx)
    assert killed.status == "cancelled"


async def test_deadline_backgrounding_emits_launched_event(tmp_path: Path) -> None:
    """
    A deadline-sidelined command emits a ``BackgroundTaskLaunchedEvent`` — the
    same lifecycle event a spawned (answer-blocking) task emits — so observers
    see it move to the background, not just a transcript note.
    """
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    call = FunctionToolCallItem(call_id="c1", name="Bash", arguments="{}")
    note, launched = await loop.bg_tasks.run_backgroundable(
        call,
        Bash(auto_background_at=0.1),
        BashInput(command="echo hi && sleep 0.4", timeout=10),
        ctx=ctx,
        exec_id="t",
        agent_ctx=loop.agent_ctx,
    )
    assert "moved to the background" in note
    assert isinstance(launched, BackgroundTaskLaunchedEvent)
    assert launched.data.tool_name == "Bash"
    assert launched.data.task_id in loop.bg_tasks._tasks  # pyright: ignore[reportPrivateUsage]
    await loop.bg_tasks.kill_task(launched.data.task_id)


async def test_foreground_finish_emits_no_launched_event(tmp_path: Path) -> None:
    """A command that finishes within the deadline returns no launched event."""
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    call = FunctionToolCallItem(call_id="c1", name="Bash", arguments="{}")
    result, launched = await loop.bg_tasks.run_backgroundable(
        call,
        Bash(auto_background_at=5.0),
        BashInput(command="echo fg"),
        ctx=ctx,
        exec_id="t",
        agent_ctx=loop.agent_ctx,
    )
    assert launched is None
    assert result.returncode == 0
    assert result.stdout.strip() == "fg"


async def test_loop_dispatch_deadline_bash_yields_launched(tmp_path: Path) -> None:
    """
    End-to-end: a long Bash call dispatched through ``execute_tools_stream``
    outlives its deadline and the loop bubbles a ``BackgroundTaskLaunchedEvent``
    (the same event a spawned task yields).
    """
    ctx = _ctx(tmp_path)
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    loop = AgentLoop[None](
        agent_name="t",
        llm=_StubLLM(),
        transcript=transcript,
        ctx=ctx,
        tools=[Bash(auto_background_at=0.1)],
        max_turns=10,
        stream_llm=False,
    )
    call = FunctionToolCallItem(
        call_id="c1", name="Bash", arguments='{"command": "echo hi && sleep 0.4"}'
    )
    events = [e async for e in loop.execute_tools_stream([call], exec_id="t")]
    launched = [e for e in events if isinstance(e, BackgroundTaskLaunchedEvent)]
    assert len(launched) == 1
    assert launched[0].data.tool_name == "Bash"
    await loop.bg_tasks.cancel_all(ctx=ctx)


async def test_companions_require_a_manager_in_scope() -> None:
    with pytest.raises(ValueError, match="background task manager"):
        await KillTask()._run(TaskIdInput(task_id="bg_1"))


# --- durable task records for backgrounded commands ---------------------------


def _ctx_with_store(
    tmp_path: Path, store: Any, session_key: str = "s1"
) -> RunContext[None]:
    env = local_environment(allowed_roots=[tmp_path])
    return RunContext(environment=env, checkpoint_store=store, session_key=session_key)


async def test_backgrounded_bash_persists_pending_record(tmp_path: Path) -> None:
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord, TaskStatus

    store = InMemoryCheckpointStore()
    ctx = _ctx_with_store(tmp_path, store)
    loop = _loop(ctx, path=[])
    _note, task_id = await _bg(loop, Bash(auto_background_at=0.1), "echo hi && sleep 5")
    assert task_id is not None

    # A backgrounded (non-resumable) shell command now leaves a PENDING record,
    # so a restart can tell the agent it was in flight.
    keys = await store.list_keys(task_prefix(ctx.session_key))
    recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
    assert recs
    assert all(r.status == TaskStatus.PENDING and r.tool_name == "Bash" for r in recs)

    await loop.bg_tasks.cancel_all(ctx=ctx)


async def test_kill_marks_record_cancelled(tmp_path: Path) -> None:
    from grasp_agents.agent.background_tasks import BackgroundTaskManager
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord, TaskStatus

    store = InMemoryCheckpointStore()
    ctx = _ctx_with_store(tmp_path, store)
    loop = _loop(ctx, path=[])
    _note, task_id = await _bg(loop, Bash(auto_background_at=0.1), "echo hi && sleep 5")
    assert task_id is not None

    await loop.bg_tasks.kill_task(task_id, ctx=ctx)

    keys = await store.list_keys(task_prefix(ctx.session_key))
    recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
    assert recs
    assert all(r.status == TaskStatus.CANCELLED for r in recs)

    # A later resume must NOT report a deliberately-killed task as interrupted.
    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    mgr2 = BackgroundTaskManager(
        agent_name="t", transcript=transcript, tools={}, path=[]
    )
    await mgr2.resume_durable(ctx=ctx, exec_id="t")
    assert not any("interrupted" in str(m) for m in transcript.messages)


async def test_drain_marks_record_delivered(tmp_path: Path) -> None:
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord, TaskStatus

    store = InMemoryCheckpointStore()
    ctx = _ctx_with_store(tmp_path, store)
    loop = _loop(ctx, path=[])
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo hi && sleep 0.4"
    )
    assert task_id is not None

    await loop.bg_tasks.wait_idle()
    notes = await _drain_notes(loop.bg_tasks, ctx)
    assert len(notes) == 1
    assert "completed" in notes[0]

    # Draining delivers the result but defers the DELIVERED flip until a
    # checkpoint has persisted the note — a crash before that must re-inject
    # the outcome on resume.
    keys = await store.list_keys(task_prefix(ctx.session_key))
    recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
    assert recs
    assert all(r.status == TaskStatus.COMPLETED for r in recs)

    # flush_delivered (called after the agent checkpoint) flips the records,
    # so a resume won't re-inject a result the agent already saw.
    await loop.bg_tasks.flush_delivered(ctx=ctx)
    recs = [TaskRecord.model_validate_json(await store.load(k)) for k in keys]
    assert all(r.status == TaskStatus.DELIVERED for r in recs)


# --- greppable progress log + elapsed time (Phase 2) --------------------------


async def test_backgrounded_bash_writes_greppable_log(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord

    store = InMemoryCheckpointStore()
    ctx = _ctx_with_store(tmp_path, store)
    loop = _loop(ctx, path=[])
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo HELLO && sleep 5"
    )
    assert task_id is not None

    await asyncio.sleep(0.2)  # let the command emit "HELLO" before flushing
    await _flush(loop.bg_tasks, ctx)

    keys = await store.list_keys(task_prefix(ctx.session_key))
    rec = TaskRecord.model_validate_json(await store.load(keys[0]))
    # The record indexes an agent-readable .grasp/tasks log holding the output.
    assert rec.output_path is not None
    assert ".grasp/tasks" in rec.output_path
    log = _Path(rec.output_path)
    assert log.exists()
    assert "HELLO" in log.read_text()

    await loop.bg_tasks.cancel_all(ctx=ctx)


async def test_backgrounded_bash_writes_log_without_store(tmp_path: Path) -> None:
    """
    The Grep-able ``.grasp/tasks`` log is a file-backend artifact: a file backend
    alone writes it — no checkpoint store required (the store only adds the
    durable record + resume pointer). This is the data-copilot demo's setup: a
    sandbox environment (→ file backend), a sub-agent lineage, and no store.
    """
    ctx = _ctx(tmp_path)  # environment → file backend, but NO checkpoint store
    loop = _loop(ctx, path=["data_engineer"])  # sub-agent lineage, as in the demo
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo HELLO && sleep 5"
    )
    assert task_id is not None

    await asyncio.sleep(0.2)
    await _flush(loop.bg_tasks, ctx)

    logs = list((tmp_path / ".grasp" / "tasks").glob("*.log"))
    assert logs, "no log written with a file backend but no checkpoint store"
    assert any("HELLO" in p.read_text() for p in logs)

    await loop.bg_tasks.cancel_all(ctx=ctx)


async def test_resume_interrupted_points_at_log(tmp_path: Path) -> None:
    import contextlib as _contextlib

    from grasp_agents.agent.background_tasks import BackgroundTaskManager
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore

    store = InMemoryCheckpointStore()
    ctx = _ctx_with_store(tmp_path, store)
    loop = _loop(ctx, path=[])
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo HELLO && sleep 5"
    )
    assert task_id is not None

    await asyncio.sleep(0.2)
    await _flush(
        loop.bg_tasks, ctx
    )  # mirrors the .grasp log (output_path set at launch)

    # Simulate a crash: drop the in-flight task without finalizing the record.
    for pt in list(loop.bg_tasks._tasks.values()):  # pyright: ignore[reportPrivateUsage]
        pt.task.cancel()
        with _contextlib.suppress(asyncio.CancelledError):
            await pt.task

    transcript = LLMAgentTranscript()
    transcript.messages = [InputMessageItem.from_text("sys", role="system")]
    mgr2 = BackgroundTaskManager(
        agent_name="t", transcript=transcript, tools={}, path=[]
    )
    await mgr2.resume_durable(ctx=ctx, exec_id="t")

    joined = "\n".join(str(m) for m in transcript.messages)
    assert "Resumed from a checkpoint" in joined  # framing
    assert "interrupted" in joined
    assert "output_file" in joined  # points the agent at the log
    assert ".grasp/tasks" in joined
    assert "ran_for" in joined  # elapsed


async def test_completion_note_reports_elapsed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    loop = _loop(ctx)
    _note, task_id = await _bg(
        loop, Bash(auto_background_at=0.1), "echo hi && sleep 0.5"
    )
    assert task_id is not None

    await loop.bg_tasks.wait_idle()
    notes = await _drain_notes(loop.bg_tasks, ctx)
    assert len(notes) == 1
    assert "ran_for" in notes[0]  # elapsed is surfaced in the completion note


# The persistent-session tool lives in test_bash_session.py.
