"""
Context compaction: ``Compactor`` planning and the summarizing fold in
the agent loop (``context.compaction``, ``agent.context_window``,
``agent.agent_loop``).

Covers rounds and the kept window, the threshold, what stays pinned, the summary
request sent on the agent's own request (prefix, tools, usage, hooks), retries
of tool-call answers and oversized requests, custom compactors, folds
superseding each other, compaction before new input, failure handling, the
context-window-error path, persistence and rollback.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context import (
    COMPACTION_REQUEST_TAG,
    COMPACTION_SECTION_NAME,
    SUMMARY_TAG,
    Compactor,
    ContextBudget,
    InputAttachment,
    count_rounds,
)
from grasp_agents.context.compaction import (
    MAX_COMPACTION_FAILURES,
    compaction_request,
    keep_recent_start,
    round_starts,
    summary_text,
)
from grasp_agents.context.projection import fold_summary_text, is_fold_summary
from grasp_agents.context.system_reminder import wrap_in_system_reminder
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpoint
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import CompactionDoneEvent, CompactionStartEvent
from grasp_agents.types.folds import CompactionPlan, FoldSpec
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmBadRequestError, LlmContextWindowError
from grasp_agents.types.response import Response, ResponseUsage
from tests._helpers import MockLLM, _text_response, _tool_call_response

_REQ = httpx.Request("POST", "https://x")


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _call(call_id: str) -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=call_id, name="big", arguments="{}")


def _result(call_id: str, output: str = "ok") -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id=call_id, output=output)


def _answer(text: str) -> OutputMessageItem:
    return OutputMessageItem(content=[OutputMessageText(text=text)], status="completed")


def _reasoning() -> ReasoningItem:
    return ReasoningItem(summary=[ReasoningSummary(text="thinking")])


def _budget(max_input: int) -> ContextBudget:
    return ContextBudget("mock", max_input_tokens=max_input, buffer_tokens=0)


def _overflow() -> LlmContextWindowError:
    return LlmContextWindowError(
        "prompt is too long", response=httpx.Response(400, request=_REQ), body=None
    )


def _is_summary_request(view: Sequence[InputItem]) -> bool:
    last = view[-1] if view else None
    return isinstance(last, InputMessageItem) and last.text.startswith(
        f"<{COMPACTION_REQUEST_TAG}>"
    )


def _plan(
    compactor: Compactor,
    messages: Sequence[InputItem],
    *,
    input_tokens: int = 1_000_000,
    folds: Sequence[FoldSpec] = (),
    running_call_ids: frozenset[str] = frozenset(),
    force: bool = False,
) -> CompactionPlan | None:
    return compactor.plan(
        messages,
        input_tokens=input_tokens,
        folds=folds,
        running_call_ids=running_call_ids,
        force=force,
    )


def _rounds(n: int, output: str = "ok") -> list[InputItem]:
    """``n`` rounds: a question, then a tool call and its result."""
    log: list[InputItem] = []
    for i in range(n):
        log += [_user(f"q{i}"), _call(f"c{i}"), _result(f"c{i}", output)]
    return log


def _compactor(**kwargs: Any) -> Compactor:
    """A compactor that always folds; the kept window is capped by rounds only."""
    kwargs.setdefault("budget", _budget(1))
    kwargs.setdefault("keep_recent_tokens", 10**9)
    return Compactor(**kwargs)


# --- Rounds ---


def test_a_round_is_inputs_response_and_its_tool_results() -> None:
    log: list[InputItem] = [
        _user("q1"),
        _reasoning(),
        _call("c1"),
        _result("c1"),
        _reasoning(),
        _answer("a1"),
        _user("q2"),
    ]
    assert round_starts(log) == [0, 4, 6]


def test_rounds_do_not_depend_on_reasoning_items() -> None:
    plain: list[InputItem] = [
        _user("q1"),
        _call("c1"),
        _result("c1"),
        _answer("a1"),
        _user("q2"),
    ]
    assert count_rounds(plain) == 3


def test_no_round_starts_while_a_call_is_open() -> None:
    log: list[InputItem] = [_user("q"), _call("c1"), _call("c2"), _result("c1")]
    assert round_starts(log) == [0]


def test_kept_window_shrinks_to_its_token_cap_but_keeps_the_latest_round() -> None:
    log = _rounds(3, output="Z" * 4000)
    starts = round_starts(log)
    assert keep_recent_start(log, 3) == 0
    assert keep_recent_start(log, 3, keep_recent_tokens=1) == starts[-1]


# --- Compactor: gating and span ---


def test_under_threshold_is_no_plan() -> None:
    compactor = Compactor(budget=_budget(1000))
    assert _plan(compactor, _rounds(5), input_tokens=10) is None


def test_target_tokens_lowers_the_threshold() -> None:
    compactor = Compactor(
        budget=_budget(10_000_000), target_tokens=100, keep_recent_rounds=2
    )
    assert compactor.threshold == 100
    assert _plan(compactor, _rounds(5), input_tokens=100) is None
    assert _plan(compactor, _rounds(5), input_tokens=101) is not None


def test_folds_everything_before_the_recent_rounds() -> None:
    log = _rounds(5)
    plan = _plan(_compactor(keep_recent_rounds=2), log)
    assert plan is not None
    assert plan.end == round_starts(log)[-2]


def test_window_over_its_cap_shrinks_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = _rounds(3, output="Z" * 4000)
    compactor = Compactor(
        budget=_budget(1), keep_recent_rounds=8, keep_recent_tokens=1500
    )
    with caplog.at_level(logging.WARNING, logger="grasp_agents.context.compaction"):
        plan = _plan(compactor, log)
    assert plan is not None
    assert plan.end == round_starts(log)[-1]
    assert any("instead of 8" in r.getMessage() for r in caplog.records)


def test_force_plans_even_when_everything_is_recent() -> None:
    log = _rounds(3)
    compactor = Compactor(budget=_budget(10_000_000))
    assert _plan(compactor, log, input_tokens=1) is None
    plan = _plan(compactor, log, input_tokens=1, force=True)
    assert plan is not None
    assert plan.end == round_starts(log)[-1]


def test_before_new_input_the_input_is_the_newest_kept_round() -> None:
    log: list[InputItem] = []
    for i in range(4):
        log += [_user(f"q{i}"), _answer(f"a{i}")]
    starts = round_starts(log)

    plan = _plan(_compactor(keep_recent_rounds=3), log)
    assert plan is not None
    assert plan.end == starts[2]  # two earlier rounds, then the new input

    plan = _plan(_compactor(keep_recent_rounds=1), log)
    assert plan is not None
    assert plan.end == len(log)  # everything; the newest question stays pinned
    assert plan.pinned[-1] == starts[-1]


def test_a_single_round_cannot_be_folded() -> None:
    assert _plan(_compactor(), _rounds(1), force=True) is None


def test_a_new_fold_supersedes_the_previous_one() -> None:
    log = _rounds(6)
    starts = round_starts(log)
    previous = FoldSpec(start=0, end=starts[2], summary="OLD")
    plan = _plan(_compactor(keep_recent_rounds=2), log, folds=[previous])
    assert plan is not None
    assert plan.to_fold("NEW") == FoldSpec(
        start=0, end=starts[-2], summary="NEW", pinned=plan.pinned
    )


def test_no_progress_past_the_previous_fold_is_no_plan() -> None:
    log = _rounds(4)
    previous = FoldSpec(start=0, end=round_starts(log)[-2], summary="OLD")
    assert _plan(_compactor(keep_recent_rounds=2), log, folds=[previous]) is None


# --- What stays pinned ---


def test_pins_the_latest_user_messages_before_the_window() -> None:
    reminder = _user(wrap_in_system_reminder("12:00", subject="time"))
    task_note = _user("<task_notification> done </task_notification>")
    log: list[InputItem] = [
        _user("first"),
        _answer("a"),
        _user("second"),
        _answer("b"),
        reminder,
        _answer("c"),
        _user("third"),
        _answer("d"),
        task_note,
        _answer("e"),
        _user("fourth"),
        _answer("f"),
        _user("now"),
    ]
    plan = _plan(_compactor(keep_recent_rounds=1, pinned_user_messages=3), log)
    assert plan is not None
    # The newest three user messages; framework notes are not theirs.
    assert plan.pinned == [2, 6, 10]


def test_the_newest_user_message_is_pinned_whatever_its_size() -> None:
    log: list[InputItem] = [
        _user("short"),
        _answer("a"),
        _user("x" * 10_000),
        _call("c1"),
        _result("c1"),
        _answer("b"),
    ]
    plan = _plan(_compactor(keep_recent_rounds=1, pinned_user_tokens=100), log)
    assert plan is not None
    # Mid-run: the run's own request is folded, yet stays verbatim. The earlier
    # message no longer fits the token cap.
    assert plan.pinned == [2]


def test_earlier_pins_stop_at_the_first_message_over_the_token_cap() -> None:
    log: list[InputItem] = [
        _user("short"),
        _answer("a"),
        _user("x" * 10_000),
        _answer("b"),
        _user("tiny"),
        _answer("c"),
        _user("now"),
    ]
    plan = _plan(_compactor(keep_recent_rounds=1, pinned_user_tokens=100), log)
    assert plan is not None
    assert plan.pinned == [4]


def test_launch_notes_of_running_tasks_are_pinned() -> None:
    log = _rounds(3)
    plan = _plan(
        _compactor(keep_recent_rounds=1), log, running_call_ids=frozenset({"c0"})
    )
    assert plan is not None
    # The questions, and c0's launch note between them; c1's task is done.
    assert plan.pinned == [0, 2, 3]


def test_at_least_one_user_message_is_pinned() -> None:
    with pytest.raises(ValueError, match="pinned_user_messages"):
        Compactor(pinned_user_messages=0)


# --- The summary request and its reply ---


def test_the_request_carries_the_instructions_in_a_compaction_tag() -> None:
    request = compaction_request("Summarize it.")
    assert request.role == "user"
    assert request.text == (
        f"<{COMPACTION_REQUEST_TAG}>\nSummarize it.\n</{COMPACTION_REQUEST_TAG}>"
    )


def test_an_echoed_summary_wrapper_is_stripped() -> None:
    response = _text_response(f"<{SUMMARY_TAG}>\nS\n</{SUMMARY_TAG}>")
    assert summary_text(response) == "S"


def test_a_reply_that_calls_a_tool_holds_no_summary() -> None:
    assert summary_text(_tool_call_response("big", "{}", "x1")) == ""


@pytest.mark.asyncio
async def test_summarize_appends_the_request_and_reports_each_response() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "x1"),
            _text_response("SUMMARY"),
        ]
    )
    context = [_user("header")]
    body = _rounds(2)
    seen: list[Response] = []
    compactor = _compactor(instructions="Summarize it.")

    summary = await compactor.summarize(
        llm,
        initial_context=context,
        body=body,
        tools=None,
        tool_choice=None,
        output_schema=None,
        stream=False,
        on_response=seen.append,
    )

    assert summary == "SUMMARY"
    assert len(seen) == 2  # the tool-call reply too
    calls = llm.calls  # type: ignore[attr-defined]
    *prefix, request = calls[0].input
    assert prefix == [*context, *body]
    assert request.text == compaction_request("Summarize it.").text
    assert calls[1].input == calls[0].input


# --- Through the agent loop ---


@dataclass(frozen=True)
class _RecordingLLM(MockLLM):
    """MockLLM recording each call's input, tools, tool_choice and schema."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "calls", [])

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Any = None,
        output_schema: Any = None,
        tool_choice: Any = None,
        **extra_llm_settings: Any,
    ) -> Response:
        self.calls.append(  # type: ignore[attr-defined]
            SimpleNamespace(
                input=list(input),
                tools=tools,
                tool_choice=tool_choice,
                output_schema=output_schema,
            )
        )
        return await super()._generate_response_once(
            input,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )


class _BigInput(BaseModel):
    pass


_BIG = "Z" * 5000


class _BigTool(BaseTool[_BigInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="big", description="Returns a large output.")
        object.__setattr__(self, "runs", 0)

    async def _run(self, inp: _BigInput, **kwargs: Any) -> str:
        object.__setattr__(self, "runs", self.runs + 1)  # type: ignore[attr-defined]
        return _BIG


def _agent(
    llm: MockLLM,
    *,
    compactor: Compactor | None = None,
    ctx: SessionContext[None] | None = None,
    stream_llm: bool = False,
) -> tuple[LLMAgent[str, str, None], _BigTool]:
    tool = _BigTool()
    agent = LLMAgent[str, str, None](
        name="a",
        ctx=ctx or SessionContext(),
        llm=llm,
        tools=[tool],
        stream_llm=stream_llm,
    )
    agent.add_compaction(
        compactor or Compactor(budget=_budget(5), keep_recent_rounds=1)
    )
    return agent, tool


def _summaries(view: Sequence[InputItem]) -> list[str]:
    return [m.text for m in view if is_fold_summary(m)]


@pytest.mark.parametrize("stream_llm", [False, True])
@pytest.mark.asyncio
async def test_summary_request_extends_the_agents_own_request(
    *, stream_llm: bool
) -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm, stream_llm=stream_llm)
    out = await agent.run("go")
    assert out.payloads[0] == "done"

    calls = llm.calls  # type: ignore[attr-defined]
    before, summary_call, after = calls[1], calls[2], calls[3]
    assert _is_summary_request(summary_call.input)
    # Same prefix, tools, tool choice and schema as the request before it: the
    # provider's prompt cache serves it.
    assert summary_call.input[: len(before.input)] == before.input
    assert summary_call.tools == before.tools
    assert summary_call.tool_choice == before.tool_choice
    assert summary_call.output_schema == before.output_schema

    # The next generation sees the pinned question, then the summary.
    assert _summaries(after.input) == [fold_summary_text("SUMMARY")]
    texts = [m.text for m in after.input if isinstance(m, InputMessageItem)]
    assert texts.index("go") < texts.index(fold_summary_text("SUMMARY"))
    assert not _is_summary_request(after.input)
    # The log keeps every original message and never holds the request.
    assert not any(
        isinstance(m, InputMessageItem) and COMPACTION_REQUEST_TAG in m.text
        for m in agent.transcript
    )


@pytest.mark.asyncio
async def test_summary_call_is_accounted_but_runs_no_llm_hooks() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    ctx: SessionContext[None] = SessionContext()
    agent, _ = _agent(llm, ctx=ctx)
    before: list[int] = []
    after: list[str] = []

    @agent.add_before_llm_hook
    async def _before(*, exec_id: str, turn: int, extra_llm_settings: Any) -> None:
        del exec_id, extra_llm_settings
        before.append(turn)

    @agent.add_after_llm_hook
    async def _after(response: Response, *, exec_id: str, turn: int) -> None:
        del exec_id, turn
        after.append(response.output_text)

    await agent.run("go")
    assert len(before) == 3
    assert after == ["", "", "done"]
    assert ctx.usage_tracker.total_usage.input_tokens == 4 * 10


@pytest.mark.asyncio
async def test_a_tool_call_answer_is_retried_and_never_executed() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _tool_call_response("big", "{}", "sneaky"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, tool = _agent(llm)
    await agent.run("go")
    assert tool.runs == 2  # type: ignore[attr-defined]
    calls = llm.calls  # type: ignore[attr-defined]
    # The retry is the same request: tool settings untouched, so cached.
    assert calls[3].input == calls[2].input
    assert calls[3].tool_choice == calls[2].tool_choice
    assert agent.agent_ctx.cw.folds[-1].summary == "SUMMARY"


@pytest.mark.asyncio
async def test_no_summary_after_every_attempt_records_no_fold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _tool_call_response("big", "{}", "x1"),
            _text_response("  "),
            _tool_call_response("big", "{}", "x2"),
            _text_response("done"),
        ]
    )
    agent, tool = _agent(llm)
    with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.context_window"):
        out = await agent.run("go")
    assert out.payloads[0] == "done"
    assert tool.runs == 2  # type: ignore[attr-defined]
    assert agent.agent_ctx.cw.folds == []
    assert any("compaction failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_failed_summary_does_not_fail_the_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass(frozen=True)
    class _FailingSummaryLLM(MockLLM):
        async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
            if _is_summary_request(input):
                raise LlmBadRequestError(
                    "boom", response=httpx.Response(400, request=_REQ), body=None
                )
            return await super()._generate_response_once(input, **kwargs)

    llm = _FailingSummaryLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.context_window"):
        out = await agent.run("go")
    assert out.payloads[0] == "done"
    assert agent.agent_ctx.cw.folds == []
    assert any("compaction failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_failures_stop_turn_boundary_compaction_but_not_recovery() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            *[_tool_call_response("big", "{}", f"c{i}") for i in range(2, 7)],
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    summaries = 0

    async def _failing(*, agent_ctx: AgentContext, ctx: SessionContext[Any]) -> str:
        nonlocal summaries
        del agent_ctx, ctx
        summaries += 1
        return ""

    agent.add_custom_compactor(_failing)
    await agent.run("go")
    assert summaries == MAX_COMPACTION_FAILURES


@pytest.mark.asyncio
async def test_an_oversized_summary_request_is_shrunk_and_retried() -> None:
    @dataclass(frozen=True)
    class _TightLLM(_RecordingLLM):
        def __post_init__(self) -> None:
            super().__post_init__()
            object.__setattr__(self, "overflows", 2)

        async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
            if _is_summary_request(input) and self.overflows:  # type: ignore[attr-defined]
                object.__setattr__(self, "overflows", self.overflows - 1)  # type: ignore[attr-defined]
                self.calls.append(SimpleNamespace(input=list(input)))  # type: ignore[attr-defined]
                raise _overflow()
            return await super()._generate_response_once(input, **kwargs)

    llm = _TightLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("S"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    await agent.run("go")
    assert agent.agent_ctx.cw.folds[-1].summary == "S"

    requests = [c.input for c in llm.calls if _is_summary_request(c.input)]  # type: ignore[attr-defined]
    full, collapsed, dropped = requests[:3]

    def _outputs(view: Sequence[InputItem]) -> list[str]:
        return [m.text for m in view if isinstance(m, FunctionToolOutputItem)]

    assert all(len(o) == len(_BIG) for o in _outputs(full))
    # Older outputs collapsed; the latest round in full.
    assert all(len(o) < len(_BIG) for o in _outputs(collapsed)[:-1])
    assert len(_outputs(collapsed)[-1]) == len(_BIG)
    # Then the oldest half of the rounds is left out, with a marker.
    assert any(
        isinstance(m, InputMessageItem) and "left out of this request" in m.text
        for m in dropped
    )
    assert len(_outputs(dropped)) < len(_outputs(collapsed))


@pytest.mark.asyncio
async def test_summaries_supersede_rather_than_stack() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("S1"),
            _tool_call_response("big", "{}", "c3"),
            _text_response("S2"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    await agent.run("go")

    calls = llm.calls  # type: ignore[attr-defined]
    second_request = calls[4]
    assert _is_summary_request(second_request.input)
    # The second summary is written from a view that holds the first.
    assert _summaries(second_request.input) == [fold_summary_text("S1")]
    assert _summaries(calls[-1].input) == [fold_summary_text("S2")]
    assert [f.start for f in agent.agent_ctx.cw.folds] == [0, 0]


@pytest.mark.asyncio
async def test_compaction_event_reports_rounds_and_pins() -> None:
    llm = MockLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    events = [event async for event in agent.run_stream("go")]
    compactions = [e for e in events if isinstance(e, CompactionDoneEvent)]
    assert len(compactions) == 1
    info = compactions[0].data
    assert (info.folded_rounds, info.kept_rounds, info.pinned_messages) == (1, 1, 1)
    assert info.summary == "SUMMARY"
    assert not info.forced
    assert compactions[0].source == "a"


@pytest.mark.asyncio
async def test_a_running_tasks_launch_note_stays_in_front_of_the_summary() -> None:
    release = asyncio.Event()

    @function_tool(auto_background_at=0, blocks_final_answer=False)
    async def watch(topic: str) -> str:
        """Watch a topic."""
        await release.wait()
        return f"seen {topic}"

    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("watch", '{"topic": "x"}', "w1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent = LLMAgent[str, str, None](
        name="a", ctx=SessionContext(), llm=llm, tools=[_BigTool(), watch]
    )
    agent.add_compaction(Compactor(budget=_budget(5), keep_recent_rounds=1))
    try:
        await agent.run("go")
    finally:
        release.set()
        await agent.aclose()

    after = llm.calls[-1].input  # type: ignore[attr-defined]
    texts = [m.text for m in after if isinstance(m, InputMessageItem)]
    note = next(t for t in texts if t.startswith("<task_notification>"))
    assert "w1" in note
    assert texts.index("go") < texts.index(note)
    assert texts.index(note) < texts.index(fold_summary_text("SUMMARY"))


# --- Registration and custom compactors ---


@pytest.mark.asyncio
async def test_add_compaction_registers_the_section_and_injects_the_budget() -> None:
    agent = LLMAgent[str, str, None](
        name="a", ctx=SessionContext(), llm=MockLLM(responses_queue=[])
    )
    compactor = agent.add_compaction()
    assert agent.agent_ctx.cw.compactor is compactor
    assert compactor.budget is not None
    names = [s.name for s in agent.system_prompt_sections]
    assert COMPACTION_SECTION_NAME in names


class _Verdict(BaseModel):
    ok: bool


def test_own_summaries_need_a_reply_that_can_be_text() -> None:
    structured = LLMAgent[str, _Verdict, None](
        name="a", ctx=SessionContext(), llm=MockLLM(responses_queue=[])
    )
    with pytest.raises(ValueError, match="output schema"):
        structured.add_compaction()

    react = LLMAgent[str, str, None](
        name="b",
        ctx=SessionContext(),
        llm=MockLLM(responses_queue=[]),
        force_react_mode=True,
    )
    with pytest.raises(ValueError, match="force_react_mode"):
        react.add_compaction()


@pytest.mark.asyncio
async def test_a_custom_compactor_writes_the_summary() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("done"),
        ]
    )
    ctx: SessionContext[None] = SessionContext()
    agent, _ = _agent(llm, ctx=ctx)
    seen: list[tuple[int, SessionContext[Any]]] = []

    @agent.add_custom_compactor
    async def _summarize(*, agent_ctx: AgentContext, ctx: SessionContext[Any]) -> str:
        seen.append((len(agent_ctx.cw.transcript), ctx))
        return "  CUSTOM  "

    await agent.run("go")
    assert seen == [(5, ctx)]
    assert agent.agent_ctx.cw.folds[-1].summary == "CUSTOM"
    calls = llm.calls  # type: ignore[attr-defined]
    assert len(calls) == 3  # the model wrote no summary
    assert not any(_is_summary_request(c.input) for c in calls)


@pytest.mark.asyncio
async def test_a_custom_compactor_enables_compaction_for_any_agent() -> None:
    agent = LLMAgent[str, _Verdict, None](
        name="a", ctx=SessionContext(), llm=MockLLM(responses_queue=[])
    )

    async def _summarize(*, agent_ctx: AgentContext, ctx: SessionContext[Any]) -> str:
        del agent_ctx, ctx
        return "S"

    agent.add_custom_compactor(_summarize)
    compactor = agent.add_compaction(Compactor(target_tokens=1000))
    assert agent.agent_ctx.cw.compactor is compactor


# --- Compaction before new input ---


@pytest.mark.asyncio
async def test_a_steps_input_lands_after_the_fold() -> None:
    llm = MockLLM(responses_queue=[_text_response(a) for a in ("a1", "a2", "a3")])
    agent = LLMAgent[str, str, None](name="a", ctx=SessionContext(), llm=llm)
    compactor = agent.add_compaction(
        Compactor(budget=_budget(10**9), keep_recent_rounds=1)
    )
    logs: list[list[str]] = []
    views: list[list[str]] = []

    @agent.add_custom_compactor
    async def _summarize(*, agent_ctx: AgentContext, ctx: SessionContext[Any]) -> str:
        del ctx
        logs.append(
            [m.text for m in agent_ctx.cw.transcript if isinstance(m, InputMessageItem)]
        )
        return "S"

    await agent.run("q1")
    await agent.run("q2")

    def compute(*, messages: Sequence[InputItem] | None = None, **_: Any) -> None:
        views.append(
            [m.text for m in messages or [] if isinstance(m, InputMessageItem)]
        )

    agent.add_input_attachment(InputAttachment(name="probe", compute=compute))
    compactor.budget = _budget(1)
    events = [event async for event in agent.run_stream("q3")]

    # The fold came first: the summary never covers q3, and q3's attachments
    # see the summarized conversation.
    assert logs[0] == ["q1", "q2"]
    assert fold_summary_text("S") in views[0]
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("CompactionStartEvent") < kinds.index("UserMessageEvent")


# --- Compaction start event ---


@pytest.mark.asyncio
async def test_start_event_arrives_before_the_summary_returns() -> None:
    release = asyncio.Event()

    @dataclass(frozen=True)
    class _GatedSummaryLLM(MockLLM):
        async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
            if _is_summary_request(input):
                await release.wait()
            return await super()._generate_response_once(input, **kwargs)

    llm = _GatedSummaryLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    kinds: list[str] = []
    # The summary only returns once the start event has reached the consumer;
    # a start event held back until the fold would never arrive.
    async with asyncio.timeout(5):
        async for event in agent.run_stream("go"):
            if isinstance(event, CompactionStartEvent):
                kinds.append("start")
                release.set()
            elif isinstance(event, CompactionDoneEvent):
                kinds.append("fold")
    assert kinds == ["start", "fold"]


@pytest.mark.asyncio
async def test_start_event_reports_the_context_being_compacted() -> None:
    llm = MockLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(
        llm,
        compactor=Compactor(
            budget=ContextBudget("mock", max_input_tokens=5, buffer_tokens=0),
            keep_recent_rounds=1,
        ),
    )
    events = [event async for event in agent.run_stream("go")]
    start = next(e for e in events if isinstance(e, CompactionStartEvent))
    assert start.source == "a"
    assert start.data.context_window == 5
    assert start.data.context_tokens > 5
    assert not start.data.forced


@pytest.mark.asyncio
async def test_no_start_event_without_a_fold_to_make() -> None:
    llm = MockLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm, compactor=Compactor(budget=_budget(10_000_000)))
    events = [event async for event in agent.run_stream("go")]
    assert not any(isinstance(e, CompactionStartEvent) for e in events)


@pytest.mark.asyncio
async def test_a_failed_summary_has_a_start_but_no_fold() -> None:
    @dataclass(frozen=True)
    class _FailingSummaryLLM(MockLLM):
        async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
            if _is_summary_request(input):
                raise LlmBadRequestError(
                    "boom", response=httpx.Response(400, request=_REQ), body=None
                )
            return await super()._generate_response_once(input, **kwargs)

    llm = _FailingSummaryLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    events = [event async for event in agent.run_stream("go")]
    assert sum(isinstance(e, CompactionStartEvent) for e in events) == 1
    assert not any(isinstance(e, CompactionDoneEvent) for e in events)


@pytest.mark.asyncio
async def test_leaving_the_stream_at_the_start_event_sends_no_summary_request() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
        ]
    )
    agent, _ = _agent(llm)
    stream = agent.run_stream("go")
    async for event in stream:
        if isinstance(event, CompactionStartEvent):
            break
    await stream.aclose()  # type: ignore[attr-defined]
    assert not any(_is_summary_request(c.input) for c in llm.calls)  # type: ignore[attr-defined]


# --- Context-window error → forced compaction + retry ---


@dataclass(frozen=True)
class _OverflowOnceLLM(MockLLM):
    """Raises a context-window error on its third call, then behaves normally."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "_calls", 0)

    async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
        object.__setattr__(self, "_calls", self._calls + 1)  # type: ignore[attr-defined]
        if self._calls == 3:  # type: ignore[attr-defined]
            raise _overflow()
        return await super()._generate_response_once(input, **kwargs)


@pytest.mark.asyncio
async def test_context_window_error_compacts_and_retries() -> None:
    llm = _OverflowOnceLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUM"),
            _text_response("recovered"),
        ]
    )
    # A huge budget: only the forced path compacts.
    agent, _ = _agent(llm, compactor=Compactor(budget=_budget(10_000_000)))
    events = [event async for event in agent.run_stream("go")]

    assert events[-1].data.payloads[0] == "recovered"  # type: ignore[union-attr]
    compactions = [e for e in events if isinstance(e, CompactionDoneEvent)]
    assert [c.data.forced for c in compactions] == [True]
    starts = [e for e in events if isinstance(e, CompactionStartEvent)]
    assert [s.data.forced for s in starts] == [True]
    assert agent.agent_ctx.cw.folds[-1].summary == "SUM"


# --- Persistence and rollback ---


def test_folds_and_usage_round_trip_through_checkpoint() -> None:
    checkpoint = AgentCheckpoint(
        session_key="s",
        processor_name="a",
        messages=[],
        folds=[FoldSpec(start=0, end=3, summary="S", pinned=[1])],
        usage=ResponseUsage(input_tokens=1234, output_tokens=5),
    )
    reloaded = AgentCheckpoint.model_validate_json(checkpoint.model_dump_json())
    assert reloaded.folds == checkpoint.folds
    assert reloaded.usage == checkpoint.usage


@pytest.mark.asyncio
async def test_folds_and_usage_restored_on_resume_without_a_new_summary() -> None:
    store = InMemoryCheckpointStore()
    ctx: SessionContext[None] = SessionContext(checkpoint_store=store, session_key="s1")
    agent, _ = _agent(
        MockLLM(
            responses_queue=[
                _tool_call_response("big", "{}", "c1"),
                _tool_call_response("big", "{}", "c2"),
                _text_response("SUM"),
                _text_response("done"),
            ]
        ),
        ctx=ctx,
    )
    await agent.run("go")
    folds = list(agent.agent_ctx.cw.folds)
    usage = agent.agent_ctx.cw.last_usage
    assert folds
    assert usage is not None

    ctx2: SessionContext[None] = SessionContext(
        checkpoint_store=store, session_key="s1"
    )
    # One reply: a re-summarization would exhaust the queue and fail the run.
    agent2, _ = _agent(
        MockLLM(responses_queue=[_text_response("after")]),
        ctx=ctx2,
        compactor=Compactor(budget=_budget(10_000_000)),
    )
    await agent2.load_checkpoint(exec_id="x")
    assert agent2.agent_ctx.cw.last_usage == usage
    out = await agent2.run("again")
    assert out.payloads[0] == "after"
    assert agent2.agent_ctx.cw.folds == folds


class _CrashingStore(InMemoryCheckpointStore):
    """Raises on the first save once armed, as a process dying mid-save would."""

    armed = False

    async def save(self, key: str, data: bytes) -> None:
        if self.armed:
            self.armed = False
            raise OSError("crash")
        await super().save(key, data)


@pytest.mark.asyncio
async def test_a_crash_before_the_fold_is_saved_redoes_it_on_redelivery() -> None:
    store = _CrashingStore()

    def _build(*replies: str) -> tuple[LLMAgent[str, str, None], Compactor]:
        agent = LLMAgent[str, str, None](
            name="a",
            ctx=SessionContext(checkpoint_store=store, session_key="s1"),
            llm=MockLLM(responses_queue=[_text_response(r) for r in replies]),
        )
        compactor = agent.add_compaction(
            Compactor(budget=_budget(10**9), keep_recent_rounds=1)
        )
        return agent, compactor

    agent, compactor = _build("a1", "a2", "S1")
    await agent.run("q1", step=0)
    await agent.run("q2", step=1)
    compactor.budget = _budget(1)
    store.armed = True
    # The fold is made before q3 lands; the save that would persist both fails.
    with pytest.raises(ProcRunError):
        await agent.run("q3", step=2)

    resumed, compactor = _build("S2", "a3")
    compactor.budget = _budget(1)
    out = await resumed.run("q3", step=2)

    assert out.payloads[0] == "a3"
    inputs = [m.text for m in resumed.transcript if isinstance(m, InputMessageItem)]
    assert inputs == ["q1", "q2", "q3"]
    assert [f.summary for f in resumed.agent_ctx.cw.folds] == ["S2"]


@pytest.mark.asyncio
async def test_rollback_below_the_newest_fold_falls_back_to_the_previous() -> None:
    from tests.durability.test_sessions import _make_agent

    store = InMemoryCheckpointStore()
    agent, _ = _make_agent(
        [_text_response(a) for a in ("a0", "a1", "a2")],
        session_key="s1",
        store=store,
    )
    await agent.run("q0", step=0)
    await agent.run("q1", step=1)
    n_after_1 = len(agent.transcript.messages)
    await agent.run("q2", step=2)

    early = FoldSpec(start=0, end=2, summary="early")
    late = FoldSpec(start=0, end=n_after_1 + 1, summary="late")
    agent.agent_ctx.cw.folds = [early, late]
    await agent.rollback_to_step(1)

    assert agent.agent_ctx.cw.folds == [early]
    view = agent.agent_ctx.cw.folded_messages()
    assert _summaries(view) == [fold_summary_text("early")]


@pytest.mark.asyncio
async def test_attachments_see_the_folded_conversation() -> None:
    agent = LLMAgent[str, str, None](
        name="a",
        ctx=SessionContext(),
        llm=MockLLM(responses_queue=[_text_response("ok")]),
    )
    agent.replace_transcript([_user("old"), _answer("a"), _user("older")])
    agent.agent_ctx.cw.folds = [FoldSpec(start=0, end=2, summary="S")]
    seen: list[list[str]] = []

    def compute(*, messages: Sequence[InputItem] | None = None, **_: Any) -> None:
        seen.append([m.text for m in messages or [] if isinstance(m, InputMessageItem)])

    agent.add_input_attachment(InputAttachment(name="probe", compute=compute))
    await agent.run("new")
    assert seen == [[fold_summary_text("S"), "older"]]
