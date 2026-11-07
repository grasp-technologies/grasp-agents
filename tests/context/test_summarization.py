"""
Summarizing compaction: ``SummarizingCompactor`` and its wiring in the agent loop
(``context.compaction``, ``agent.context_window``, ``agent.agent_loop``).

Covers rounds and the kept window, the threshold, pinned user messages, the
summary request sent on the agent's own request (prefix, tools, usage, hooks),
recovery from tool-call answers and oversized requests, folds superseding each
other, failure handling, the context-window-error path, restoration after a
fold, persistence and rollback.
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

from grasp_agents.agent.context_window import (
    MAX_COMPACTION_FAILURES,
    ContextWindowManager,
)
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context import (
    COMPACTION_REQUEST_TAG,
    COMPACTION_SECTION_NAME,
    SUMMARY_TAG,
    ContextBudget,
    InputAttachment,
    SummarizingCompactor,
    count_rounds,
)
from grasp_agents.context.compaction import keep_recent_start, round_starts
from grasp_agents.context.projection import fold_summary_text, is_fold_summary
from grasp_agents.context.system_reminder import wrap_in_system_reminder
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpoint
from grasp_agents.hooks import ViewTransform
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.events import CompactionDoneEvent, CompactionStartEvent
from grasp_agents.types.folds import FoldSpec, TokenAnchor
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmBadRequestError, LlmContextWindowError
from grasp_agents.types.response import Response
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


class _FakeGenerate:
    """A ``CompactionGenerate`` replying from a queue (an exception is raised)."""

    def __init__(self, *replies: Response | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[SimpleNamespace] = []

    async def __call__(
        self,
        request: InputMessageItem,
        *,
        transform: ViewTransform | None = None,
        force_text: bool = False,
    ) -> Response:
        self.calls.append(
            SimpleNamespace(request=request, transform=transform, force_text=force_text)
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def _compact(
    compactor: SummarizingCompactor,
    messages: Sequence[InputItem],
    generate: _FakeGenerate,
    *,
    input_tokens: int = 1_000_000,
    folds: Sequence[FoldSpec] = (),
    force: bool = False,
) -> FoldSpec | None:
    return await compactor(
        messages,
        input_tokens=input_tokens,
        folds=folds,
        exec_id="x",
        generate=generate,
        force=force,
    )


def _rounds(n: int, output: str = "ok") -> list[InputItem]:
    """``n`` rounds: a question, then a tool call and its result."""
    log: list[InputItem] = []
    for i in range(n):
        log += [_user(f"q{i}"), _call(f"c{i}"), _result(f"c{i}", output)]
    return log


def _compactor(**kwargs: Any) -> SummarizingCompactor:
    """A compactor that always fires; the kept window is capped by rounds only."""
    kwargs.setdefault("budget", _budget(1))
    kwargs.setdefault("keep_recent_tokens", 10**9)
    return SummarizingCompactor(**kwargs)


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


# --- SummarizingCompactor: gating and span ---


@pytest.mark.asyncio
async def test_under_threshold_is_a_noop() -> None:
    generate = _FakeGenerate()
    compactor = SummarizingCompactor(budget=_budget(1000))
    assert await _compact(compactor, _rounds(5), generate, input_tokens=10) is None
    assert generate.calls == []


@pytest.mark.asyncio
async def test_target_tokens_lowers_the_threshold() -> None:
    compactor = SummarizingCompactor(
        budget=_budget(10_000_000), target_tokens=100, keep_recent_rounds=2
    )
    assert compactor.threshold == 100
    generate = _FakeGenerate(_text_response("S"))
    fold = await _compact(compactor, _rounds(5), generate, input_tokens=101)
    assert fold is not None


@pytest.mark.asyncio
async def test_folds_everything_before_the_recent_rounds() -> None:
    log = _rounds(5)
    compactor = _compactor(keep_recent_rounds=2)
    fold = await _compact(compactor, log, _FakeGenerate(_text_response("SUMMARY")))
    assert fold is not None
    assert (fold.start, fold.end) == (0, round_starts(log)[-2])
    assert fold.summary == "SUMMARY"


@pytest.mark.asyncio
async def test_window_over_its_cap_shrinks_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = _rounds(3, output="Z" * 4000)
    compactor = SummarizingCompactor(
        budget=_budget(1), keep_recent_rounds=8, keep_recent_tokens=1500
    )
    with caplog.at_level(logging.WARNING, logger="grasp_agents.context.compaction"):
        fold = await _compact(compactor, log, _FakeGenerate(_text_response("S")))
    assert fold is not None
    assert fold.end == round_starts(log)[-1]
    assert any("instead of 8" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_force_folds_even_when_everything_is_recent() -> None:
    log = _rounds(3)
    compactor = SummarizingCompactor(budget=_budget(10_000_000))
    generate = _FakeGenerate(_text_response("S"))
    assert await _compact(compactor, log, generate, input_tokens=1) is None
    fold = await _compact(compactor, log, generate, input_tokens=1, force=True)
    assert fold is not None
    assert fold.end == round_starts(log)[-1]


@pytest.mark.asyncio
async def test_a_single_round_cannot_be_folded() -> None:
    compactor = SummarizingCompactor(budget=_budget(1))
    fold = await _compact(compactor, _rounds(1), _FakeGenerate(), force=True)
    assert fold is None


@pytest.mark.asyncio
async def test_a_new_fold_supersedes_the_previous_one() -> None:
    log = _rounds(6)
    starts = round_starts(log)
    previous = FoldSpec(start=0, end=starts[2], summary="OLD")
    compactor = _compactor(keep_recent_rounds=2)
    fold = await _compact(
        compactor, log, _FakeGenerate(_text_response("NEW")), folds=[previous]
    )
    assert fold is not None
    assert (fold.start, fold.end) == (0, starts[-2])


@pytest.mark.asyncio
async def test_no_progress_past_the_previous_fold_is_no_fold() -> None:
    log = _rounds(4)
    starts = round_starts(log)
    previous = FoldSpec(start=0, end=starts[-2], summary="OLD")
    compactor = _compactor(keep_recent_rounds=2)
    generate = _FakeGenerate()
    assert await _compact(compactor, log, generate, folds=[previous]) is None
    assert generate.calls == []


# --- Pinned user messages ---


@pytest.mark.asyncio
async def test_pins_the_latest_user_messages_before_the_window() -> None:
    note = _user(wrap_in_system_reminder("task done", subject="task"))
    log: list[InputItem] = [
        _user("first"),
        _answer("a"),
        _user("second"),
        _answer("b"),
        note,
        _answer("c"),
        _user("third"),
        _answer("d"),
        _user("fourth"),
        _answer("e"),
        _user("now"),
    ]
    compactor = _compactor(keep_recent_rounds=1, pinned_user_messages=3)
    fold = await _compact(compactor, log, _FakeGenerate(_text_response("S")))
    assert fold is not None
    # The newest three learner messages; the framework note is not one.
    assert fold.pinned == [2, 6, 8]


@pytest.mark.asyncio
async def test_pins_stop_at_the_first_message_over_the_token_cap() -> None:
    log: list[InputItem] = [
        _user("short"),
        _answer("a"),
        _user("x" * 10_000),
        _answer("b"),
        _user("tiny"),
        _answer("c"),
        _user("now"),
    ]
    compactor = _compactor(keep_recent_rounds=1, pinned_user_tokens=100)
    fold = await _compact(compactor, log, _FakeGenerate(_text_response("S")))
    assert fold is not None
    assert fold.pinned == [4]


# --- The summary request ---


@pytest.mark.asyncio
async def test_request_carries_the_instructions_in_a_compaction_tag() -> None:
    generate = _FakeGenerate(_text_response("S"))
    compactor = SummarizingCompactor(budget=_budget(1), instructions="Summarize it.")
    await _compact(compactor, _rounds(3), generate)
    request = generate.calls[0].request
    assert request.role == "user"
    assert request.text == (
        f"<{COMPACTION_REQUEST_TAG}>\nSummarize it.\n</{COMPACTION_REQUEST_TAG}>"
    )
    assert generate.calls[0].transform is None  # the cache-sharing request


@pytest.mark.asyncio
async def test_a_tool_call_answer_is_retried_as_text() -> None:
    generate = _FakeGenerate(
        _tool_call_response("big", "{}", "x1"), _text_response("SUMMARY")
    )
    compactor = SummarizingCompactor(budget=_budget(1))
    fold = await _compact(compactor, _rounds(3), generate)
    assert fold is not None
    assert fold.summary == "SUMMARY"
    assert [c.force_text for c in generate.calls] == [False, True]


@pytest.mark.asyncio
async def test_an_echoed_summary_wrapper_is_stripped() -> None:
    generate = _FakeGenerate(_text_response(f"<{SUMMARY_TAG}>\nS\n</{SUMMARY_TAG}>"))
    fold = await _compact(SummarizingCompactor(budget=_budget(1)), _rounds(3), generate)
    assert fold is not None
    assert fold.summary == "S"


@pytest.mark.asyncio
async def test_an_empty_summary_is_no_fold() -> None:
    generate = _FakeGenerate(_text_response("  "))
    assert (
        await _compact(SummarizingCompactor(budget=_budget(1)), _rounds(3), generate)
        is None
    )


@pytest.mark.asyncio
async def test_an_oversized_request_is_shrunk_and_retried() -> None:
    body: list[InputItem] = [
        _user("pinned"),
        _user(fold_summary_text("OLD")),
        *_rounds(4, output="Z" * 5000),
    ]
    seen: list[list[InputItem]] = []

    class _Shrinking(_FakeGenerate):
        async def __call__(
            self,
            request: InputMessageItem,
            *,
            transform: ViewTransform | None = None,
            force_text: bool = False,
        ) -> Response:
            seen.append(transform(list(body)) if transform else list(body))
            return await super().__call__(
                request, transform=transform, force_text=force_text
            )

    generate = _Shrinking(_overflow(), _overflow(), _text_response("S"))
    fold = await _compact(SummarizingCompactor(budget=_budget(1)), _rounds(4), generate)
    assert fold is not None

    collapsed, dropped = seen[1], seen[2]
    outputs = [m for m in collapsed if isinstance(m, FunctionToolOutputItem)]
    assert all(len(o.text) < 5000 for o in outputs[:-1])  # older outputs collapsed
    assert len(outputs[-1].text) == 5000  # the latest round in full
    # The last step also leaves out the oldest half of the rounds not yet summarized,
    # with a marker, and keeps the summary block in front.
    assert is_fold_summary(dropped[1])
    assert "left out of this request" in dropped[2].text  # type: ignore[union-attr]
    assert count_rounds(dropped[3:]) == 2


@pytest.mark.asyncio
async def test_an_overflow_after_every_shrink_propagates() -> None:
    generate = _FakeGenerate(_overflow(), _overflow(), _overflow())
    with pytest.raises(LlmContextWindowError):
        await _compact(SummarizingCompactor(budget=_budget(1)), _rounds(3), generate)


# --- ContextWindowManager: failures ---


class _RaisingCompactor:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, *_: Any, **__: Any) -> FoldSpec | None:
        self.calls += 1
        raise LlmBadRequestError(
            "boom", response=httpx.Response(400, request=_REQ), body=None
        )


@pytest.mark.asyncio
async def test_failures_are_logged_and_stop_turn_boundary_compaction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cw = ContextWindowManager(model_name="mock")
    cw.add_messages(_rounds(3))
    compactor = _RaisingCompactor()
    cw.set_compactor(compactor)
    generate = _FakeGenerate()

    with caplog.at_level(logging.WARNING, logger="grasp_agents.agent.context_window"):
        for _ in range(MAX_COMPACTION_FAILURES + 2):
            assert await cw.maybe_compact(exec_id="x", generate=generate) is None
    assert compactor.calls == MAX_COMPACTION_FAILURES
    assert any(
        "turn-boundary compaction stops" in r.getMessage() for r in caplog.records
    )

    # The context-window-error path still tries.
    assert await cw.maybe_compact(exec_id="x", generate=generate, force=True) is None
    assert compactor.calls == MAX_COMPACTION_FAILURES + 1


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
    compactor: SummarizingCompactor | None = None,
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
        compactor or SummarizingCompactor(budget=_budget(5), keep_recent_rounds=1)
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
    # Same prefix, tools and tool choice as the request before it: the
    # provider's prompt cache serves it.
    assert summary_call.input[: len(before.input)] == before.input
    assert summary_call.tools == before.tools
    assert summary_call.tool_choice == before.tool_choice
    assert summary_call.output_schema is None

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
async def test_summary_call_is_accounted_like_any_other() -> None:
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
    seen: list[str] = []

    @agent.add_after_llm_hook
    async def _record(response: Response, *, exec_id: str, turn: int) -> None:
        del exec_id, turn
        seen.append(response.output_text)

    await agent.run("go")
    assert seen == ["", "", "SUMMARY", "done"]
    assert ctx.usage_tracker.total_usage.input_tokens == 4 * 10


@pytest.mark.asyncio
async def test_a_tool_call_in_the_summary_answer_is_never_executed() -> None:
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
    assert llm.calls[3].tool_choice == "none"  # type: ignore[attr-defined]
    assert agent.agent_ctx.cw.folds[-1].summary == "SUMMARY"


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
async def test_add_compaction_registers_the_section_and_injects_the_budget() -> None:
    agent = LLMAgent[str, str, None](
        name="a", ctx=SessionContext(), llm=MockLLM(responses_queue=[])
    )
    compactor = agent.add_compaction()
    assert agent.agent_ctx.cw.compactor is compactor
    assert compactor.budget is not None
    names = [s.name for s in agent.system_prompt_sections]
    assert COMPACTION_SECTION_NAME in names


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
        compactor=SummarizingCompactor(
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
async def test_no_start_event_without_a_summary_request() -> None:
    llm = MockLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm, compactor=SummarizingCompactor(budget=_budget(10_000_000)))
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
async def test_leaving_the_stream_at_the_start_event_cancels_the_summary() -> None:
    cancelled = asyncio.Event()

    @dataclass(frozen=True)
    class _HangingSummaryLLM(MockLLM):
        async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
            if _is_summary_request(input):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return await super()._generate_response_once(input, **kwargs)

    llm = _HangingSummaryLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
        ]
    )
    agent, _ = _agent(llm)
    stream = agent.run_stream("go")
    async with asyncio.timeout(5):
        async for event in stream:
            if isinstance(event, CompactionStartEvent):
                break
        await stream.aclose()  # type: ignore[attr-defined]
        await cancelled.wait()


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
    agent, _ = _agent(llm, compactor=SummarizingCompactor(budget=_budget(10_000_000)))
    events = [event async for event in agent.run_stream("go")]

    assert events[-1].data.payloads[0] == "recovered"  # type: ignore[union-attr]
    compactions = [e for e in events if isinstance(e, CompactionDoneEvent)]
    assert [c.data.forced for c in compactions] == [True]
    starts = [e for e in events if isinstance(e, CompactionStartEvent)]
    assert [s.data.forced for s in starts] == [True]
    assert agent.agent_ctx.cw.folds[-1].summary == "SUM"


# --- Restoration after a fold ---


@pytest.mark.asyncio
async def test_restore_flagged_attachments_rerun_after_a_fold() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("SUMMARY"),
            _text_response("done"),
        ]
    )
    agent, _ = _agent(llm)
    seen: list[list[str]] = []

    def compute(*, messages: Sequence[InputItem] | None = None, **_: Any) -> str:
        seen.append([m.text for m in messages or [] if isinstance(m, InputMessageItem)])
        return "the document"

    agent.add_input_attachment(
        InputAttachment(name="doc", compute=compute, restore_after_compaction=True)
    )
    await agent.run("go")

    # Once for the input, once after the fold — against the folded view.
    assert len(seen) == 2
    assert fold_summary_text("SUMMARY") in seen[1]
    # Delivered as input right after the fold, before the final answer.
    restored = agent.transcript[-2]
    assert isinstance(restored, InputMessageItem)
    assert restored.text == wrap_in_system_reminder("the document", subject="doc")
    assert restored in llm.calls[-1].input  # type: ignore[attr-defined]


# --- Persistence and rollback ---


def test_folds_and_anchor_round_trip_through_checkpoint() -> None:
    checkpoint = AgentCheckpoint(
        session_key="s",
        processor_name="a",
        messages=[],
        folds=[FoldSpec(start=0, end=3, summary="S", pinned=[1])],
        token_anchor=TokenAnchor(input_tokens=1234, message_count=3),
    )
    reloaded = AgentCheckpoint.model_validate_json(checkpoint.model_dump_json())
    assert reloaded.folds == checkpoint.folds
    assert reloaded.token_anchor == checkpoint.token_anchor


@pytest.mark.asyncio
async def test_folds_and_anchor_restored_on_resume_without_resummarizing() -> None:
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
    assert folds

    ctx2: SessionContext[None] = SessionContext(
        checkpoint_store=store, session_key="s1"
    )
    # One reply: a re-summarization would exhaust the queue and fail the run.
    agent2, _ = _agent(
        MockLLM(responses_queue=[_text_response("after")]),
        ctx=ctx2,
        compactor=SummarizingCompactor(budget=_budget(10_000_000)),
    )
    out = await agent2.run("again")
    assert out.payloads[0] == "after"
    assert agent2.agent_ctx.cw.folds == folds


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
