"""
Reactive summarization: ``SummarizingCompactor`` + the ``Compaction`` bundle
(``context.compaction``).

Covers reactive gating + turn-boundary span selection (by ``keep_recent_turns``),
that a summary fold reaches the model-facing view while the durable log keeps
originals, fold persistence across resume, rollback dropping folds past the
rewind point, and the context-window-error recovery path.
"""

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context.compaction import (
    CollapseToolOutputsProjector,
    Compaction,
    ContextBudget,
    LLMSummarizer,
    SummarizingCompactor,
)
from grasp_agents.context.projection import apply_folds
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.durability.checkpoints import AgentCheckpoint
from grasp_agents.run_context import RunContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.events import CompactionEvent
from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmContextWindowError
from tests._helpers import MockLLM, _text_response, _tool_call_response

BIGMSG = "x" * 100


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _mock_llm(text: str = "SUMMARY") -> MockLLM:
    return MockLLM(responses_queue=[_text_response(text)])


def _summarizer(text: str = "SUMMARY") -> LLMSummarizer:
    return LLMSummarizer(_mock_llm(text))


def _budget(max_input: int) -> ContextBudget:
    return ContextBudget("mock", max_input_tokens=max_input, buffer_tokens=0)


# --- SummarizingCompactor (reactive) ---


@pytest.mark.asyncio
async def test_under_budget_is_a_noop() -> None:
    comp = SummarizingCompactor(summarizer=_summarizer(), budget=_budget(10_000_000))
    msgs = [_user(BIGMSG) for _ in range(10)]
    assert await comp(msgs, input_tokens=10, folds=[], exec_id="x") is None


@pytest.mark.asyncio
async def test_over_budget_folds_oldest_keeping_recent() -> None:
    comp = SummarizingCompactor(
        summarizer=_summarizer("BRIEF"), budget=_budget(100), keep_recent_turns=2
    )
    msgs = [_user(BIGMSG) for _ in range(10)]
    fold = await comp(msgs, input_tokens=200, folds=[], exec_id="x")
    assert fold is not None
    assert fold.start == 0
    assert 0 < fold.end < len(msgs)  # recent messages kept verbatim
    assert fold.summary == "BRIEF"


@pytest.mark.asyncio
async def test_force_folds_regardless_of_budget() -> None:
    comp = SummarizingCompactor(
        summarizer=_summarizer(), budget=_budget(100), keep_recent_turns=2
    )
    msgs = [_user(BIGMSG) for _ in range(10)]
    fold = await comp(msgs, input_tokens=1, folds=[], exec_id="x", force=True)
    assert fold is not None


@pytest.mark.asyncio
async def test_returns_none_when_everything_is_folded() -> None:
    comp = SummarizingCompactor(summarizer=_summarizer(), budget=_budget(100))
    msgs = [_user(BIGMSG) for _ in range(4)]
    fold = await comp(
        msgs,
        input_tokens=200,
        folds=[FoldSpec(start=0, end=4, summary="prev")],
        exec_id="x",
        force=True,
    )
    assert fold is None


@pytest.mark.asyncio
async def test_fold_end_snaps_off_a_tool_pair() -> None:
    # keep_recent_turns=2 keeps the last two turns ([call+result], [user]); the
    # fold ends at the turn boundary before them, never between a call and result.
    comp = SummarizingCompactor(
        summarizer=_summarizer(), budget=_budget(100), keep_recent_turns=2
    )
    msgs = [
        _user(BIGMSG),
        FunctionToolCallItem(call_id="c1", name="t", arguments="{}"),
        FunctionToolOutputItem.from_tool_result(call_id="c1", output=BIGMSG),
        _user(BIGMSG),
    ]
    fold = await comp(msgs, input_tokens=200, folds=[], exec_id="x")
    assert fold is not None
    assert fold.end == 1  # snapped back from 2 (between c1's call and result)


@pytest.mark.asyncio
async def test_fold_keeps_reasoning_with_its_call() -> None:
    # A reasoning item preceding a tool call must not be folded away while the
    # call is kept — providers reject a function call sent without its reasoning.
    comp = SummarizingCompactor(
        summarizer=_summarizer(), budget=_budget(100), keep_recent_turns=2
    )
    msgs = [
        _user(BIGMSG),
        ReasoningItem(encrypted_content="enc"),
        FunctionToolCallItem(call_id="c1", name="t", arguments="{}"),
        FunctionToolOutputItem.from_tool_result(call_id="c1", output=BIGMSG),
        _user(BIGMSG),
    ]
    fold = await comp(msgs, input_tokens=200, folds=[], exec_id="x")
    assert fold is not None
    assert fold.end == 1  # pulled back to before the reasoning+call+result turn
    # In the projected view the call is still immediately preceded by reasoning.
    view = apply_folds(msgs, [fold])
    call_idx = next(
        i for i, m in enumerate(view) if isinstance(m, FunctionToolCallItem)
    )
    assert isinstance(view[call_idx - 1], ReasoningItem)


@pytest.mark.asyncio
async def test_fold_keeps_reasoning_with_call_after_text() -> None:
    # reasoning → assistant text → call: the call still depends on the reasoning,
    # so a fold cannot drop the reasoning even though text sits between them.
    comp = SummarizingCompactor(
        summarizer=_summarizer(), budget=_budget(100), keep_recent_turns=2
    )
    msgs = [
        _user(BIGMSG),
        ReasoningItem(encrypted_content="enc"),
        OutputMessageItem(
            content_parts=[OutputMessageText(text="let me check")], status="completed"
        ),
        FunctionToolCallItem(call_id="c1", name="t", arguments="{}"),
        FunctionToolOutputItem.from_tool_result(call_id="c1", output=BIGMSG),
        _user(BIGMSG),
    ]
    fold = await comp(msgs, input_tokens=200, folds=[], exec_id="x")
    assert fold is not None
    view = apply_folds(msgs, [fold])
    call_idx = next(
        i for i, m in enumerate(view) if isinstance(m, FunctionToolCallItem)
    )
    assert any(isinstance(m, ReasoningItem) for m in view[:call_idx])


@pytest.mark.asyncio
async def test_fold_dense_reasoning_chain_keeps_every_pair() -> None:
    # A real tool chain: reasoning before every call, only tool results between
    # turns (no standalone text messages). Folds must land on turn boundaries so
    # every kept call still has its reasoning item.
    comp = SummarizingCompactor(
        summarizer=_summarizer(), budget=_budget(100), keep_recent_turns=2
    )
    msgs: list[Any] = [_user(BIGMSG)]
    for n in range(4):
        msgs += [
            ReasoningItem(encrypted_content=f"enc{n}"),
            FunctionToolCallItem(call_id=f"c{n}", name="t", arguments="{}"),
            FunctionToolOutputItem.from_tool_result(call_id=f"c{n}", output=BIGMSG),
        ]
    fold = await comp(msgs, input_tokens=500, folds=[], exec_id="x")
    assert fold is not None
    view = apply_folds(msgs, [fold])
    reasoning_seen = False
    for item in view:
        if isinstance(item, ReasoningItem):
            reasoning_seen = True
        elif isinstance(item, FunctionToolCallItem):
            assert reasoning_seen, "a kept call lost its preceding reasoning item"
        elif isinstance(item, FunctionToolOutputItem):
            reasoning_seen = False


@pytest.mark.asyncio
async def test_oversized_span_folds_incrementally() -> None:
    # The fold span is bounded to max_summary_input_tokens so the summary call
    # can't overflow the summarizer; an over-large backlog folds one chunk now and
    # the rest on later passes rather than handing the summarizer the whole thing.
    async def _const(_messages: Any) -> str:
        return "SUM"

    comp = SummarizingCompactor(
        summarizer=_const,
        budget=_budget(100),
        keep_recent_turns=1,
        max_summary_input_tokens=50,  # ~200 chars
    )
    msgs = [_user(BIGMSG) for _ in range(10)]  # each ~100 chars (~25 tok)
    fold = await comp(msgs, input_tokens=300, folds=[], exec_id="x")
    assert fold is not None
    assert fold.start == 0
    assert fold.end < 9  # capped well short of the keep-1 boundary

    # The remaining old turns fold on the next pass, resuming where this left off.
    fold2 = await comp(msgs, input_tokens=300, folds=[fold], exec_id="x")
    assert fold2 is not None
    assert fold2.start == fold.end
    assert fold2.end > fold.end


@pytest.mark.asyncio
async def test_compaction_gate_measures_post_projection_view() -> None:
    # Escalation: with a collapse projector registered, the fold gate must measure
    # the POST-projection (collapsed) view, not the raw log — else it would fold
    # (an LLM call) when free collapse alone fits. Exercises the recount fallback.
    agent = LLMAgent[str, str, None](
        name="a", ctx=RunContext(), llm=MockLLM(responses_queue=[])
    )
    cw = agent._cw
    cw.initial_context = []
    agent.transcript.messages = [
        _user("q"),
        FunctionToolCallItem(call_id="c1", name="t", arguments="{}"),
        FunctionToolOutputItem.from_tool_result(call_id="c1", output="Z" * 40_000),
        _user("next"),
    ]
    cw.add_view_projector(
        CollapseToolOutputsProjector(
            proactive=True, keep_recent_turns=1, head_chars=100, tail_chars=50
        )
    )
    cw.reset_anchor()  # force the projector-blind recount fallback

    raw = cw.effective_input_tokens()
    projected = await cw.projected_input_tokens(exec_id="x")
    assert projected < raw  # the big old output is collapsed before the gate reads it


# --- Compaction bundle ---


def test_compaction_for_model_builds_both_strategies() -> None:
    compaction = Compaction.for_model("mock", llm=_mock_llm())
    assert isinstance(compaction.collapse, CollapseToolOutputsProjector)
    assert compaction.summarize is not None


def test_compaction_without_llm_has_no_summarizer() -> None:
    compaction = Compaction.for_model("mock")
    assert compaction.collapse is not None
    assert compaction.summarize is None


# --- Budget auto-init from the agent's model (no manual ContextBudget) ---


def test_add_compaction_no_arg_auto_configures_from_agent() -> None:
    agent = LLMAgent[str, str, None](
        name="a", ctx=RunContext(), llm=MockLLM(responses_queue=[])
    )
    compaction = agent.add_compaction()  # no budget, no model name passed
    assert compaction.collapse in agent._cw.view_projectors
    assert agent._cw.compactor is compaction.summarize
    # budgets were injected from the agent's model, not constructed by the caller
    assert compaction.collapse.budget is not None
    assert compaction.summarize is not None
    assert compaction.summarize.budget is not None


def test_manager_injects_budget_into_budgetless_projector() -> None:
    agent = LLMAgent[str, str, None](
        name="a", ctx=RunContext(), llm=MockLLM(responses_queue=[])
    )
    proj = CollapseToolOutputsProjector(keep_recent_turns=1)
    assert proj.budget is None
    agent.add_view_projector(proj)
    assert proj.budget is not None  # injected on registration
    assert proj.budget.model == "mock"


def test_summarizer_cap_tracks_summarizer_model_not_agent() -> None:
    # The span fed to the summarizer is bounded by the SUMMARIZER's own window,
    # inferred from its model — not the agent's budget.
    comp = SummarizingCompactor(summarizer=LLMSummarizer(_mock_llm()))
    assert comp._summary_budget is not None
    assert comp._summary_budget.model == "mock"
    # An injected (different) agent budget does not override the summarizer's cap.
    comp.budget = ContextBudget("agent-model", max_input_tokens=999, buffer_tokens=0)
    assert comp._summary_cap() == comp._summary_budget.soft_limit
    assert comp._summary_cap() != 999


# --- Integration fixtures ---


@dataclass(frozen=True)
class _RecordingLLM(MockLLM):
    """MockLLM that records each call's projected view in ``views``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "views", [])

    async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
        self.views.append(list(input))  # type: ignore[attr-defined]
        return await super()._generate_response_once(input, **kwargs)


@dataclass(frozen=True)
class _OverflowOnceLLM(MockLLM):
    """Raises a context-window error on its second call, then behaves normally."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "_calls", 0)

    async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
        object.__setattr__(self, "_calls", self._calls + 1)  # type: ignore[attr-defined]
        if self._calls == 2:  # type: ignore[attr-defined]
            raise LlmContextWindowError(
                "context length exceeded",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://x")
                ),
                body=None,
            )
        return await super()._generate_response_once(input, **kwargs)


class _BigInput(BaseModel):
    pass


_BIG = "Z" * 5000


class _BigTool(BaseTool[_BigInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="big", description="Returns a large output.")

    async def _run(self, inp: _BigInput, **kwargs: Any) -> str:
        return _BIG


# --- Integration: fold reaches the view; log keeps originals ---


@pytest.mark.asyncio
async def test_summary_fold_reaches_view_log_keeps_originals() -> None:
    summary_llm = MockLLM(responses_queue=[_text_response("SUMMARY-OF-EARLIER")])
    agent_llm = _RecordingLLM(
        responses_queue=[_tool_call_response("big", "{}", "c1"), _text_response("done")]
    )
    ctx: RunContext[None] = RunContext()
    agent = LLMAgent[str, str, None](
        name="a", ctx=ctx, llm=agent_llm, tools=[_BigTool()]
    )
    # Tiny budget + keep the last message only ⇒ compaction fires at turn 1.
    agent.add_compactor(
        SummarizingCompactor(
            summarizer=LLMSummarizer(summary_llm),
            budget=_budget(5),
            keep_recent_turns=1,
        )
    )

    out = await agent.run("go")
    assert out.payloads[0] == "done"

    assert agent._cw.folds  # a fold was recorded
    final_view = agent_llm.views[-1]  # type: ignore[attr-defined]
    assert any(
        isinstance(m, InputMessageItem) and "SUMMARY-OF-EARLIER" in m.text
        for m in final_view
    )
    # The durable log kept the full original tool output.
    assert any(
        isinstance(m, FunctionToolOutputItem) and len(m.text) == len(_BIG)
        for m in agent.transcript.messages
    )


@pytest.mark.asyncio
async def test_run_stream_emits_compaction_event() -> None:
    agent_llm = MockLLM(
        responses_queue=[_tool_call_response("big", "{}", "c1"), _text_response("done")]
    )
    ctx: RunContext[None] = RunContext()
    agent = LLMAgent[str, str, None](
        name="a", ctx=ctx, llm=agent_llm, tools=[_BigTool()]
    )
    agent.add_compactor(
        SummarizingCompactor(
            summarizer=_summarizer("SUMMARY-OF-EARLIER"),
            budget=_budget(5),
            keep_recent_turns=1,
        )
    )
    events = [event async for event in agent.run_stream("go")]
    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert compactions, "a CompactionEvent should be emitted when a fold is recorded"
    info = compactions[0].data
    assert info.folded_turns > 0
    assert info.summary == "SUMMARY-OF-EARLIER"  # the fold's summary rides along
    assert compactions[0].source == "a"


# --- LLMSummarizer renders a text transcript (not live messages) ---


@pytest.mark.asyncio
async def test_summarizer_sends_text_transcript_not_live_messages() -> None:
    rec = _RecordingLLM(responses_queue=[_text_response("SUMMARY")])
    summary = await LLMSummarizer(rec)(
        [
            _user("what is X?"),
            ReasoningItem(
                summary_parts=[ReasoningSummary(text="private thoughts")],
                encrypted_content="enc",
            ),
            FunctionToolCallItem(call_id="c1", name="lookup", arguments='{"q": "X"}'),
            FunctionToolOutputItem.from_tool_result(
                call_id="c1", output="X is a thing"
            ),
            OutputMessageItem(
                content_parts=[OutputMessageText(text="X is a thing.")],
                status="completed",
            ),
        ]
    )
    assert summary == "SUMMARY"
    sent = rec.views[-1]  # type: ignore[attr-defined]
    # A system instruction + ONE user-role transcript message — not the raw,
    # replayable conversation items (reasoning/tool-call objects).
    assert len(sent) == 2
    assert all(isinstance(m, InputMessageItem) for m in sent)
    # The instruction frames the task unambiguously as summarizing a transcript.
    assert "transcript" in sent[0].text.lower()
    transcript = sent[1].text
    assert "<transcript>" in transcript  # delimited so it can't read as a request
    assert "</transcript>" in transcript
    assert "what is X?" in transcript
    assert "lookup" in transcript
    assert "X is a thing" in transcript
    assert "private thoughts" not in transcript  # reasoning omitted from context


# --- Persistence: folds round-trip + restore on resume ---


def test_folds_round_trip_through_checkpoint() -> None:
    checkpoint = AgentCheckpoint(
        session_key="s",
        processor_name="a",
        messages=[],
        folds=[FoldSpec(start=0, end=3, summary="S")],
    )
    reloaded = AgentCheckpoint.model_validate_json(checkpoint.model_dump_json())
    assert reloaded.folds == [FoldSpec(start=0, end=3, summary="S")]


@pytest.mark.asyncio
async def test_folds_restored_on_resume_without_resummarizing() -> None:
    store = InMemoryCheckpointStore()
    agent_llm = MockLLM(
        responses_queue=[_tool_call_response("big", "{}", "c1"), _text_response("done")]
    )
    ctx: RunContext[None] = RunContext(checkpoint_store=store, session_key="s1")
    agent = LLMAgent[str, str, None](
        name="a", ctx=ctx, llm=agent_llm, tools=[_BigTool()]
    )
    agent.add_compactor(
        SummarizingCompactor(
            summarizer=LLMSummarizer(_mock_llm("SUM")),
            budget=_budget(5),
            keep_recent_turns=1,
        )
    )
    await agent.run("go")
    saved = [f.summary for f in agent._cw.folds]
    assert saved  # folded during the run

    # Fresh instance, same store → resume restores folds. The summarizer's LLM
    # has an empty queue, so a re-summarization would raise — proving it doesn't.
    ctx2: RunContext[None] = RunContext(checkpoint_store=store, session_key="s1")
    agent2 = LLMAgent[str, str, None](
        name="a", ctx=ctx2, llm=MockLLM(responses_queue=[_text_response("after")])
    )
    agent2.add_compactor(
        SummarizingCompactor(
            summarizer=LLMSummarizer(MockLLM(responses_queue=[])),
            budget=_budget(10_000_000),
            keep_recent_turns=1,
        )
    )
    await agent2.run("again")
    assert [f.summary for f in agent2._cw.folds][: len(saved)] == saved


# --- Rollback drops folds past the rewind point ---


@pytest.mark.asyncio
async def test_rollback_drops_folds_past_rewind() -> None:
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

    agent._cw.folds = [
        FoldSpec(start=0, end=1, summary="early"),  # within step 0
        FoldSpec(start=n_after_1, end=n_after_1 + 1, summary="late"),  # in step 2
    ]
    await agent.rollback_to_step(1)

    summaries = [f.summary for f in agent._cw.folds]
    assert "late" not in summaries  # past the rewind → dropped
    assert "early" in summaries  # within the kept prefix → survives


# --- Reactive net: context-window error → force-compact + retry ---


@pytest.mark.asyncio
async def test_context_window_error_compacts_and_retries() -> None:
    summary_llm = MockLLM(responses_queue=[_text_response("SUM")])
    agent_llm = _OverflowOnceLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _text_response("recovered"),
        ]
    )
    ctx: RunContext[None] = RunContext()
    agent = LLMAgent[str, str, None](
        name="a", ctx=ctx, llm=agent_llm, tools=[_BigTool()]
    )
    # Huge budget ⇒ proactive compaction never fires; only the reactive (force)
    # path on the overflow does.
    agent.add_compactor(
        SummarizingCompactor(
            summarizer=LLMSummarizer(summary_llm),
            budget=_budget(10_000_000),
            keep_recent_turns=1,
        )
    )

    out = await agent.run("go")
    assert out.payloads[0] == "recovered"  # recovered via compact + retry
    assert agent._cw.folds  # the forced fold was recorded
