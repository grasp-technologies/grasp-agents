"""
Recency-gated collapse of large tool outputs (``context.compaction``).

Collapse keeps the most recent ``keep_recent_turns`` turns' tool outputs verbatim
and snippets older ones to head+tail — only ever hiding outputs the model has moved
past (which it could not recover, there being no spill-to-file). The
:class:`CollapseToolOutputsProjector` decides *when*: budget-gated (only over
budget) by default, or ``proactive`` (every turn). The integration test confirms
the collapsed view reaches the provider while the durable transcript keeps the
full output.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.context.compaction import (
    DEFAULT_MAX_INPUT_TOKENS,
    CollapseToolOutputsProjector,
    ContextBudget,
    collapse_tool_outputs,
)
from grasp_agents.llm import count_input_tokens
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from tests._helpers import MockLLM, _text_response, _tool_call_response

NOTICE = "elided by context management"
BIG = "A" * 5000
OVER = 1_000_000  # an input_tokens far above any test soft_limit


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _call(call_id: str) -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=call_id, name="tool", arguments="{}")


def _result(call_id: str, output: str) -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id=call_id, output=output)


def _outputs(messages: Sequence[Any]) -> list[FunctionToolOutputItem]:
    return [m for m in messages if isinstance(m, FunctionToolOutputItem)]


def _budget(max_input: int) -> ContextBudget:
    # buffer 0 ⇒ soft_limit == max_input, so a test controls the limit directly.
    return ContextBudget("mock", max_input_tokens=max_input, buffer_tokens=0)


def _no_metadata(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(max_input_tokens=None, max_output_tokens=None)


# --- ContextBudget ---


def test_soft_limit_is_window_minus_buffer() -> None:
    assert ContextBudget("m", 1000, buffer_tokens=150).soft_limit == 850


def test_unknown_window_disables_the_budget() -> None:
    budget = ContextBudget("m", max_input_tokens=None)
    assert budget.soft_limit is None
    assert budget.overflow(OVER) == 0
    assert not budget.is_exceeded(OVER)


def test_budget_detects_overflow() -> None:
    assert _budget(100).overflow(50) == 0
    assert _budget(100).overflow(180) == 80
    assert _budget(100).is_exceeded(180)


# --- Fallbacks when litellm lacks metadata ---


def test_for_model_falls_back_to_default_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "grasp_agents.context.compaction.get_model_capabilities", _no_metadata
    )
    budget = ContextBudget.for_model("unknown-model")
    assert budget.max_input_tokens == DEFAULT_MAX_INPUT_TOKENS
    assert budget.soft_limit is not None
    assert budget.soft_limit > 0


def test_for_model_fallback_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "grasp_agents.context.compaction.get_model_capabilities", _no_metadata
    )
    budget = ContextBudget.for_model("unknown-model", default_max_input_tokens=None)
    assert budget.max_input_tokens is None
    assert budget.soft_limit is None


# --- collapse_tool_outputs (recency-gated) ---


def test_collapses_older_outputs_keeps_recent() -> None:
    msgs = [
        _call("c1"),
        _result("c1", "A" * 5000),
        _call("c2"),
        _result("c2", "B" * 5000),
    ]
    # keep_recent_turns=1 keeps the c2 turn verbatim; the older c1 output collapses.
    out = collapse_tool_outputs(
        msgs, keep_recent_turns=1, head_chars=100, tail_chars=50
    )
    c1 = next(m for m in _outputs(out) if m.call_id == "c1")
    c2 = next(m for m in _outputs(out) if m.call_id == "c2")
    assert NOTICE in c1.text  # older → collapsed
    assert c1.text.startswith("A" * 100)
    assert c2.text == "B" * 5000  # recent → verbatim


def test_recent_window_is_kept_verbatim() -> None:
    # Fewer turns than keep_recent_turns ⇒ nothing is old ⇒ no collapse.
    msgs = [_call("c1"), _result("c1", BIG)]
    out = collapse_tool_outputs(msgs, keep_recent_turns=8)
    assert out == msgs
    assert out[1] is msgs[1]  # same object, untouched


def test_keep_recent_tokens_shrinks_the_window() -> None:
    # Three big tool turns; keep_recent_turns=3 would keep all of them, but a token
    # cap sized to the last two turns shrinks the kept window so the oldest folds.
    msgs = [
        _call("c1"),
        _result("c1", "A" * 5000),
        _call("c2"),
        _result("c2", "B" * 5000),
        _call("c3"),
        _result("c3", "C" * 5000),
    ]
    cap = count_input_tokens("mock", msgs[2:])  # exactly the last two turns
    out = collapse_tool_outputs(
        msgs,
        keep_recent_turns=3,
        keep_recent_tokens=cap,
        head_chars=100,
        tail_chars=50,
        model="mock",
    )
    by_id = {m.call_id: m for m in _outputs(out)}
    assert NOTICE in by_id["c1"].text  # oldest folds — window shrank from 3 to 2
    assert by_id["c2"].text == "B" * 5000  # kept
    assert by_id["c3"].text == "C" * 5000  # kept


def test_short_outputs_are_kept() -> None:
    # An OLD short output isn't worth collapsing (no gain from a snippet).
    msgs = [_call("c1"), _result("c1", "ok"), _call("c2"), _result("c2", BIG)]
    out = collapse_tool_outputs(msgs, keep_recent_turns=1)
    assert out[1] is msgs[1]


def test_output_with_image_is_kept() -> None:
    item = FunctionToolOutputItem(
        call_id="c1",
        output_parts=[
            InputText(text=BIG),
            InputImage(image_url="https://example.com/x.png"),
        ],
    )
    # Old (a later turn follows) but carries a non-text part → left intact.
    msgs = [_call("c1"), item, _call("c2"), _result("c2", BIG)]
    out = collapse_tool_outputs(msgs, keep_recent_turns=1)
    assert out[1] is item


def test_non_tool_items_pass_through() -> None:
    msgs = [_user("hi"), _call("c1"), _result("c1", BIG), _user("next")]
    out = collapse_tool_outputs(msgs, keep_recent_turns=1)
    assert out[0] is msgs[0]  # user message untouched
    assert NOTICE in _outputs(out)[0].text  # c1 output is older → collapsed


def test_pairing_is_preserved() -> None:
    msgs = [_user("q"), _call("c1"), _result("c1", BIG), _user("next")]
    out = collapse_tool_outputs(msgs, keep_recent_turns=1)
    assert [m.call_id for m in _outputs(out)] == ["c1"]
    transcript = LLMAgentTranscript()
    transcript.messages = list(out)
    transcript.validate_tool_call_pairing()  # raises if collapse orphaned a pair


def test_collapse_is_deterministic() -> None:
    msgs = [_call("c1"), _result("c1", BIG), _call("c2"), _result("c2", BIG)]
    once = collapse_tool_outputs(msgs, keep_recent_turns=1)
    twice = collapse_tool_outputs(msgs, keep_recent_turns=1)
    assert _outputs(once)[0].text == _outputs(twice)[0].text


# --- CollapseToolOutputsProjector (ViewProjector) ---


@pytest.mark.asyncio
async def test_projector_is_budget_gated() -> None:
    msgs = [_call("c1"), _result("c1", BIG), _call("c2"), _result("c2", BIG)]
    under = await CollapseToolOutputsProjector(
        budget=_budget(10_000_000), keep_recent_turns=1
    )(msgs, exec_id="x", input_tokens=10)
    assert under == msgs  # under budget ⇒ nothing collapsed
    over = await CollapseToolOutputsProjector(
        budget=_budget(1), keep_recent_turns=1, head_chars=100, tail_chars=50
    )(msgs, exec_id="x", input_tokens=OVER)
    c1 = next(m for m in _outputs(over) if m.call_id == "c1")
    assert NOTICE in c1.text  # old output collapsed once over budget


@pytest.mark.asyncio
async def test_projector_proactive_ignores_budget() -> None:
    msgs = [_call("c1"), _result("c1", BIG), _call("c2"), _result("c2", BIG)]
    # input_tokens far under any budget, but proactive collapses old outputs anyway.
    out = await CollapseToolOutputsProjector(
        proactive=True, keep_recent_turns=1, head_chars=100, tail_chars=50
    )(msgs, exec_id="x", input_tokens=1)
    c1 = next(m for m in _outputs(out) if m.call_id == "c1")
    assert NOTICE in c1.text


@pytest.mark.asyncio
async def test_projector_without_budget_is_inert() -> None:
    # No budget + not proactive: the agent injects its model-derived budget on
    # registration; until then the projector is inert rather than erroring.
    msgs = [_call("c1"), _result("c1", BIG), _call("c2"), _result("c2", BIG)]
    out = await CollapseToolOutputsProjector(keep_recent_turns=1)(
        msgs, exec_id="x", input_tokens=OVER
    )
    assert out == msgs


# --- End-to-end: collapsed view reaches the provider, log stays full ---


@dataclass(frozen=True)
class _RecordingLLM(MockLLM):
    """MockLLM that records the projected view passed on each call."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "views", [])

    async def _generate_response_once(self, input: Any, **kwargs: Any) -> Any:
        self.views.append(list(input))  # type: ignore[attr-defined]
        return await super()._generate_response_once(input, **kwargs)


class _BigInput(BaseModel):
    pass


_BIG_PAYLOAD = "Z" * 5000


class _BigTool(BaseTool[_BigInput, str, Any]):
    def __init__(self) -> None:
        super().__init__(name="big", description="Returns a large output.")

    async def _run(self, inp: _BigInput, **kwargs: Any) -> str:
        return _BIG_PAYLOAD


@pytest.mark.asyncio
async def test_collapsed_view_reaches_llm_while_log_keeps_full_output() -> None:
    llm = _RecordingLLM(
        responses_queue=[
            _tool_call_response("big", "{}", "c1"),
            _tool_call_response("big", "{}", "c2"),
            _text_response("done"),
        ]
    )
    ctx: SessionContext[None] = SessionContext()
    agent = LLMAgent[str, str, None](
        name="collapse_agent", ctx=ctx, llm=llm, tools=[_BigTool()]
    )
    # keep_recent_turns=1 ⇒ the c1 output is old by the final call and collapses;
    # soft_limit 1 < the mock's reported input_tokens ⇒ the budget gate fires.
    agent.add_view_projector(
        CollapseToolOutputsProjector(
            budget=_budget(1), keep_recent_turns=1, head_chars=100, tail_chars=50
        )
    )

    out = await agent.run("go")
    assert out.payloads[0] == "done"

    # The final LLM call saw the older c1 output collapsed...
    view_outputs = _outputs(llm.views[-1])  # type: ignore[attr-defined]
    c1_view = next(m for m in view_outputs if m.call_id == "c1")
    assert NOTICE in c1_view.text
    assert len(c1_view.text) < len(_BIG_PAYLOAD)

    # ...but the durable transcript kept the full output for rollback / resume.
    log_c1 = next(m for m in _outputs(agent.transcript.messages) if m.call_id == "c1")
    assert log_c1.text == _BIG_PAYLOAD
