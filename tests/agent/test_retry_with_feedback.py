"""
Retry with feedback — a final answer the output parser rejects stays in the
transcript, and the parser's error text follows it as the next user message,
so the retry corrects the answer instead of regenerating it blind:

* the retry sees the prompt, the rejected answer, then the error text
* each later retry sees every earlier rejection; once an answer parses, the
  rejected answers and their feedback leave the transcript
* the feedback message is surfaced as a ``UserMessageEvent``
* a final failure (no retries, or the last one) settles exactly as before:
  no rejected answer, no feedback
* a reset agent's retry continues the conversation; its next run starts clean
* a failure raised by the loop (not the parser) stores no feedback
* a ``final_answer`` tool call is retried the same way, its pairing intact
"""

from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent, render_retry_feedback
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import UserMessageEvent
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from tests._helpers import _text_response, _tool_call_response
from tests.durability.test_durability_resume import (
    _FailsOnceOnCallLLM,
    _RecordingLLM,
)
from tests.durability.test_loop_durability import (
    _EchoTool,
    _make_agent,
    _make_answer_agent,
)

# Each marker holds letters outside the hex alphabet: it never matches an item id.
PROMPT = "Write the lesson."
ERROR_TEXT = "unclosed reading fence at line 41"
REJECTED_ANSWER = "rejected lesson one"
SECOND_REJECTED_ANSWER = "rejected lesson two"
ACCEPTED_ANSWER = "corrected lesson"


def _reject_answers_marked_rejected(agent: LLMAgent[str, Any, None]) -> None:
    @agent.add_output_parser
    def parse(final_answer: str, *, in_args: Any = None, exec_id: str) -> Any:
        del in_args, exec_id
        if "rejected" in final_answer:
            raise ValueError(ERROR_TEXT)
        return agent.parse_output_default(final_answer)


def _position(items: list[Any], text: str) -> int:
    """Index of the single item whose rendering contains ``text``."""
    positions = [i for i, item in enumerate(items) if text in str(item)]
    assert len(positions) == 1, (text, positions)
    return positions[0]


def _mentions(items: list[Any], text: str) -> bool:
    return any(text in str(item) for item in items)


class TestRetryWithFeedback:
    @pytest.mark.asyncio
    async def test_retry_sees_rejected_answer_then_error_after_prompt(self) -> None:
        llm = _RecordingLLM(
            responses_queue=[
                _text_response(REJECTED_ANSWER),
                _text_response(ACCEPTED_ANSWER),
            ]
        )
        agent, _ = _make_agent([], llm=llm, max_retries=1)
        _reject_answers_marked_rejected(agent)

        out = await agent.run(PROMPT)

        assert out.payloads[0] == ACCEPTED_ANSWER
        retry_input = llm.recorded_inputs[1]
        assert (
            _position(retry_input, PROMPT)
            < _position(retry_input, REJECTED_ANSWER)
            < _position(retry_input, ERROR_TEXT)
        )

    @pytest.mark.asyncio
    async def test_a_successful_retry_keeps_only_the_prompt_and_the_accepted_answer(
        self,
    ) -> None:
        llm = _RecordingLLM(
            responses_queue=[
                _text_response(REJECTED_ANSWER),
                _text_response(SECOND_REJECTED_ANSWER),
                _text_response(ACCEPTED_ANSWER),
            ]
        )
        agent, _ = _make_agent([], llm=llm, max_retries=2)
        _reject_answers_marked_rejected(agent)

        out = await agent.run(PROMPT)

        assert out.payloads[0] == ACCEPTED_ANSWER
        last_retry_input = llm.recorded_inputs[2]
        assert (
            _position(last_retry_input, PROMPT)
            < _position(last_retry_input, REJECTED_ANSWER)
            < _position(last_retry_input, SECOND_REJECTED_ANSWER)
        )
        assert sum(ERROR_TEXT in str(item) for item in last_retry_input) == 2
        messages = agent.transcript.messages
        assert len(messages) == 2
        assert _position(messages, PROMPT) < _position(messages, ACCEPTED_ANSWER)

    @pytest.mark.asyncio
    async def test_feedback_is_emitted_as_user_message_event(self) -> None:
        agent, _ = _make_agent(
            [_text_response(REJECTED_ANSWER), _text_response(ACCEPTED_ANSWER)],
            max_retries=1,
        )
        _reject_answers_marked_rejected(agent)

        events = [event async for event in agent.run_stream(PROMPT)]

        feedback_events = [
            event
            for event in events
            if isinstance(event, UserMessageEvent) and ERROR_TEXT in event.data.text
        ]
        assert len(feedback_events) == 1

    @pytest.mark.asyncio
    async def test_no_retry_prunes_the_rejected_answer(self) -> None:
        agent, _ = _make_agent([_text_response(REJECTED_ANSWER)], max_retries=0)
        _reject_answers_marked_rejected(agent)

        with pytest.raises(ProcRunError) as excinfo:
            await agent.run(PROMPT)

        assert ERROR_TEXT in str(excinfo.value.__cause__)
        messages = agent.transcript.messages
        assert len(messages) == 1
        assert _mentions(messages, PROMPT)
        assert agent._pending_retry_feedback is None

    @pytest.mark.asyncio
    async def test_exhausted_retries_prune_every_rejected_answer(self) -> None:
        agent, _ = _make_agent(
            [
                _text_response(REJECTED_ANSWER),
                _text_response(SECOND_REJECTED_ANSWER),
            ],
            max_retries=1,
        )
        _reject_answers_marked_rejected(agent)

        with pytest.raises(ProcRunError):
            await agent.run(PROMPT)

        messages = agent.transcript.messages
        assert len(messages) == 1
        assert _mentions(messages, PROMPT)
        assert agent._pending_retry_feedback is None

    @pytest.mark.asyncio
    async def test_reset_agent_retry_keeps_feedback_and_next_run_starts_clean(
        self,
    ) -> None:
        next_prompt = "Write the next lesson."
        llm = _RecordingLLM(
            responses_queue=[
                _text_response(REJECTED_ANSWER),
                _text_response(ACCEPTED_ANSWER),
                _text_response("next lesson"),
            ]
        )
        agent, _ = _make_agent([], llm=llm, max_retries=1, reset_transcript_on_run=True)
        _reject_answers_marked_rejected(agent)

        await agent.run(PROMPT)
        retry_input = llm.recorded_inputs[1]
        assert (
            _position(retry_input, PROMPT)
            < _position(retry_input, REJECTED_ANSWER)
            < _position(retry_input, ERROR_TEXT)
        )

        await agent.run(next_prompt)
        next_input = llm.recorded_inputs[2]
        assert _mentions(next_input, next_prompt)
        assert not _mentions(next_input, PROMPT)
        assert not _mentions(next_input, REJECTED_ANSWER)
        assert not _mentions(next_input, ERROR_TEXT)

    @pytest.mark.asyncio
    async def test_loop_failure_settles_as_before_without_feedback(self) -> None:
        # The tool round completes; the next generation fails inside the loop.
        llm = _FailsOnceOnCallLLM(
            responses_queue=[_tool_call_response("echo", '{"text": "hi"}', "c1")],
            fail_on_call=2,
        )
        agent, _ = _make_agent([], llm=llm, tools=[_EchoTool()], max_retries=0)
        _reject_answers_marked_rejected(agent)

        with pytest.raises(ProcRunError) as excinfo:
            await agent.run(PROMPT)

        assert "transient boom" in str(excinfo.value.__cause__)
        # Settled to the last closed round: the completed tool round stays.
        assert [type(message) for message in agent.transcript.messages] == [
            InputMessageItem,
            FunctionToolCallItem,
            FunctionToolOutputItem,
        ]
        assert not _mentions(agent.transcript.messages, ERROR_TEXT)
        assert agent._pending_retry_feedback is None

    @pytest.mark.asyncio
    async def test_final_answer_tool_call_retry_follows_its_synthetic_result(
        self,
    ) -> None:
        llm = _RecordingLLM(
            responses_queue=[
                _tool_call_response(
                    "final_answer", f'{{"answer": "{REJECTED_ANSWER}"}}', "fa_1"
                ),
                _tool_call_response(
                    "final_answer", f'{{"answer": "{ACCEPTED_ANSWER}"}}', "fa_2"
                ),
            ]
        )
        agent = _make_answer_agent([], llm=llm, max_retries=1)
        _reject_answers_marked_rejected(agent)

        out = await agent.run(PROMPT)

        assert out.payloads[0].answer == ACCEPTED_ANSWER
        retry_input = llm.recorded_inputs[1]
        call_position = next(
            i
            for i, item in enumerate(retry_input)
            if isinstance(item, FunctionToolCallItem) and item.call_id == "fa_1"
        )
        result_position = next(
            i
            for i, item in enumerate(retry_input)
            if isinstance(item, FunctionToolOutputItem) and item.call_id == "fa_1"
        )
        assert (
            _position(retry_input, PROMPT)
            < call_position
            < result_position
            < _position(retry_input, ERROR_TEXT)
        )
        assert not _mentions(agent.transcript.messages, ERROR_TEXT)
        agent.transcript.validate_tool_call_pairing()


def test_render_retry_feedback_quotes_error_verbatim() -> None:
    error_text = "unclosed reading fence at line 41\n  {'kind': \"fence\"}"

    feedback = render_retry_feedback(error_text)

    assert error_text in feedback
    assert feedback.startswith("Your answer was rejected: ")
    assert "Fix exactly that and return the whole corrected answer." in feedback
