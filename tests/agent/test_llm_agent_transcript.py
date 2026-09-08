"""``LLMAgentTranscript``: a read-only view over the log its manager owns."""

import pytest

from grasp_agents.agent.context_window import ContextWindowManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.errors import TranscriptInvariantError
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
)


def _sys(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="system")


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _assistant(text: str) -> OutputMessageItem:
    return OutputMessageItem(content=[OutputMessageText(text=text)], status="completed")


def _call(cid: str, name: str = "t", arguments: str = "{}") -> FunctionToolCallItem:
    return FunctionToolCallItem(call_id=cid, name=name, arguments=arguments)


def _result(cid: str, output: object = "ok") -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id=cid, output=output)


def _cw(messages: list[InputItem] | None = None) -> ContextWindowManager:
    cw = ContextWindowManager(model_name="mock", source="t")
    if messages:
        cw.add_messages(messages)
    return cw


class TestAppend:
    def test_add_messages_appends_and_returns_length(self):
        cw = _cw()
        assert cw.add_messages([_sys("sys")]) == 1
        assert cw.add_messages([_user("Hello"), _assistant("Hi!")]) == 3

        t = cw.transcript
        assert len(t) == 3
        first = t[0]
        assert isinstance(first, InputMessageItem)
        assert first.role == "system"
        assert isinstance(t[1], InputMessageItem)
        assert isinstance(t[2], OutputMessageItem)

    def test_tool_call_items(self):
        cw = _cw([_call("tc1", name="search"), _result("tc1", "result")])
        t = cw.transcript
        assert len(t) == 2
        assert isinstance(t[0], FunctionToolCallItem)
        assert isinstance(t[1], FunctionToolOutputItem)

    def test_preserves_order(self):
        items: list[InputItem] = [
            _user("Hello"),
            _assistant("Hi"),
            _call("c1"),
            _result("c1", "r"),
        ]
        t = _cw(items).transcript
        assert list(t) == items
        assert [type(m) for m in t] == [
            InputMessageItem,
            OutputMessageItem,
            FunctionToolCallItem,
            FunctionToolOutputItem,
        ]


class TestView:
    def test_sequence_protocol(self):
        items: list[InputItem] = [_user(str(i)) for i in range(4)]
        t = _cw(items).transcript
        assert len(t) == 4
        assert t[-1] is items[3]
        assert t[1:3] == items[1:3]
        assert list(t) == items
        assert list(reversed(t)) == items[::-1]
        assert items[2] in t
        assert list(t.messages) == items

    def test_standalone_view_over_a_list(self):
        backing: list[InputItem] = [_user("q")]
        t = LLMAgentTranscript(backing)
        assert len(t) == 1
        backing.append(_user("r"))
        assert len(t) == 2  # the view shares the list it was given
        assert LLMAgentTranscript().is_empty

    def test_view_tracks_every_manager_write(self):
        cw = _cw()
        t = cw.transcript
        assert t.is_empty

        cw.add_messages([_sys("sys")])
        assert not t.is_empty
        assert len(t) == 1

        cw.truncate_transcript(0)
        assert t.is_empty

        cw.replace_transcript([_user("a"), _user("b")])
        assert len(t) == 2

        cw.clear_transcript()
        assert t.is_empty
        assert cw.transcript is t  # one view for the manager's lifetime

    def test_truncate(self):
        cw = _cw([_user(str(i)) for i in range(4)])
        cw.truncate_transcript(2)
        assert len(cw.transcript) == 2
        cw.truncate_transcript(5)  # past the end → unchanged
        assert len(cw.transcript) == 2

    def test_view_has_no_mutators(self):
        t = _cw([_sys("sys")]).transcript
        for name in ("update", "clear", "truncate", "append", "extend"):
            assert not hasattr(t, name)
        with pytest.raises(AttributeError):
            t.messages = []  # pyright: ignore[reportAttributeAccessIssue]

    def test_repr(self):
        t = _cw([_sys("sys"), _user("x")]).transcript
        assert "2" in repr(t)

    def test_owes_response(self):
        cw = _cw()
        t = cw.transcript
        assert not t.owes_response  # empty

        cw.add_messages([_user("q")])
        assert t.owes_response  # unanswered user message

        cw.add_messages([_assistant("a")])
        assert not t.owes_response  # completed assistant turn

        cw.add_messages([_call("c1")])
        assert t.owes_response  # dangling tool call

        cw.add_messages([_result("c1", "r")])
        assert t.owes_response  # tool result awaiting a continuation


class TestFullConversation:
    def test_simulates_agentic_loop(self):
        """Simulate a full loop: system, user, assistant(+tools), tool, assistant."""
        cw = _cw([_sys("You are a calculator.")])

        # User turn
        cw.add_messages([_user("What is 2+2?")])

        # First LLM response (with tool call)
        cw.add_messages(
            [
                _assistant("Let me calculate."),
                _call("add_1", name="add", arguments='{"a":2,"b":2}'),
            ]
        )

        # Tool execution result
        cw.add_messages([_result("add_1", {"result": 4})])

        # Second LLM response (final answer)
        cw.add_messages([_assistant("2 + 2 = 4")])

        t = cw.transcript
        assert len(t) == 6
        assert [type(m).__name__ for m in t] == [
            "InputMessageItem",  # system
            "InputMessageItem",  # user
            "OutputMessageItem",  # assistant text
            "FunctionToolCallItem",  # tool call
            "FunctionToolOutputItem",  # tool result
            "OutputMessageItem",  # final answer
        ]


class TestToolCallPairing:
    """`validate_tool_call_pairing` enforces the provider pairing invariant."""

    def test_valid_pairing_passes(self):
        t = _cw([_sys("sys"), _user("go"), _call("c1"), _result("c1")]).transcript
        t.validate_tool_call_pairing()  # no raise

    def test_multiple_calls_then_results_pass(self):
        t = _cw([_call("c1"), _call("c2"), _result("c1"), _result("c2")]).transcript
        t.validate_tool_call_pairing()

    def test_same_turn_text_between_call_and_result_allowed(self):
        """Assistant text/reasoning is same-turn — allowed mid-batch."""
        t = _cw([_call("c1"), _assistant("thinking"), _result("c1")]).transcript
        t.validate_tool_call_pairing()

    def test_dangling_tool_call_raises(self):
        t = _cw([_call("c1")]).transcript
        with pytest.raises(TranscriptInvariantError, match="unresolved"):
            t.validate_tool_call_pairing()

    def test_user_message_between_call_and_result_raises(self):
        t = _cw([_call("c1"), _user("interrupt"), _result("c1")]).transcript
        with pytest.raises(TranscriptInvariantError, match="not resolved before"):
            t.validate_tool_call_pairing()
