"""
Model turns LiteLLM produced for Gemini, held untagged in a transcript: tool
call ids carrying a signature are split, and the turn is attributed to Gemini,
so no other provider receives its signatures or its ids.
"""

from __future__ import annotations

from typing import Any

import pytest

from grasp_agents.llm.thought_signatures import normalize_litellm_gemini_items
from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from tests.llm.test_native_provider_name import (
    _ANTHROPIC_PROVIDER,
    _OPENAI_PROVIDER,
    TextBlock,
    _anthropic_message,
    _anthropic_thinking_blocks,
    _openai_message_item,
    _openai_response,
    _StubAnthropicLLM,
    _StubOpenAIResponsesLLM,
)

_SIG = "EogGCoUGAR+thought/signature=="
_SIGNED_ID = f"call_1__thought__{_SIG}"


def _reasoning(native_provider_name: str | None = None) -> ReasoningItem:
    return ReasoningItem(
        summary=[ReasoningSummary(text="thinking")],
        encrypted_content=_SIG,
        native_provider_name=native_provider_name,
    )


def _tool_turn(call_fields: dict[str, Any] | None = None) -> list[InputItem]:
    return [
        InputMessageItem.from_text("add 1 and 2"),
        _reasoning(),
        FunctionToolCallItem(
            call_id=_SIGNED_ID,
            name="add",
            arguments='{"a": 1, "b": 2}',
            provider_specific_fields=call_fields,
        ),
        FunctionToolOutputItem(call_id=_SIGNED_ID, output="3"),
    ]


def _text_turn() -> list[InputItem]:
    return [
        InputMessageItem.from_text("why is the sky blue?"),
        _reasoning(),
        OutputMessageItem(
            status="completed",
            content=[OutputMessageText(text="scattering")],
            provider_specific_fields={"thought_signatures": [_SIG]},
        ),
        InputMessageItem.from_text("shorter"),
    ]


def _of[T](items: list[InputItem], typ: type[T]) -> list[T]:
    return [i for i in items if isinstance(i, typ)]


class TestNormalization:
    def test_signed_call_id_is_split_on_the_call_and_its_output(self) -> None:
        normalized = normalize_litellm_gemini_items(_tool_turn())

        (call,) = _of(normalized, FunctionToolCallItem)
        (output,) = _of(normalized, FunctionToolOutputItem)
        assert call.call_id == output.call_id == "call_1"
        assert call.provider_specific_fields == {"thought_signature": _SIG}

    def test_other_call_fields_are_kept_and_an_existing_signature_wins(self) -> None:
        normalized = normalize_litellm_gemini_items(
            _tool_turn({"other": "x", "thought_signature": "on-the-call"})
        )

        (call,) = _of(normalized, FunctionToolCallItem)
        assert call.provider_specific_fields == {
            "other": "x",
            "thought_signature": "on-the-call",
        }

    @pytest.mark.parametrize("history", [_tool_turn(), _text_turn()])
    def test_untagged_items_of_the_turn_are_attributed_to_gemini(
        self, history: list[InputItem]
    ) -> None:
        normalized = normalize_litellm_gemini_items(history)

        turn = [
            i
            for i in normalized
            if isinstance(i, (ReasoningItem, OutputMessageItem, FunctionToolCallItem))
        ]
        assert turn
        assert {i.native_provider_name for i in turn} == {"gemini"}

    def test_other_turns_and_tagged_items_are_left_alone(self) -> None:
        anthropic_turn: list[InputItem] = [
            InputMessageItem.from_text("earlier"),
            _reasoning(),
            OutputMessageItem(
                status="completed", content=[OutputMessageText(text="done")]
            ),
        ]
        tagged = _reasoning("vertex_ai")
        history = [*anthropic_turn, InputMessageItem.from_text("now"), tagged]
        history += _tool_turn()[2:]

        normalized = normalize_litellm_gemini_items(history)

        assert normalized[:3] == anthropic_turn
        assert normalized[4] is tagged

    def test_given_items_are_not_modified(self) -> None:
        history = _tool_turn()
        snapshot = [i.model_copy(deep=True) for i in history]

        normalize_litellm_gemini_items(history)

        assert history == snapshot


class TestForeignProvidersGetPlainTurns:
    @pytest.mark.asyncio
    async def test_anthropic_gets_plain_ids_and_no_gemini_thinking(self) -> None:
        llm = _StubAnthropicLLM(
            model_name="claude-sonnet-4-5",
            api_provider=_ANTHROPIC_PROVIDER,
            served=_anthropic_message([TextBlock(type="text", text="ok")]),
        )

        await llm.generate_response(_tool_turn())

        assert _anthropic_thinking_blocks(llm.captured_api_input) == []
        blocks = [
            b
            for m in llm.captured_api_input
            if isinstance(m["content"], list)
            for b in m["content"]
        ]
        assert [b["id"] for b in blocks if b.get("type") == "tool_use"] == ["call_1"]
        assert [b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"] == [
            "call_1"
        ]

    @pytest.mark.asyncio
    async def test_openai_responses_gets_plain_ids_and_no_gemini_reasoning(
        self,
    ) -> None:
        llm = _StubOpenAIResponsesLLM(
            model_name="gpt-5.1",
            api_provider=_OPENAI_PROVIDER,
            served=_openai_response([_openai_message_item("ok")]),
        )

        await llm.generate_response(_tool_turn())

        captured = [d for d in llm.captured_api_input if isinstance(d, dict)]
        assert not [d for d in captured if d.get("type") == "reasoning"]
        assert [d["call_id"] for d in captured if "call_id" in d] == [
            "call_1",
            "call_1",
        ]
