"""LiteLLM non-streaming conversion: Gemini thought signatures on tool calls."""

from __future__ import annotations

from typing import Any

from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)

from grasp_agents.llm_providers.litellm.provider_output_to_response import (
    provider_output_to_response,
)
from grasp_agents.types.items import FunctionToolCallItem

_SIG = "EogGCoUGAR-thought-signature"


def _call(**fields: Any) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        type="function",
        function=Function(name="add", arguments='{"a": 1, "b": 2}'),
        **fields,
    )


def _completion(tool_call: ChatCompletionMessageToolCall) -> ModelResponse:
    message = Message(content=None, role="assistant", tool_calls=[tool_call])
    return ModelResponse(
        id="resp_1",
        model="gemini-3.1-flash-lite",
        choices=[Choices(index=0, finish_reason="tool_calls", message=message)],
    )


def _only_call(response: Any) -> FunctionToolCallItem:
    (call,) = [i for i in response.output if isinstance(i, FunctionToolCallItem)]
    return call


class TestToolCallThoughtSignature:
    """
    LiteLLM embeds the Gemini signature into the tool call id as well as
    ``provider_specific_fields``; other providers reject such ids, so the item
    keeps the plain id and carries the signature in its fields.
    """

    def test_signature_embedded_in_call_id_is_split_out(self) -> None:
        response = provider_output_to_response(
            _completion(_call(id=f"call_1__thought__{_SIG}"))
        )

        call = _only_call(response)
        assert call.call_id == "call_1"
        assert call.provider_specific_fields == {"thought_signature": _SIG}

    def test_signature_on_the_call_wins_over_the_id(self) -> None:
        response = provider_output_to_response(
            _completion(
                _call(
                    id="call_1__thought__stale",
                    provider_specific_fields={"thought_signature": _SIG},
                )
            )
        )

        call = _only_call(response)
        assert call.call_id == "call_1"
        assert call.provider_specific_fields == {"thought_signature": _SIG}

    def test_other_fields_on_the_call_are_kept(self) -> None:
        response = provider_output_to_response(
            _completion(
                _call(
                    id=f"call_1__thought__{_SIG}",
                    provider_specific_fields={"other": "x"},
                )
            )
        )

        call = _only_call(response)
        assert call.provider_specific_fields == {
            "other": "x",
            "thought_signature": _SIG,
        }

    def test_plain_call_id_is_untouched(self) -> None:
        response = provider_output_to_response(_completion(_call(id="call_1")))

        call = _only_call(response)
        assert call.call_id == "call_1"
        assert call.provider_specific_fields is None
