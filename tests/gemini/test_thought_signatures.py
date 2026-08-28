"""
Gemini thought signatures across providers.

Gemini attaches its signed reasoning payload to regular text and function-call
parts, so it lands on ``OutputMessageItem`` / ``FunctionToolCallItem`` rather
than on ``ReasoningItem``. These tests pin the round trip end to end: the items
are stamped with Gemini's origin, Gemini gets its signatures back, and a
foreign-origin signature never reaches a Gemini request.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    Part,
)

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
)

_GEMINI_PROVIDER = APIProvider(name="gemini", base_url=None, api_key="test-key")

_MSG_SIG = b"msg-signature"
_FC_SIG = b"fc-signature"


@dataclass(frozen=True)
class _StubGeminiLLM(GeminiLLM):
    """Real Gemini converters/request-building; only the wire call is stubbed."""

    served: Any = None
    captured_api_input: list[Any] = field(default_factory=list)

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any:
        object.__setattr__(self, "captured_api_input", api_input)
        return self.served


def _signed_gemini_response() -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id="resp_test",
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(text="calling add", thought_signature=_MSG_SIG),
                        Part(
                            function_call=FunctionCall(
                                id="call_1", name="add", args={"a": 1, "b": 2}
                            ),
                            thought_signature=_FC_SIG,
                        ),
                    ],
                ),
                finish_reason=FinishReason.STOP,
            )
        ],
    )


def _signed_history(origin: str | None) -> list[InputItem]:
    return [
        InputMessageItem.from_text("add 1 and 2"),
        OutputMessageItem(
            status="completed",
            origin=origin,
            content=[OutputMessageText(text="calling add")],
            provider_specific_fields={
                "thought_signature": base64.b64encode(_MSG_SIG).decode()
            },
        ),
        FunctionToolCallItem(
            call_id="call_1",
            name="add",
            arguments='{"a": 1, "b": 2}',
            origin=origin,
            provider_specific_fields={
                "thought_signature": base64.b64encode(_FC_SIG).decode()
            },
        ),
        FunctionToolOutputItem(call_id="call_1", output="3"),
    ]


def _model_part_signatures(contents: list[Any]) -> list[Any]:
    return [
        part.thought_signature
        for content in contents
        if content.role == "model"
        for part in (content.parts or [])
    ]


@pytest.mark.asyncio
async def test_signed_items_are_stamped_with_gemini_origin() -> None:
    llm = _StubGeminiLLM(
        model_name="gemini-3-pro",
        api_provider=_GEMINI_PROVIDER,
        served=_signed_gemini_response(),
    )

    response = await llm.generate_response([InputMessageItem.from_text("add 1 and 2")])

    message = next(i for i in response.output if isinstance(i, OutputMessageItem))
    tool_call = next(i for i in response.output if isinstance(i, FunctionToolCallItem))
    assert message.origin == "gemini"
    assert tool_call.origin == "gemini"
    assert (message.provider_specific_fields or {})["thought_signature"] == (
        base64.b64encode(_MSG_SIG).decode()
    )


@pytest.mark.asyncio
async def test_gemini_gets_its_own_signatures_back() -> None:
    llm = _StubGeminiLLM(
        model_name="gemini-3-pro",
        api_provider=_GEMINI_PROVIDER,
        served=_signed_gemini_response(),
    )

    await llm.generate_response(_signed_history(origin="gemini"))

    assert _model_part_signatures(llm.captured_api_input) == [_MSG_SIG, _FC_SIG]


@pytest.mark.asyncio
async def test_foreign_signatures_never_reach_a_gemini_request() -> None:
    llm = _StubGeminiLLM(
        model_name="gemini-3-pro",
        api_provider=_GEMINI_PROVIDER,
        served=_signed_gemini_response(),
    )
    history = _signed_history(origin="openai")

    await llm.generate_response(history)

    assert _model_part_signatures(llm.captured_api_input) == [None, None]
    # The message and the tool call still go out, so tool-call pairing holds.
    model_parts = [
        part
        for content in llm.captured_api_input
        if content.role == "model"
        for part in (content.parts or [])
    ]
    assert model_parts[0].text == "calling add"
    assert model_parts[1].function_call is not None
    # The transcript keeps the signatures for a later Gemini turn.
    assert [
        (item.provider_specific_fields or {}).get("thought_signature")
        for item in history
        if isinstance(item, (OutputMessageItem, FunctionToolCallItem))
    ] == [
        base64.b64encode(_MSG_SIG).decode(),
        base64.b64encode(_FC_SIG).decode(),
    ]


@pytest.mark.asyncio
async def test_untagged_signatures_reach_gemini() -> None:
    llm = _StubGeminiLLM(
        model_name="gemini-3-pro",
        api_provider=_GEMINI_PROVIDER,
        served=_signed_gemini_response(),
    )

    await llm.generate_response(_signed_history(origin=None))

    assert _model_part_signatures(llm.captured_api_input) == [_MSG_SIG, _FC_SIG]
