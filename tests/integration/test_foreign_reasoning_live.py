"""
Reasoning payloads across providers, against the real APIs.

A turn produced by one provider (reasoning + tool call, with whatever signed
payload that vendor attaches) must continue on every other provider. The
framework drops reasoning items tagged for another vendor and strips foreign
thought signatures; Gemini 3 additionally needs a placeholder signature on a
function call it did not produce. Transcripts can also hold LiteLLM-Gemini
turns without a native provider name, their signature encoded in the call id.

Run with:
uv run --no-sync pytest tests/integration/test_foreign_reasoning_live.py -m integration
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.llm.cloud_llm import APIProvider, CloudLLM
from grasp_agents.types.content import ReasoningSummary
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_errors import LlmBadRequestError, LlmNotFoundError

if TYPE_CHECKING:
    from grasp_agents.tools.base import BaseTool

# Producers cover every kind of signed payload: OpenAI encrypted reasoning,
# Anthropic signed thinking, Gemini thought signatures (native and via
# LiteLLM), and Anthropic thinking via LiteLLM.
PRODUCERS = [
    "openai_responses",
    "anthropic",
    "gemini",
    "litellm_gemini",
    "litellm_anthropic",
]
CONSUMERS = [
    "openai_responses",
    "openai_completions",
    "anthropic",
    "gemini",
    "litellm_gemini",
    "litellm_anthropic",
    "litellm_openai",
]

_ADD_PROMPT = "What is 17 + 25? Use the add tool."
# Anthropic does not allow tool_choice="required" with thinking enabled, so
# the prompt has to earn the tool call.
_THINK_PROMPT = (
    "From the set {6, 11, 17, 19, 25, 33}, find the unique pair whose sum is "
    "exactly 42. Work through the combinations, then use the add tool to verify."
)
_THINKING_WITHOUT_REQUIRED = {"anthropic", "litellm_anthropic"}

_SignedTurn = tuple[InputMessageItem, list[OutputItem], list[FunctionToolOutputItem]]
_TURNS: dict[str, _SignedTurn] = {}


def _llm(name: str, keys: dict[str, str]) -> CloudLLM:
    if name == "openai_responses":
        from grasp_agents.llm_providers.openai_responses.responses_llm import (
            OpenAIResponsesLLM,
        )

        return OpenAIResponsesLLM(
            model_name="gpt-5.4-nano",
            llm_settings={
                "max_output_tokens": 2048,
                "reasoning": {"effort": "low", "summary": "auto"},
                "store": False,
                "include": ["reasoning.encrypted_content"],
            },
        )
    if name == "openai_completions":
        from grasp_agents.llm_providers.openai_completions.completions_llm import (
            OpenAILLM,
        )

        return OpenAILLM(
            model_name="gpt-5.4-nano", llm_settings={"max_completion_tokens": 1024}
        )
    if name == "anthropic":
        from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model_name="claude-haiku-4-5-20251001",
            api_provider=APIProvider(
                name="anthropic", base_url=None, api_key=keys["anthropic"]
            ),
            llm_settings={
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        )
    if name == "gemini":
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-3.1-flash-lite",
            api_provider=APIProvider(
                name="google", base_url=None, api_key=keys["google"]
            ),
            llm_settings={
                "max_output_tokens": 2048,
                "thinking_config": {"thinking_level": "low", "include_thoughts": True},
            },
        )

    from grasp_agents.llm_providers.litellm.lite_llm import LiteLLM

    if name == "litellm_gemini":
        return LiteLLM(
            model_name="gemini/gemini-3.1-flash-lite",
            api_provider=APIProvider(
                name="gemini", base_url=None, api_key=keys["google"]
            ),
            llm_settings={"reasoning_effort": "low", "max_completion_tokens": 1024},
        )
    if name == "litellm_anthropic":
        return LiteLLM(
            model_name="anthropic/claude-haiku-4-5-20251001",
            api_provider=APIProvider(
                name="anthropic", base_url=None, api_key=keys["anthropic"]
            ),
            llm_settings={
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        )
    if name == "litellm_openai":
        return LiteLLM(
            model_name="gpt-5.4-nano",
            api_provider=APIProvider(
                name="openai", base_url=None, api_key=keys["openai"]
            ),
            llm_settings={"max_completion_tokens": 1024},
        )
    raise ValueError(name)


def _tool_outputs(turn_output: list[OutputItem]) -> list[FunctionToolOutputItem]:
    outputs: list[FunctionToolOutputItem] = []
    for item in turn_output:
        if item.type != "function_call":
            continue
        args = json.loads(item.arguments)
        result = args["a"] + args["b"] if item.name == "add" else args["a"] * args["b"]
        outputs.append(
            FunctionToolOutputItem.from_tool_result(call_id=item.call_id, output=result)
        )
    return outputs


async def _signed_turn(
    producer: str, keys: dict[str, str], tools: dict[str, BaseTool[Any, Any, Any]]
) -> _SignedTurn:
    """One reasoning + tool-call turn per producer, shared across the matrix."""
    if producer not in _TURNS:
        llm = _llm(producer, keys)
        if producer in _THINKING_WITHOUT_REQUIRED:
            user_msg = InputMessageItem.from_text(_THINK_PROMPT)
            response = await llm.generate_response([user_msg], tools=tools)
        else:
            user_msg = InputMessageItem.from_text(_ADD_PROMPT)
            response = await llm.generate_response(
                [user_msg], tools=tools, tool_choice="required"
            )
        assert response.tool_call_items, f"{producer} did not call a tool"
        output = list(response.output)
        _TURNS[producer] = (user_msg, output, _tool_outputs(output))
    return _TURNS[producer]


@pytest.fixture
def keys(
    openai_api_key: str, anthropic_api_key: str, google_api_key: str
) -> dict[str, str]:
    return {
        "openai": openai_api_key,
        "anthropic": anthropic_api_key,
        "google": google_api_key,
    }


@pytest.mark.integration
class TestForeignReasoningAcrossProviders:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("consumer", CONSUMERS)
    @pytest.mark.parametrize("producer", PRODUCERS)
    async def test_turn_from_any_provider_continues_on_any_other(
        self,
        producer: str,
        consumer: str,
        keys: dict[str, str],
        parallel_tools: dict[str, BaseTool[Any, Any, Any]],
    ) -> None:
        user_msg, output, tool_outputs = await _signed_turn(
            producer, keys, parallel_tools
        )

        response = await _llm(consumer, keys).generate_response(
            [user_msg, *output, *tool_outputs], tools=parallel_tools
        )

        assert response.status == "completed"
        assert response.output_text


@pytest.mark.integration
class TestForeignReasoningIsRejectedAtTheWire:
    """
    Why the drop exists: replayed as-is, a foreign reasoning payload is
    rejected by the consuming vendor rather than ignored.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("producer", "consumer"),
        [
            # Anthropic's thinking signature is not OpenAI encrypted content.
            ("anthropic", "openai_responses"),
            # A Gemini reasoning summary has no payload; OpenAI looks the
            # item up by id and does not find it.
            ("gemini", "openai_responses"),
            # A thinking block without a valid signature.
            ("gemini", "anthropic"),
        ],
    )
    async def test_without_the_drop_the_request_fails(
        self,
        producer: str,
        consumer: str,
        keys: dict[str, str],
        parallel_tools: dict[str, BaseTool[Any, Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_msg, output, tool_outputs = await _signed_turn(
            producer, keys, parallel_tools
        )
        monkeypatch.setattr(
            CloudLLM, "_drop_foreign_reasoning", lambda self, items: items
        )

        with pytest.raises((LlmBadRequestError, LlmNotFoundError)):
            await _llm(consumer, keys).generate_response(
                [user_msg, *output, *tool_outputs], tools=parallel_tools
            )


def _untagged_with_signed_reasoning(
    output: list[OutputItem], signature: str
) -> list[OutputItem]:
    """A turn as stored without native provider names, reasoning signed."""
    untagged: list[OutputItem] = [
        item.model_copy(update={"native_provider_name": None})
        if isinstance(item, (ReasoningItem, OutputMessageItem, FunctionToolCallItem))
        else item
        for item in output
    ]
    if not any(isinstance(i, ReasoningItem) for i in untagged):
        untagged.insert(0, ReasoningItem(summary=[ReasoningSummary(text="thinking")]))
    return [
        item.model_copy(update={"encrypted_content": signature})
        if isinstance(item, ReasoningItem)
        else item
        for item in untagged
    ]


def _legacy_tool_turn(turn: _SignedTurn) -> list[Any]:
    user_msg, output, tool_outputs = turn
    encoded: dict[str, str] = {}
    signed_output: list[OutputItem] = []
    for item in output:
        if isinstance(item, FunctionToolCallItem) and item.provider_specific_fields:
            signature = item.provider_specific_fields["thought_signature"]
            encoded[item.call_id] = f"{item.call_id}__thought__{signature}"
            signed_output.append(
                item.model_copy(update={"call_id": encoded[item.call_id]})
            )
        else:
            signed_output.append(item)
    signature = next(iter(encoded.values())).partition("__thought__")[2]
    outputs = [
        o.model_copy(update={"call_id": encoded.get(o.call_id, o.call_id)})
        for o in tool_outputs
    ]
    return [
        user_msg,
        *_untagged_with_signed_reasoning(signed_output, signature),
        *outputs,
    ]


_TEXT_TURNS: dict[str, list[Any]] = {}


async def _legacy_text_turn(keys: dict[str, str]) -> list[Any]:
    if "litellm_gemini" not in _TEXT_TURNS:
        user_msg = InputMessageItem.from_text(
            "Why is the sky blue? Answer in one sentence."
        )
        response = await _llm("litellm_gemini", keys).generate_response([user_msg])
        (message,) = [i for i in response.output if isinstance(i, OutputMessageItem)]
        assert message.provider_specific_fields
        signature = message.provider_specific_fields["thought_signatures"][0]
        _TEXT_TURNS["litellm_gemini"] = [
            user_msg,
            *_untagged_with_signed_reasoning(list(response.output), signature),
            InputMessageItem.from_text("Now say it in three words."),
        ]
    return _TEXT_TURNS["litellm_gemini"]


@pytest.mark.integration
class TestUntaggedLiteLLMGeminiTurns:
    """
    LiteLLM-Gemini turns stored without native provider names: the tool call
    id carries the signature and the reasoning item a Gemini signature. Other
    providers reject both, so every consumer must get plain ids and no Gemini
    signatures.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("consumer", CONSUMERS)
    async def test_tool_call_turn_continues_on_any_provider(
        self,
        consumer: str,
        keys: dict[str, str],
        parallel_tools: dict[str, BaseTool[Any, Any, Any]],
    ) -> None:
        history = _legacy_tool_turn(
            await _signed_turn("litellm_gemini", keys, parallel_tools)
        )

        response = await _llm(consumer, keys).generate_response(
            history, tools=parallel_tools
        )

        assert response.status == "completed"
        assert response.output_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("consumer", CONSUMERS)
    async def test_text_turn_continues_on_any_provider(
        self, consumer: str, keys: dict[str, str]
    ) -> None:
        history = await _legacy_text_turn(keys)

        response = await _llm(consumer, keys).generate_response(history)

        assert response.status == "completed"
        assert response.output_text
