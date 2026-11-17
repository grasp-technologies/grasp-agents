"""
Reasoning replay on Amazon Bedrock, against the real endpoints.

GPT and Claude each run on the Bedrock runtime (``platform="bedrock"``) and on
Bedrock Mantle (``platform="bedrock_mantle"``). A conversation continues on the
same model, on the same vendor's model on the other platform, and on the other
vendor. Each platform encrypts reasoning with its own keys, so a replayed
foreign payload is rejected; the framework must drop it.

Needs a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) or AWS credentials with
access to the models below in AWS_REGION (default us-east-1). Run with:
uv run --no-sync pytest tests/integration/test_bedrock_live.py -m integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from grasp_agents.llm.cloud_llm import CloudLLM
from grasp_agents.llm_providers.anthropic.anthropic_llm import AnthropicLLM
from grasp_agents.llm_providers.openai_responses.responses_llm import OpenAIResponsesLLM
from grasp_agents.types.items import InputMessageItem, ReasoningItem

_GPT_SETTINGS: dict[str, Any] = {
    "reasoning": {"effort": "medium", "summary": "detailed"},
    "include": ["reasoning.encrypted_content"],
    "store": False,
    "max_output_tokens": 2000,
}
_CLAUDE_SETTINGS: dict[str, Any] = {
    "max_tokens": 4000,
    "thinking": {"type": "adaptive", "display": "summarized"},
    "output_config": {"effort": "medium"},
}

# The runtime serves models through cross-region inference profiles; Mantle
# takes the plain model ID. Claude Opus 5 is pinned because it is served on
# both platforms.
_MODELS: dict[str, tuple[type[CloudLLM], str, str, dict[str, Any]]] = {
    "gpt-runtime": (
        OpenAIResponsesLLM,
        "global.openai.gpt-6-luna",
        "bedrock",
        _GPT_SETTINGS,
    ),
    "gpt-mantle": (
        OpenAIResponsesLLM,
        "openai.gpt-6-luna",
        "bedrock_mantle",
        _GPT_SETTINGS,
    ),
    "claude-runtime": (
        AnthropicLLM,
        "global.anthropic.claude-opus-5",
        "bedrock",
        _CLAUDE_SETTINGS,
    ),
    "claude-mantle": (
        AnthropicLLM,
        "anthropic.claude-opus-5",
        "bedrock_mantle",
        _CLAUDE_SETTINGS,
    ),
}

_PAIRS = [
    *((name, name) for name in _MODELS),
    ("gpt-mantle", "gpt-runtime"),
    ("gpt-runtime", "gpt-mantle"),
    ("claude-mantle", "claude-runtime"),
    ("claude-runtime", "claude-mantle"),
    ("gpt-runtime", "claude-runtime"),
    ("claude-mantle", "gpt-mantle"),
]


@pytest.fixture
def bedrock_region() -> str:
    if not (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    ):
        pytest.skip("no Bedrock credentials: AWS_BEARER_TOKEN_BEDROCK or AWS ones")
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _llm(name: str, region: str) -> CloudLLM:
    llm_cls, model_name, platform, settings = _MODELS[name]
    return llm_cls(
        model_name=model_name,
        platform=platform,
        platform_config={"aws_region": region},
        llm_settings=settings,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(("producer", "consumer"), _PAIRS)
async def test_conversation_continues_with_reasoning_replayed(
    producer: str, consumer: str, bedrock_region: str
) -> None:
    first = _llm(producer, bedrock_region)
    second = _llm(consumer, bedrock_region)
    history = [InputMessageItem.from_text("Think it through: what is 17 * 23?")]

    first_response = await first.generate_response(history)

    reasoning = [i for i in first_response.output if isinstance(i, ReasoningItem)]
    assert reasoning
    assert {(i.native_provider_name, i.platform) for i in reasoning} == {
        (first.native_provider_name, first.platform)
    }

    second_response = await second.generate_response(
        [
            *history,
            *first_response.output,
            InputMessageItem.from_text("Now add 9 to that."),
        ]
    )

    assert second_response.status == "completed"
    assert second_response.output_text
