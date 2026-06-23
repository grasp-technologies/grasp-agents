"""
previous_response_id / conversation are explicit per-call params, not settings.

Both are server-side multi-turn-state pointers (not stable model config), so
they are passed straight to ``responses.create`` next to ``input``/``tools`` and
gate the "send only items after the last model output" slicing.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)

_OPENAI = APIProvider(name="openai", base_url=None, api_key="sk-x")

# A user turn, a model reply, then a new user turn — slicing keeps only the
# trailing item(s) that postdate the model's last output.
_API_INPUT: list[Any] = [
    {"role": "user", "content": "hi"},
    {"type": "message", "role": "assistant", "content": "hello"},
    {"role": "user", "content": "next"},
]


def test_server_state_params_are_not_settings() -> None:
    keys = OpenAIResponsesLLMSettings.__optional_keys__
    assert "previous_response_id" not in keys
    assert "conversation" not in keys
    # ``instructions`` is the system prompt (built from input); not a setting.
    assert "instructions" not in keys


def test_make_api_input_extracts_server_state() -> None:
    # Passed as per-call kwargs, they are extracted into the api-call params
    # (next to input/tools), NOT left in the settings bag.
    llm = OpenAIResponsesLLM(model_name="gpt-4o", api_provider=_OPENAI)
    params = llm._make_api_input(
        [], previous_response_id="resp_1", conversation="conv_1"
    )
    assert params["previous_response_id"] == "resp_1"
    assert params["conversation"] == "conv_1"
    extra = params.get("extra_settings", {})
    assert "previous_response_id" not in extra
    assert "conversation" not in extra


@pytest.mark.asyncio
async def test_previous_response_id_threaded_and_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = OpenAIResponsesLLM(model_name="gpt-4o", api_provider=_OPENAI)
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(llm.client.responses, "create", fake_create)
    await llm._get_api_response(_API_INPUT, previous_response_id="resp_1")

    assert captured["previous_response_id"] == "resp_1"
    assert captured["input"] == _API_INPUT[2:]  # sliced to the new user turn


@pytest.mark.asyncio
async def test_conversation_threaded_and_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = OpenAIResponsesLLM(model_name="gpt-4o", api_provider=_OPENAI)
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(llm.client.responses, "create", fake_create)
    await llm._get_api_response(_API_INPUT, conversation="conv_1")

    assert captured["conversation"] == "conv_1"
    assert captured["input"] == _API_INPUT[2:]


@pytest.mark.asyncio
async def test_no_server_state_sends_full_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = OpenAIResponsesLLM(model_name="gpt-4o", api_provider=_OPENAI)
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(llm.client.responses, "create", fake_create)
    await llm._get_api_response(_API_INPUT)

    assert captured["input"] == _API_INPUT  # no slicing without server state
