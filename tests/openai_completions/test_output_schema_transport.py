"""
How an output schema reaches the Chat Completions API.

Mirrors `tests/openai_responses/test_output_schema_transport.py`: with
`tolerate_output_around_json` set, the strict json_schema must still reach the
provider — so generation stays constrained — while the reply's parsing is left
to `LLM._validate_response`. `.parse()` would do both, and its parse rejects a
reply whose JSON value is followed by extra bytes, which is what the tolerance
exists to forgive.

Chat Completions is affected by the same provider behaviour as Responses: on
Bedrock's Gemma 4, a few percent of schema-constrained replies carry trailing
bytes after the closing brace.
"""

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from grasp_agents.llm_providers.openai_completions.completions_llm import OpenAILLM


class Answer(BaseModel):
    capital: str
    population_millions: int


def _completion() -> ChatCompletion:
    """Minimal reply the converter accepts — a choice with schema-valid content."""
    return ChatCompletion.model_validate(
        {
            "id": "cmpl_1",
            "created": 0,
            "model": "m",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"capital":"Paris","population_millions":11}',
                    },
                }
            ],
        }
    )


def _llm(*, tolerate: bool = True, **kwargs: Any) -> OpenAILLM:
    llm = OpenAILLM(
        model_name="google.gemma-4-31b",
        api_provider={"name": "x", "base_url": "https://x/openai/v1", "api_key": "k"},
        apply_output_schema_via_provider=True,
        tolerate_output_around_json=tolerate,
        **kwargs,
    )
    object.__setattr__(llm.client, "chat", AsyncMock())
    object.__setattr__(llm.client, "beta", AsyncMock())
    llm.client.chat.completions.create.return_value = _completion()  # type: ignore[attr-defined]
    llm.client.beta.chat.completions.parse.return_value = _completion()  # type: ignore[attr-defined]
    return llm


class TestOutputSchemaTransport:
    @pytest.mark.asyncio
    async def test_uses_create_not_parse(self) -> None:
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer)

        llm.client.chat.completions.create.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.beta.chat.completions.parse.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_schema_is_sent_strict(self) -> None:
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer)

        fmt = llm.client.chat.completions.create.await_args.kwargs["response_format"]  # type: ignore[attr-defined]
        assert fmt["type"] == "json_schema"
        js = fmt["json_schema"]
        assert js["strict"] is True
        assert js["name"] == "Answer"
        # Strict mode's own requirements, so enforcement is not silently weakened.
        assert js["schema"]["additionalProperties"] is False
        assert set(js["schema"]["required"]) == {"capital", "population_millions"}

    @pytest.mark.asyncio
    async def test_stream_false_is_sent(self) -> None:
        """`create()` defaults to non-streaming only if we say so explicitly."""
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer)

        assert llm.client.chat.completions.create.await_args.kwargs["stream"] is False  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_forwarded_kwargs_are_all_accepted_by_the_real_create(self) -> None:
        """
        The mocked client accepts anything, so a kwarg the real SDK rejects would
        pass every other test here and fail in production. Bind the captured call
        against the genuine signature instead.

        This is what caught `verbosity` on the Responses side, where `parse()`
        accepted a parameter `create()` does not.
        """
        llm = _llm()
        await llm._get_api_response(
            [],
            api_output_schema=Answer,
            temperature=0.25,
            max_tokens=256,
            reasoning_effort="none",
        )

        kwargs = llm.client.chat.completions.create.await_args.kwargs  # type: ignore[attr-defined]
        real = AsyncOpenAI(api_key="x").chat.completions.create
        inspect.signature(real).bind(**kwargs)

    @pytest.mark.asyncio
    async def test_non_pydantic_schema_falls_back_to_parse(self) -> None:
        """`str` and other non-model schemas keep the previous transport."""
        llm = _llm()
        await llm._get_api_response([], api_output_schema=str)

        llm.client.beta.chat.completions.parse.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.chat.completions.create.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_non_strict_tool_is_still_rejected(self) -> None:
        """
        `.parse()` refuses a tool without `strict: true`; the create() route must
        not quietly start accepting one.
        """
        llm = _llm()
        loose_tool = {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
        with pytest.raises(ValueError, match="strict"):
            await llm._get_api_response(
                [], api_tools=[loose_tool], api_output_schema=Answer
            )

    @pytest.mark.asyncio
    async def test_non_function_tool_is_still_rejected(self) -> None:
        llm = _llm()
        with pytest.raises(ValueError, match="function"):
            await llm._get_api_response(
                [], api_tools=[{"type": "custom"}], api_output_schema=Answer
            )

    @pytest.mark.asyncio
    async def test_strict_tool_passes_through(self) -> None:
        llm = _llm()
        strict_tool = {
            "type": "function",
            "function": {
                "name": "f",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
        await llm._get_api_response(
            [], api_tools=[strict_tool], api_output_schema=Answer
        )

        assert llm.client.chat.completions.create.await_args.kwargs["tools"] == [  # type: ignore[attr-defined]
            strict_tool
        ]

    @pytest.mark.asyncio
    async def test_model_level_settings_survive_end_to_end(self) -> None:
        """
        Driven through `_generate_response_once`, because that is where
        `llm_settings` is merged and forwarded — `_get_api_response` only ever
        sees settings as keyword arguments.
        """
        llm = _llm(llm_settings={"temperature": 0.25})
        await llm._generate_response_once([], output_schema=Answer)

        kwargs = llm.client.chat.completions.create.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["temperature"] == pytest.approx(0.25)
        assert kwargs["response_format"]["json_schema"]["strict"] is True


class TestTransportUnchangedWhenNotTolerating:
    """
    With `tolerate_output_around_json` off — the default — a model must run the
    exact code path it ran before the flag existed.
    """

    @pytest.mark.asyncio
    async def test_uses_parse(self) -> None:
        llm = _llm(tolerate=False)
        await llm._get_api_response([], api_output_schema=Answer)

        llm.client.beta.chat.completions.parse.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.chat.completions.create.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_schema_still_reaches_the_provider(self) -> None:
        """Enforcement is not what the flag gates — only the transport is."""
        llm = _llm(tolerate=False)
        await llm._get_api_response([], api_output_schema=Answer)

        kwargs = llm.client.beta.chat.completions.parse.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["response_format"] is Answer

    @pytest.mark.asyncio
    async def test_default_is_not_tolerating(self) -> None:
        llm = OpenAILLM(
            model_name="m",
            api_provider={"name": "x", "base_url": "https://x", "api_key": "k"},
        )
        assert llm.tolerate_output_around_json is False

    @pytest.mark.asyncio
    async def test_gate_off_sends_no_schema(self) -> None:
        llm = OpenAILLM(
            model_name="m",
            api_provider={"name": "x", "base_url": "https://x", "api_key": "k"},
            apply_output_schema_via_provider=False,
        )
        object.__setattr__(llm.client, "chat", AsyncMock())
        llm.client.chat.completions.create.return_value = _completion()  # type: ignore[attr-defined]

        await llm._get_api_response([], api_output_schema=Answer)

        kwargs = llm.client.chat.completions.create.await_args.kwargs  # type: ignore[attr-defined]
        assert "response_format" not in kwargs
