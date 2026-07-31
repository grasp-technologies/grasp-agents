"""
How an output schema reaches the Responses API.

`apply_output_schema_via_provider` must send a *strict* json_schema so the
provider constrains generation, while leaving the parsing of the reply to
`LLM._validate_response`. Routing it through `responses.parse()` would do both,
and its parse rejects a reply whose JSON value is followed by extra bytes —
which some providers emit, because a schema constrains the value but not the
stopping, turning a schema-valid response into a wasted re-sample.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import Response as OpenAIResponse
from pydantic import BaseModel

from grasp_agents.llm_providers.openai_responses.responses_llm import (
    OpenAIResponsesLLM,
)


class Answer(BaseModel):
    capital: str
    population_millions: int


def _empty_response() -> OpenAIResponse:
    return OpenAIResponse.model_construct(
        id="resp_1",
        created_at=0.0,
        object="response",
        status="completed",
        model="m",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _llm(*, tolerate: bool = True, **kwargs: Any) -> OpenAIResponsesLLM:
    llm = OpenAIResponsesLLM(
        model_name="google.gemma-4-31b",
        api_provider={"name": "x", "base_url": "https://x/openai/v1", "api_key": "k"},
        apply_output_schema_via_provider=True,
        tolerate_output_around_json=tolerate,
        **kwargs,
    )
    object.__setattr__(llm.client, "responses", AsyncMock())
    llm.client.responses.create.return_value = _empty_response()  # type: ignore[attr-defined]
    llm.client.responses.parse.return_value = _empty_response()  # type: ignore[attr-defined]
    return llm


class TestOutputSchemaTransport:
    @pytest.mark.asyncio
    async def test_uses_create_not_parse(self) -> None:
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer)

        llm.client.responses.create.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.responses.parse.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_schema_is_sent_strict(self) -> None:
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer)

        fmt = llm.client.responses.create.await_args.kwargs["text"]["format"]  # type: ignore[attr-defined]
        assert fmt["type"] == "json_schema"
        assert fmt["strict"] is True
        # Strict mode's own requirements, so enforcement is not silently weakened.
        assert fmt["schema"]["additionalProperties"] is False
        assert set(fmt["schema"]["required"]) == {"capital", "population_millions"}

    @pytest.mark.asyncio
    async def test_caller_text_settings_are_merged_not_replaced(self) -> None:
        """`format` is added to the caller's `text`, not swapped in for it."""
        llm = _llm()
        await llm._get_api_response(
            [], api_output_schema=Answer, text={"verbosity": "medium"}
        )

        text = llm.client.responses.create.await_args.kwargs["text"]  # type: ignore[attr-defined]
        assert text["verbosity"] == "medium"
        assert text["format"]["strict"] is True

    @pytest.mark.asyncio
    async def test_top_level_verbosity_is_relocated_not_dropped(self) -> None:
        """
        `create()` rejects a top-level `verbosity`; only `parse()` accepts it.

        Forwarding it unchanged would raise `TypeError` for any model that sets
        it, so it has to move under `text`, where the API reads it from.
        """
        llm = _llm()
        await llm._get_api_response([], api_output_schema=Answer, verbosity="medium")

        kwargs = llm.client.responses.create.await_args.kwargs  # type: ignore[attr-defined]
        assert "verbosity" not in kwargs, "would be a TypeError against the real SDK"
        assert kwargs["text"]["verbosity"] == "medium"

    @pytest.mark.asyncio
    async def test_explicit_text_verbosity_wins_over_top_level(self) -> None:
        llm = _llm()
        await llm._get_api_response(
            [], api_output_schema=Answer, verbosity="low", text={"verbosity": "high"}
        )

        assert llm.client.responses.create.await_args.kwargs["text"]["verbosity"] == (  # type: ignore[attr-defined]
            "high"
        )

    @pytest.mark.asyncio
    async def test_forwarded_kwargs_are_all_accepted_by_the_real_create(self) -> None:
        """
        Guards the whole switch, not just `verbosity`.

        The mocked client accepts anything, so a kwarg the real SDK rejects
        would pass every other test here and fail in production. Bind the
        captured call against the genuine signature instead.
        """
        import inspect

        from openai import AsyncOpenAI

        llm = _llm()
        await llm._get_api_response(
            [],
            api_output_schema=Answer,
            verbosity="medium",
            reasoning={"effort": "none"},
            max_output_tokens=256,
            store=False,
        )

        kwargs = llm.client.responses.create.await_args.kwargs  # type: ignore[attr-defined]
        real = AsyncOpenAI(api_key="x").responses.create
        inspect.signature(real).bind(**kwargs)

    @pytest.mark.asyncio
    async def test_model_level_text_settings_survive_end_to_end(self) -> None:
        """
        A model configured with `text.verbosity` keeps it.

        Driven through `_generate_response_once`, because that is where
        `llm_settings` is merged and forwarded — `_get_api_response` only ever
        sees settings as keyword arguments.
        """
        llm = _llm(llm_settings={"text": {"verbosity": "medium"}})
        await llm._generate_response_once([], output_schema=Answer)

        text = llm.client.responses.create.await_args.kwargs["text"]  # type: ignore[attr-defined]
        assert text["verbosity"] == "medium"
        assert text["format"]["strict"] is True

    @pytest.mark.asyncio
    async def test_non_pydantic_schema_falls_back_to_parse(self) -> None:
        """`str` and other non-model schemas keep the previous transport."""
        llm = _llm()
        await llm._get_api_response([], api_output_schema=str)

        llm.client.responses.parse.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.responses.create.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_caller_supplied_format_is_not_silently_replaced(self) -> None:
        """`.parse()` raises on this; the create() route must not differ."""
        llm = _llm()
        with pytest.raises(TypeError, match="Cannot mix and match"):
            await llm._get_api_response(
                [],
                api_output_schema=Answer,
                text={"format": {"type": "json_object"}},
            )


class TestTransportUnchangedWhenNotTolerating:
    """
    With `tolerate_output_around_json` off — the default — a model must run the
    exact code path it ran before the flag existed.
    """

    @pytest.mark.asyncio
    async def test_uses_parse(self) -> None:
        llm = _llm(tolerate=False)
        await llm._get_api_response([], api_output_schema=Answer)

        llm.client.responses.parse.assert_awaited_once()  # type: ignore[attr-defined]
        llm.client.responses.create.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_schema_still_reaches_the_provider(self) -> None:
        """Enforcement is not what the flag gates — only the transport is."""
        llm = _llm(tolerate=False)
        await llm._get_api_response([], api_output_schema=Answer)

        assert llm.client.responses.parse.await_args.kwargs["text_format"] is Answer  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_default_is_not_tolerating(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="m",
            api_provider={"name": "x", "base_url": "https://x", "api_key": "k"},
        )
        assert llm.tolerate_output_around_json is False

    @pytest.mark.asyncio
    async def test_gate_off_sends_no_schema(self) -> None:
        llm = OpenAIResponsesLLM(
            model_name="m",
            api_provider={"name": "x", "base_url": "https://x", "api_key": "k"},
            apply_output_schema_via_provider=False,
        )
        object.__setattr__(llm.client, "responses", AsyncMock())
        llm.client.responses.create.return_value = _empty_response()  # type: ignore[attr-defined]

        await llm._get_api_response([], api_output_schema=Answer)

        assert "text" not in llm.client.responses.create.await_args.kwargs  # type: ignore[attr-defined]
