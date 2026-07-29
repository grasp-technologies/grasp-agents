"""
How an output schema reaches the Responses API.

`apply_output_schema_via_provider` must send a *strict* json_schema so the
provider constrains generation, while leaving the parsing of the reply to
`LLM._validate_response`. Routing it through `responses.parse()` would do both,
and its parse rejects a reply whose JSON value is followed by extra bytes —
which some providers emit (Bedrock's decoder appends a redundant `}` for
certain models), turning a schema-valid response into a wasted re-sample.
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
        model="m",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _llm(**kwargs: Any) -> OpenAIResponsesLLM:
    llm = OpenAIResponsesLLM(
        model_name="google.gemma-4-31b",
        api_provider={"name": "x", "base_url": "https://x/openai/v1", "api_key": "k"},
        apply_output_schema_via_provider=True,
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
    async def test_existing_text_settings_are_preserved(self) -> None:
        """A model configured with `text.verbosity` must keep it."""
        llm = _llm(llm_settings={"text": {"verbosity": "medium"}})
        await llm._get_api_response(
            [], api_output_schema=Answer, text={"verbosity": "medium"}
        )

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
