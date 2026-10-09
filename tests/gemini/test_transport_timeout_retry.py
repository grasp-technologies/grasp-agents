"""
A transport timeout must reach the API retry loop.

Mocks the SDK client method itself, so the real ``_get_api_response``,
``_raise_mapped``, ``map_api_error`` and retry loop all run: the loop filters on
``LlmErrorTuple``, so an unmapped exception skips retry and fallback entirely.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm.resilience import RetryPolicy
from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM
from grasp_agents.types.items import InputMessageItem

_USER_MSG = [InputMessageItem.from_text("hi")]


def _llm(api_retries: int = 1) -> GeminiLLM:
    return GeminiLLM(
        model_name="gemini-2.5-flash",
        api_provider=APIProvider(name="google", base_url=None, api_key="dummy"),
        retry_policy=RetryPolicy(api_retries=api_retries, jitter=0.0),
    )


def _response(text: str) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id="resp_retry",
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text=text)]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )


def _fail_then_succeed(
    monkeypatch: pytest.MonkeyPatch,
    llm: GeminiLLM,
    error: Exception,
    *,
    fail_count: int = 1,
) -> dict[str, int]:
    calls = {"n": 0}

    async def fake_generate_content(**_: Any) -> GenerateContentResponse:
        calls["n"] += 1
        if calls["n"] <= fail_count:
            raise error
        return _response("ok")

    monkeypatch.setattr(
        llm.client.aio.models, "generate_content", fake_generate_content
    )
    return calls


@pytest.mark.asyncio
@patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
async def test_aiohttp_timeout_is_retried_and_recovers(
    mock_sleep: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _llm()
    calls = _fail_then_succeed(monkeypatch, llm, TimeoutError())

    response = await llm.generate_response(_USER_MSG)

    assert response.output_text == "ok"
    assert calls["n"] == 2
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
@patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
async def test_aiohttp_timeout_exhausting_retries_raises_typed_error(
    mock_sleep: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from grasp_agents.types.llm_errors import LlmApiTimeoutError

    llm = _llm()
    calls = _fail_then_succeed(monkeypatch, llm, TimeoutError(), fail_count=99)

    with pytest.raises(LlmApiTimeoutError):
        await llm.generate_response(_USER_MSG)

    assert calls["n"] == 2  # initial + one retry
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
@patch("grasp_agents.llm.llm.asyncio.sleep", new_callable=AsyncMock)
async def test_non_sdk_error_still_propagates_unretried(
    mock_sleep: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mapping a timeout must not turn the mapper into a catch-all: an error from
    # our own code is a bug, not a blip, and still skips retry.
    llm = _llm()
    calls = _fail_then_succeed(monkeypatch, llm, ValueError("our bug"), fail_count=99)

    with pytest.raises(ValueError, match="our bug"):
        await llm.generate_response(_USER_MSG)

    assert calls["n"] == 1
    assert mock_sleep.await_count == 0
