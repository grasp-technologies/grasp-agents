"""TypeSafeLLM end to end, against a fake TypeSafe client."""

from typing import Any

import httpx2
import pytest
from pydantic import BaseModel, Field
from typesafe_sdk import Noul, SystemOneResponse

from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.typesafe import TypeSafeLLM, TypeSafeResponse
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.llm_errors import LlmInternalServerError
from grasp_agents.types.llm_events import ResponseCompleted

from ._helpers import jev_response, noul

TREND = "r/interesting: 55 years of inflation."
INPUT_TOKENS = 1_500_000
OUTPUT_TOKENS = 500_000
USD_PER_MILLION_TOKENS = 0.042
SERVICE_UNAVAILABLE = 503


class Learnable(BaseModel):
    """A trending item from Reddit."""

    learnable: bool = Field(description="Could a course be built from this?")


class FakeClient:
    def __init__(self, *responses: SystemOneResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> Any:
        self.calls.append({"state": state, "questions": questions, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_llm(client: FakeClient, **kwargs: Any) -> TypeSafeLLM:
    llm = TypeSafeLLM(
        model_name="jev-latest",
        api_provider=APIProvider(name="typesafe", base_url=None, api_key="test-key"),
        **kwargs,
    )
    object.__setattr__(llm, "client", client)
    return llm


def learnable_response() -> SystemOneResponse:
    return jev_response(
        {"learnable": noul(0.93)},
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )


@pytest.mark.asyncio
async def test_sends_state_and_questions_and_returns_schema_json() -> None:
    client = FakeClient(learnable_response())
    llm = make_llm(client, llm_settings={"instructions": "Judge as a course editor."})

    response = await llm.generate_response(
        [InputMessageItem.from_text(TREND)], output_schema=Learnable
    )

    [call] = client.calls
    assert call["state"] == TREND
    assert call["model"] == "jev-latest"
    assert call["questions"] == {
        "learnable": Noul(
            instructions=(
                "Judge as a course editor.\n\nA trending item from Reddit."
                "\n\nCould a course be built from this?"
            )
        )
    }
    assert Learnable.model_validate_json(response.output_text).learnable is True


@pytest.mark.asyncio
async def test_cost_is_priced_per_million_tokens() -> None:
    llm = make_llm(FakeClient(learnable_response()))

    response = await llm.generate_response(
        [InputMessageItem.from_text(TREND)], output_schema=Learnable
    )

    assert response.usage is not None
    assert response.usage.cost == pytest.approx(
        (INPUT_TOKENS + OUTPUT_TOKENS) / 1_000_000 * USD_PER_MILLION_TOKENS
    )


@pytest.mark.asyncio
async def test_timeout_and_extra_headers_are_forwarded() -> None:
    client = FakeClient(learnable_response())
    llm = make_llm(client, llm_settings={"timeout": 5.0, "temperature": 0.0})

    await llm.generate_response(
        [InputMessageItem.from_text(TREND)],
        output_schema=Learnable,
        extra_headers={"x-run": "demand"},
    )

    call = client.calls[0]
    assert call["timeout"] == pytest.approx(5.0)
    assert call["extra_headers"] == {"x-run": "demand"}
    assert "temperature" not in call


@pytest.mark.asyncio
async def test_missing_output_schema_is_refused_before_sending() -> None:
    client = FakeClient()
    llm = make_llm(client)

    with pytest.raises(ValueError, match="output_schema"):
        await llm.generate_response([InputMessageItem.from_text(TREND)])
    assert client.calls == []


@pytest.mark.asyncio
async def test_tools_are_refused_before_sending() -> None:
    client = FakeClient()
    llm = make_llm(client)
    tools: dict[str, BaseTool[Any, Any, Any]] = {"search": object()}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="tools"):
        await llm.generate_response(
            [InputMessageItem.from_text(TREND)], tools=tools, output_schema=Learnable
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_stream_yields_the_whole_answer_as_one_completed_event() -> None:
    llm = make_llm(FakeClient(learnable_response()))

    events = [
        event
        async for event in llm.generate_response_stream(
            [InputMessageItem.from_text(TREND)], output_schema=Learnable
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], ResponseCompleted)
    assert Learnable.model_validate_json(events[0].response.output_text).learnable
    assert isinstance(events[0].response, TypeSafeResponse)
    assert events[0].response.output_parsed == Learnable(learnable=True)


@pytest.mark.asyncio
async def test_generate_response_returns_the_parsed_schema() -> None:
    llm = make_llm(FakeClient(learnable_response()))

    response = await llm.generate_response(
        [InputMessageItem.from_text(TREND)], output_schema=Learnable
    )

    assert isinstance(response, TypeSafeResponse)
    assert response.output_parsed == Learnable(learnable=True)
    assert response.answers["learnable"].noul == pytest.approx(0.93)


@pytest.mark.asyncio
async def test_sdk_does_not_retry_on_its_own() -> None:
    requests: list[httpx2.Request] = []

    def unavailable(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(SERVICE_UNAVAILABLE)

    llm = TypeSafeLLM(
        model_name="jev-latest",
        api_provider=APIProvider(name="typesafe", base_url=None, api_key="test-key"),
        retry_policy=None,
        extra_typesafe_client_params={"transport": httpx2.MockTransport(unavailable)},
    )

    with pytest.raises(LlmInternalServerError):
        await llm.generate_response(
            [InputMessageItem.from_text(TREND)], output_schema=Learnable
        )
    assert len(requests) == 1


def test_capabilities_need_no_litellm_lookup() -> None:
    llm = make_llm(FakeClient())

    assert llm.capabilities.function_calling is False
    assert llm.capabilities.max_input_tokens == 64_000
