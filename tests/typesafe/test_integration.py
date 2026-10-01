"""TypeSafeLLM against the real TypeSafe API (needs TYPESAFE_API_KEY)."""

from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, Field

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers.typesafe import (
    JevChoice,
    JevNoul,
    JevScore,
    TypeSafeLLM,
    TypeSafeResponse,
)
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.llm_errors import LlmAuthenticationError, LlmContextWindowError
from grasp_agents.types.llm_events import ResponseCompleted

MODEL = "jev-latest"
TICKET = (
    "You charged my card twice for the same subscription and my account is now"
    " overdrawn. I need this reversed today or I am cancelling."
)
ANGER_LEVELS = ["calm", "annoyed", "furious"]
# Comfortably past jev-1.13's 64k-token limit for state and questions together.
OVER_LIMIT_WORDS = 90_000


class Ticket(BaseModel):
    """A customer support ticket."""

    urgent: bool = Field(description="Does this need a reply today?")
    area: Literal["billing", "bug", "account"] = Field(
        description="Which team owns it?"
    )
    churn_risk: Annotated[float, JevNoul()] = Field(
        description="Is the customer likely to cancel?"
    )
    product: Annotated[
        str, JevChoice({"subscription": "a recurring plan", "hardware": "a device"})
    ] = Field(description="What did they buy?")
    anger: Annotated[float, JevScore(ANGER_LEVELS)] = Field(
        description="How angry is the customer?"
    )


def make_llm(api_key: str) -> TypeSafeLLM:
    return TypeSafeLLM(
        model_name=MODEL,
        api_provider=APIProvider(name="typesafe", base_url=None, api_key=api_key),
    )


@pytest.mark.integration
class TestTypeSafeIntegration:
    @pytest.mark.asyncio
    async def test_every_question_kind_is_answered(self, typesafe_api_key: str) -> None:
        response = cast(
            "TypeSafeResponse",
            await make_llm(typesafe_api_key).generate_response(
                [InputMessageItem.from_text(TICKET)], output_schema=Ticket
            ),
        )

        ticket = response.output_parsed
        assert isinstance(ticket, Ticket)
        assert Ticket.model_validate_json(response.output_text) == ticket
        assert ticket.urgent is True
        assert ticket.area == "billing"
        assert ticket.product == "subscription"
        assert 0.0 <= ticket.churn_risk <= 1.0
        assert 0.0 <= ticket.anger <= len(ANGER_LEVELS) - 1
        assert set(response.answers) == set(Ticket.model_fields)
        assert response.model.startswith("jev")
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.cost is not None
        assert response.usage.cost > 0

    @pytest.mark.asyncio
    async def test_stream_returns_one_completed_event(
        self, typesafe_api_key: str
    ) -> None:
        events = [
            event
            async for event in make_llm(typesafe_api_key).generate_response_stream(
                [InputMessageItem.from_text(TICKET)], output_schema=Ticket
            )
        ]

        assert len(events) == 1
        assert isinstance(events[0], ResponseCompleted)
        assert isinstance(events[0].response, TypeSafeResponse)
        assert isinstance(events[0].response.output_parsed, Ticket)

    @pytest.mark.asyncio
    async def test_llm_agent_returns_the_schema(self, typesafe_api_key: str) -> None:
        agent = LLMAgent[str, Ticket, None](
            name="triage", llm=make_llm(typesafe_api_key), env_info=False
        )

        packet = await agent.run(TICKET)

        [ticket] = packet.payloads
        assert isinstance(ticket, Ticket)
        assert ticket.area == "billing"

    @pytest.mark.asyncio
    async def test_wrong_key_is_an_authentication_error(self) -> None:
        llm = TypeSafeLLM(
            model_name=MODEL,
            api_provider=APIProvider(
                name="typesafe", base_url=None, api_key="not-a-real-key"
            ),
            retry_policy=None,
        )

        with pytest.raises(LlmAuthenticationError):
            await llm.generate_response(
                [InputMessageItem.from_text(TICKET)], output_schema=Ticket
            )

    @pytest.mark.asyncio
    async def test_state_over_the_token_limit_is_a_context_window_error(
        self, typesafe_api_key: str
    ) -> None:
        llm = TypeSafeLLM(
            model_name=MODEL,
            api_provider=APIProvider(
                name="typesafe", base_url=None, api_key=typesafe_api_key
            ),
            retry_policy=None,
        )
        too_long = " ".join(["word"] * OVER_LIMIT_WORDS)

        with pytest.raises(LlmContextWindowError):
            await llm.generate_response(
                [InputMessageItem.from_text(too_long)], output_schema=Ticket
            )
