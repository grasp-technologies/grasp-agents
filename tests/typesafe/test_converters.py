"""Input items become Jev's state; Jev's answers become a grasp-agents Response."""

import json
from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from grasp_agents.llm_providers.typesafe import (
    JevChoice,
    JevNoul,
    JevScore,
    TypeSafeResponse,
)
from grasp_agents.llm_providers.typesafe.provider_output_to_response import (
    provider_output_to_response,
)
from grasp_agents.llm_providers.typesafe.questions import build_question_plan
from grasp_agents.llm_providers.typesafe.response_to_provider_inputs import (
    items_to_state,
)
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import InputMessageItem, OutputMessageItem

from ._helpers import choice, jev_response, noul, score

TREND = "r/interesting: 55 years of inflation."


class Learnable(BaseModel):
    learnable: bool = Field(description="Could a course be built from this?")


ANGER_LEVELS = ["calm", "annoyed", "furious"]


class Ticket(BaseModel):
    """A support ticket."""

    urgent: bool = Field(description="Does this need a reply within the hour?")
    area: Literal["billing", "bug"] = Field(description="Which team owns it?")
    churn_risk: Annotated[float, JevNoul()] = Field(description="Will they cancel?")
    product: Annotated[
        str, JevChoice({"app": "the mobile app", "web": "the website"})
    ] = Field(description="Which product is it about?")
    anger: Annotated[float, JevScore(ANGER_LEVELS)] = Field(
        description="How angry is the customer?"
    )


class Area(StrEnum):
    """Which team owns it?"""

    BILLING = "billing"
    BUG = "bug"


class StrictTicket(BaseModel):
    model_config = ConfigDict(strict=True)

    urgent: bool = Field(description="Does this need a reply within the hour?")
    area: Area


TICKET_ANSWERS = {
    "urgent": noul(0.91),
    "area": choice("billing", {"billing": 0.8, "bug": 0.2}),
    "churn_risk": noul(0.35),
    "product": choice("app", {"app": 0.7, "web": 0.3}),
    "anger": score(1.6, ANGER_LEVELS, [0.1, 0.2, 0.7]),
}


class TestItemsToState:
    def test_user_text_is_the_state(self) -> None:
        state, framing = items_to_state([InputMessageItem.from_text(TREND)])

        assert state == TREND
        assert framing is None

    def test_system_and_developer_messages_become_framing(self) -> None:
        state, framing = items_to_state(
            [
                InputMessageItem.from_text("You judge trends.", role="system"),
                InputMessageItem.from_text("For a course catalogue.", role="developer"),
                InputMessageItem.from_text(TREND),
            ]
        )

        assert state == TREND
        assert framing == "You judge trends.\n\nFor a course catalogue."

    def test_several_text_parts_are_joined(self) -> None:
        item = InputMessageItem(
            role="user", content=[InputText(text="first"), InputText(text="second")]
        )
        assert items_to_state([item]) == ("first\n\nsecond", None)

    def test_more_than_one_user_message_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one user message"):
            items_to_state(
                [
                    InputMessageItem.from_text("a"),
                    InputMessageItem.from_text("b"),
                ]
            )

    def test_no_user_message_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one user message"):
            items_to_state([InputMessageItem.from_text("only framing", role="system")])

    def test_history_is_refused(self) -> None:
        with pytest.raises(TypeError, match="OutputMessageItem"):
            items_to_state(
                [
                    InputMessageItem.from_text(TREND),
                    OutputMessageItem(status="completed"),
                ]
            )

    def test_images_are_refused(self) -> None:
        item = InputMessageItem(
            role="user",
            content=[InputText(text=TREND), InputImage(image_url="https://x/y.png")],
        )
        with pytest.raises(ValueError, match="text"):
            items_to_state([item])


class TestProviderOutputToResponse:
    def test_answers_become_json_output_and_full_detail(self) -> None:
        plan = build_question_plan(Learnable)
        raw = jev_response(
            {"learnable": noul(0.93)},
            model="jev-1.13.0",
            input_tokens=120,
            output_tokens=12,
        )

        response = provider_output_to_response(raw, plan)

        assert json.loads(response.output_text) == {"learnable": True}
        assert Learnable.model_validate_json(response.output_text).learnable is True
        assert response.model == "jev-1.13.0"
        assert response.status == "completed"
        assert response.usage is not None
        assert response.usage.input_tokens == 120
        assert response.usage.output_tokens == 12
        assert response.usage.total_tokens == 132

    def test_response_carries_the_parsed_schema_and_typed_answers(self) -> None:
        plan = build_question_plan(Learnable)
        raw = jev_response({"learnable": noul(0.93)})

        response = provider_output_to_response(raw, plan)

        assert isinstance(response, TypeSafeResponse)
        assert response.output_parsed == Learnable(learnable=True)
        assert response.answers == raw.answers

    def test_every_question_kind_matches_the_json_text(self) -> None:
        plan = build_question_plan(Ticket)

        response = provider_output_to_response(jev_response(TICKET_ANSWERS), plan)

        assert response.output_parsed == Ticket.model_validate_json(
            response.output_text
        )
        assert response.output_parsed == Ticket(
            urgent=True, area="billing", churn_risk=0.35, product="app", anger=1.6
        )

    def test_strict_schema_with_str_enum_is_built(self) -> None:
        plan = build_question_plan(StrictTicket)
        raw = jev_response(
            {
                "urgent": noul(0.91),
                "area": choice("billing", {"billing": 0.8, "bug": 0.2}),
            }
        )

        response = provider_output_to_response(raw, plan)

        assert response.output_parsed == StrictTicket(urgent=True, area=Area.BILLING)
        assert response.output_parsed == StrictTicket.model_validate_json(
            response.output_text
        )

    def test_dump_keeps_answers_and_leaves_out_the_parsed_object(self) -> None:
        plan = build_question_plan(Learnable)
        response = provider_output_to_response(
            jev_response({"learnable": noul(0.93)}), plan
        )

        dumped = response.model_dump(mode="json")

        assert "output_parsed" not in dumped
        assert dumped["answers"] == {"learnable": {"type": "noul", "noul": 0.93}}
