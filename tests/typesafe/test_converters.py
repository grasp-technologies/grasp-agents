"""Input items become Jev's state; Jev's answers become a grasp-agents Response."""

import json

import pytest
from pydantic import BaseModel, Field

from grasp_agents.llm_providers.typesafe.provider_output_to_response import (
    provider_output_to_response,
)
from grasp_agents.llm_providers.typesafe.questions import build_question_plan
from grasp_agents.llm_providers.typesafe.response_to_provider_inputs import (
    items_to_state,
)
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import InputMessageItem, OutputMessageItem

from ._helpers import jev_response, noul

TREND = "r/interesting: 55 years of inflation."


class Learnable(BaseModel):
    learnable: bool = Field(description="Could a course be built from this?")


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
        assert response.provider_specific_fields == {
            "answers": {"learnable": {"type": "noul", "noul": 0.93}}
        }
        assert response.usage is not None
        assert response.usage.input_tokens == 120
        assert response.usage.output_tokens == 12
        assert response.usage.total_tokens == 132
