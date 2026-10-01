"""Output schema fields become Jev questions, and Jev answers become field values."""

from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field
from typesafe_sdk import Choice, Noul, Score

from grasp_agents.llm_providers.typesafe.questions import (
    JevChoice,
    JevNoul,
    JevScore,
    build_question_plan,
)

from ._helpers import choice, jev_response, noul, score

COVERAGE_LEVELS = ["not taught", "mentioned", "a module", "the main subject"]


class Area(StrEnum):
    """Which team owns the ticket?"""

    BILLING = "billing"
    BUG = "bug"


class Judgment(BaseModel):
    urgent: bool = Field(description="Does this need a reply within the hour?")
    learnable: Annotated[
        float, JevNoul(true="a course could be built", false="nothing to learn")
    ] = Field(description="Could a course be built from this?")
    area: Literal["billing", "bug"] = Field(description="Which team owns it?")
    team: Area
    top_level: Annotated[
        str, JevChoice({"finance": "money and taxes", "science": "physics, biology"})
    ] = Field(description="Which area is this about?")
    coverage: Annotated[float, JevScore(COVERAGE_LEVELS)] = Field(
        description="How much of the trend does the course teach?"
    )


class TestSchemaToQuestions:
    def test_each_field_becomes_one_question_of_the_right_kind(self) -> None:
        questions = build_question_plan(Judgment).questions

        assert questions["urgent"] == Noul(
            instructions="Does this need a reply within the hour?"
        )
        assert questions["learnable"] == Noul(
            instructions="Could a course be built from this?",
            criteria={"true": "a course could be built", "false": "nothing to learn"},
        )
        assert questions["area"] == Choice(
            instructions="Which team owns it?",
            criteria={"billing": None, "bug": None},
        )
        assert questions["top_level"] == Choice(
            instructions="Which area is this about?",
            criteria={"finance": "money and taxes", "science": "physics, biology"},
        )
        assert questions["coverage"] == Score(
            instructions="How much of the trend does the course teach?",
            criteria=COVERAGE_LEVELS,
        )

    def test_enum_without_description_asks_its_docstring(self) -> None:
        assert build_question_plan(Judgment).questions["team"] == Choice(
            instructions="Which team owns the ticket?",
            criteria={"billing": None, "bug": None},
        )

    def test_framing_is_prefixed_to_every_question(self) -> None:
        class Framed(BaseModel):
            """A trending item from Reddit."""

            urgent: bool = Field(description="Is it urgent?")

        plan = build_question_plan(Framed, instructions="Judge as a course editor.")

        assert plan.questions["urgent"] == Noul(
            instructions=(
                "Judge as a course editor.\n\nA trending item from Reddit.\n\n"
                "Is it urgent?"
            )
        )

    def test_field_without_question_is_refused(self) -> None:
        class NoQuestion(BaseModel):
            urgent: bool

        with pytest.raises(ValueError, match="urgent"):
            build_question_plan(NoQuestion)

    @pytest.mark.parametrize("annotation", [int, str, list[str], float])
    def test_unsupported_field_type_is_refused(self, annotation: type) -> None:
        schema = type(
            "Unsupported",
            (BaseModel,),
            {
                "__annotations__": {"value": annotation},
                "value": Field(description="What is it?"),
            },
        )
        with pytest.raises(ValueError, match="value"):
            build_question_plan(schema)

    def test_non_model_schema_is_refused(self) -> None:
        with pytest.raises(TypeError, match="BaseModel"):
            build_question_plan(bool)


class TestAnswersToValues:
    def test_answers_are_read_into_field_values(self) -> None:
        plan = build_question_plan(Judgment)
        response = jev_response(
            {
                "urgent": noul(0.49),
                "learnable": noul(0.93),
                "area": choice("bug", {"billing": 0.2, "bug": 0.8}),
                "team": choice("billing", {"billing": 0.9, "bug": 0.1}),
                "top_level": choice("science", {"finance": 0.3, "science": 0.7}),
                "coverage": score(2.4, COVERAGE_LEVELS, [0.0, 0.1, 0.4, 0.5]),
            }
        )

        values = plan.read_answers(response)

        assert values == {
            "urgent": False,
            "learnable": 0.93,
            "area": "bug",
            "team": "billing",
            "top_level": "science",
            "coverage": 2.4,
        }
        assert Judgment.model_validate(values).team is Area.BILLING

    def test_missing_answer_is_an_error(self) -> None:
        plan = build_question_plan(Judgment)
        with pytest.raises(ValueError, match="coverage"):
            plan.read_answers(jev_response({"urgent": noul(0.9)}))
