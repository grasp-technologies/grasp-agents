"""
Output schema → Jev questions, and Jev answers → field values.

Each field of a pydantic output model is one question; its description is the
question text. The field type picks the question kind:

* ``bool`` → Noul, answered ``True`` when the probability is at least 0.5
* ``Annotated[float, JevNoul(...)]`` → Noul, answered with the probability
* ``Literal[...]`` of strings or a ``StrEnum`` → Choice among its values
* ``Annotated[str, JevChoice({...})]`` → Choice among described options
* ``Annotated[float, JevScore([...])]`` → Score, answered with the expected level
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)

NOUL_TRUE_THRESHOLD = 0.5

JevQuestion = Noul | Choice | Score


@dataclass(frozen=True)
class JevNoul:
    """A yes/no question answered with its probability; may describe both outcomes."""

    true: str | None = None
    false: str | None = None


@dataclass(frozen=True)
class JevChoice:
    """A pick-one question over options that each carry a description."""

    options: Mapping[str, str | None]


@dataclass(frozen=True)
class JevScore:
    """A question answered on ordered levels, lowest first."""

    levels: Sequence[str]


AnswerReader = Callable[[Answer], Any]


@dataclass(frozen=True)
class QuestionPlan:
    schema: type[BaseModel]
    questions: dict[str, JevQuestion]
    readers: dict[str, AnswerReader]

    def read_answers(self, response: SystemOneResponse) -> dict[str, Any]:
        missing = sorted(set(self.questions) - set(response.answers))
        if missing:
            raise ValueError(f"Jev returned no answer for {', '.join(missing)}")
        return {
            name: read(response.answers[name]) for name, read in self.readers.items()
        }


def build_question_plan(schema: Any, instructions: str | None = None) -> QuestionPlan:
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"TypeSafeLLM needs a pydantic BaseModel output schema, got {schema!r}"
        )

    framing = _join(instructions, _own_docstring(schema))
    questions: dict[str, JevQuestion] = {}
    readers: dict[str, AnswerReader] = {}
    for name, field in schema.model_fields.items():
        questions[name], readers[name] = _field_question(name, field, framing)
    return QuestionPlan(schema=schema, questions=questions, readers=readers)


def _field_question(
    name: str, field: FieldInfo, framing: str
) -> tuple[JevQuestion, AnswerReader]:
    annotation = field.annotation
    marker = next(
        (m for m in field.metadata if isinstance(m, JevNoul | JevChoice | JevScore)),
        None,
    )
    enum_docstring = (
        _own_docstring(annotation)
        if isinstance(annotation, type) and issubclass(annotation, StrEnum)
        else None
    )
    question_text = field.description or enum_docstring
    if not question_text:
        raise ValueError(
            f"Field {name!r} has no question: give it a description"
            " (or a docstring on its enum)"
        )
    text = _join(framing, question_text)

    if isinstance(marker, JevScore) and annotation is float:
        return Score(instructions=text, criteria=list(marker.levels)), _read_score
    if isinstance(marker, JevChoice) and annotation is str:
        return Choice(instructions=text, criteria=dict(marker.options)), _read_choice
    if isinstance(marker, JevNoul) and annotation is float:
        if marker.true is None and marker.false is None:
            return Noul(instructions=text), _read_probability
        criteria = NoulCriteria(true=marker.true, false=marker.false)
        return Noul(instructions=text, criteria=criteria), _read_probability
    if marker is None and annotation is bool:
        return Noul(instructions=text), _read_bool
    if marker is None and (options := _string_options(annotation)):
        return Choice(instructions=text, criteria=dict.fromkeys(options)), _read_choice

    raise ValueError(
        f"Field {name!r} of type {annotation!r} cannot be asked to Jev: use bool,"
        " a Literal or StrEnum of strings, or annotate float with JevNoul/JevScore"
        " and str with JevChoice"
    )


def _string_options(annotation: Any) -> list[str] | None:
    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        if all(isinstance(arg, str) for arg in args):
            return list(args)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return [member.value for member in annotation]
    return None


def _read_bool(answer: Answer) -> bool:
    return _expect(answer, NoulAnswer).noul >= NOUL_TRUE_THRESHOLD


def _read_probability(answer: Answer) -> float:
    return _expect(answer, NoulAnswer).noul


def _read_choice(answer: Answer) -> str:
    return _expect(answer, ChoiceAnswer).choice


def _read_score(answer: Answer) -> float:
    return _expect(answer, ScoreAnswer).score


def _expect[AnswerT](answer: Answer, answer_type: type[AnswerT]) -> AnswerT:
    if not isinstance(answer, answer_type):
        raise TypeError(
            f"Expected a {answer_type.__name__} from Jev, got {type(answer).__name__}"
        )
    return answer


def _own_docstring(annotation: Any) -> str | None:
    if not isinstance(annotation, type):
        return None
    doc = annotation.__dict__.get("__doc__")
    return inspect.cleandoc(doc) if doc else None


def _join(*parts: str | None) -> str:
    return "\n\n".join(part for part in parts if part)
