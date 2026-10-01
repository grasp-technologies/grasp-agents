"""The Response TypeSafeLLM returns: the usual fields plus the typed judgment."""

from pydantic import BaseModel, Field
from typesafe_sdk import Answer

from grasp_agents.types.response import Response


class TypeSafeResponse(Response):
    output_parsed: BaseModel | None = Field(default=None, exclude=True)
    answers: dict[str, Answer] = Field(default_factory=dict[str, Answer])
