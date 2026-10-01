"""Convert a Jev System One response → TypeSafeResponse."""

import json

from typesafe_sdk import SystemOneResponse

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.response import ResponseUsage

from .questions import QuestionPlan
from .typesafe_response import TypeSafeResponse


def provider_output_to_response(
    raw: SystemOneResponse, plan: QuestionPlan
) -> TypeSafeResponse:
    """
    The answers become the output schema twice: as JSON text, so schema
    validation and output parsing work as for any LLM, and as the built
    object in ``output_parsed``. Both come from the same JSON text, so a
    schema that passes one passes the other.
    """
    text = json.dumps(plan.read_answers(raw))
    input_tokens = raw.usage.input_tokens or 0
    output_tokens = raw.usage.output_tokens or 0
    return TypeSafeResponse(
        model=raw.model,
        status="completed",
        output=[
            OutputMessageItem(
                status="completed",
                content=[OutputMessageText(text=text, annotations=[])],
            )
        ],
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        output_parsed=plan.schema.model_validate_json(text),
        answers={name: raw.answers[name] for name in plan.questions},
    )
