"""Convert a Jev System One response → internal Response."""

import json

from typesafe_sdk import SystemOneResponse

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.response import Response, ResponseUsage

from .questions import QuestionPlan


def provider_output_to_response(raw: SystemOneResponse, plan: QuestionPlan) -> Response:
    """
    Render the answers as the output schema's JSON, so schema validation and
    output parsing work unchanged; the full answers, with probabilities and
    confidence, stay in ``provider_specific_fields["answers"]``.
    """
    values = plan.read_answers(raw)
    input_tokens = raw.usage.input_tokens or 0
    output_tokens = raw.usage.output_tokens or 0
    return Response(
        model=raw.model,
        status="completed",
        output=[
            OutputMessageItem(
                status="completed",
                content=[OutputMessageText(text=json.dumps(values), annotations=[])],
            )
        ],
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        provider_specific_fields={
            "answers": {
                name: raw.answers[name].model_dump(mode="json")
                for name in plan.questions
            }
        },
    )
