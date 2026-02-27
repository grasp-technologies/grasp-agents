"""Convert OpenAI Responses API wire format → internal Response type."""

from __future__ import annotations

from openai.types.responses import Response as SDKResponse

from ...typing.response import Response


def from_openai_response(raw: SDKResponse) -> Response:
    return Response.model_validate(raw)
