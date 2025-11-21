from __future__ import annotations

from typing import Any, Iterable
import json
from pydantic.json import pydantic_encoder

from ...typing.completion import Completion, Usage
from ...typing.message import AssistantMessage
from ...typing.tool import ToolCall
from .. import (
    OpenAIParsedResponse,
    OpenAIResponse,
)

from .. import (
    OpenAIParsedResponse,
    OpenAIResponse,
    OpenAIResponseStreamEvent,
)


def usage_from_response(raw_usage: Any) -> Usage:
    """Map a provider usage object to our Usage.

    Works for both ParsedResponse.usage and Response.usage.
    """
    return Usage(
        input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
        reasoning_tokens=getattr(raw_usage, "reasoning_tokens", None),
        cached_tokens=None,
    )


def completion_from_response(
    raw_completion: OpenAIParsedResponse[Any] | OpenAIResponse,
    *,
    name: str | None = None,
) -> Completion:
    content: str | None = None
    if isinstance(raw_completion, OpenAIParsedResponse):
        parsed = getattr(raw_completion, "output_parsed", None)
        if parsed is not None:
            content = json.dumps(parsed, default=pydantic_encoder)

    if content is None:
        parts: list[str] = []
        for out in getattr(raw_completion, "output", []) or []:
            if getattr(out, "type", "") == "message":
                for c in getattr(out, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        txt = getattr(c, "text", None)
                        if txt:
                            parts.append(txt)
        content = "\n".join(parts) if parts else "<empty>"

    usage_obj = getattr(raw_completion, "usage", None)
    usage = usage_from_response(usage_obj) if usage_obj else None

    model_name = str(getattr(raw_completion, "model", None))

    response_id = raw_completion.id
    msg = AssistantMessage(content=content, response_id=response_id)
    tool_calls: list[ToolCall] = []
    for out in getattr(raw_completion, "output", []) or []:
        if getattr(out, "type", "") == "function_call":
            # Prefer call_id to ensure correct reply via function_call_output
            tool_calls.append(
                ToolCall(
                    id=getattr(out, "call_id", None) or getattr(out, "id", None),
                    tool_name=getattr(out, "name", None),
                    tool_arguments=getattr(out, "arguments", None),
                )
            )
    if tool_calls:
        msg.tool_calls = tool_calls

    return Completion(
        model=model_name, name=name, message=msg, finish_reason=None, usage=usage
    )
