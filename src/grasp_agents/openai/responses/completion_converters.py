from __future__ import annotations

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)

from grasp_agents.typing.completion import Completion, Usage
from grasp_agents.typing.message import AssistantMessage
from grasp_agents.typing.tool import ToolCall

from .message_converters import from_api_assistant_message


def from_response_usage(raw_usage: ResponseUsage) -> Usage:
    return Usage(
        input_tokens=raw_usage.input_tokens,
        output_tokens=raw_usage.output_tokens,
        reasoning_tokens=raw_usage.output_tokens_details.reasoning_tokens,
    )


def completion_from_response(
    raw_completion: OpenAIResponse,
    *,
    name: str | None = None,
) -> Completion:
    message = from_api_assistant_message(raw_completion)
    return Completion(
        id=raw_completion.id,
        model=raw_completion.model,
        created=int(raw_completion.created_at),
        message=message,
        usage=from_response_usage(raw_completion.usage)
        if raw_completion.usage
        else None,
        finish_reason=None,
        name=name,
    )


def to_api_completion(completion: Completion) -> OpenAIResponse:
    raise NotImplementedError
