from __future__ import annotations

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)

from ...typing.completion import Completion, Usage
from ...typing.message import AssistantMessage
from ...typing.tool import ToolCall


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
    outputs = raw_completion.output
    content: list[str] = []
    refusal: str | None = None
    reasoning_summary: list[str] = []
    tool_calls: list[ToolCall] = []
    for output in outputs:
        if isinstance(output, ResponseOutputMessage):
            raw_contents = output.content
            # TODO: add refusal and annotation convertion
            for raw_content in raw_contents:
                if isinstance(raw_content, ResponseOutputText):
                    content.append(raw_content.text)
                else:
                    refusal = raw_content.refusal
        elif isinstance(output, ResponseReasoningItem):
            raw_summaries = output.summary
            for raw_summary in raw_summaries:
                reasoning_summary.append(raw_summary.text)
        elif isinstance(output, ResponseFunctionToolCall):
            tool = ToolCall(
                id=output.call_id,
                tool_arguments=output.arguments,
                tool_name=output.name,
            )
            tool_calls.append(tool)
    message = AssistantMessage(
        content=" ".join(content),
        reasoning_content=" ".join(reasoning_summary),
        tool_calls=tool_calls,
        refusal=refusal,
        response_id=raw_completion.id,
        thinking_blocks=[
            {"type": "thinking", "thinking": item} for item in reasoning_summary
        ],
    )
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
