"""
OpenAI Responses API – Converters (Reference Implementation)

This module shows, end‑to‑end, how to implement a full Converters subclass
for the OpenAI Responses API pathway. It is intentionally verbose with
explanatory comments so you can see exactly how each piece maps.

Key ideas:

- Converters translate between our internal types (messages, tools, content,
  completion, streaming chunks) and provider‑specific types.
- For Responses, we can (and should) reuse most low‑level helpers that already
  work for Chat/LiteLLM (content parts, basic message shapes, tool schemas),
  and only add thin adapters where the Responses API differs.
- Streaming for Responses is an event bus. We convert only delta events we care
  about into our internal CompletionChunk and let the shared
  postprocess_event_stream layer handle start/end boundaries uniformly.

This reference is standalone and does not change current wiring. You can switch
OpenAIResponsesLLM.converters to this class once you’re ready.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Generic, TypeVar, overload

from pydantic.json import pydantic_encoder

from ...typing.completion import Completion, Usage
from ...typing.completion_chunk import CompletionChunk
from ...typing.content import Content
from ...typing.converters import Converters
from ...typing.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ...typing.tool import BaseTool, NamedToolChoice, ToolChoice

# Responses API typed params and objects (imported via our OpenAI facade)
from .. import (
    OpenAIParsedResponse,  # Final object when using .parse(...)
    OpenAIResponse,  # Final object (unparsed)
    OpenAIResponsesInputParam,  # Union[str, ResponseInputParam]
)

# Reuse the battle‑tested low‑level helpers used by Chat/LiteLLM
from ..content_converters import from_api_content
from ..message_converters import (
    from_api_assistant_message,
    from_api_system_message,
    from_api_tool_message,
    from_api_user_message,
    to_api_assistant_message,
    to_api_system_message,
    to_api_user_message,
)
from .responses_chunk_converters import (
    from_api_response_stream_event,
    is_supported_stream_event,
)
from .responses_completion_converters import (
    completion_from_response,
)
from .responses_completion_converters import (
    usage_from_response as _usage_from_response,
)
from .responses_content_converters import to_responses_content
from .responses_tool_converters import (
    to_responses_function_tool,
    to_responses_tool_choice,
)


class OpenAIResponsesConverters(Converters):
    @staticmethod
    def to_system_message(system_message: SystemMessage, **kwargs: Any) -> Any:
        # Responses API input message (system)
        return {
            "type": "message",
            "role": "system",
            "content": to_responses_content(system_message.content),
        }

    @staticmethod
    def to_user_message(user_message: UserMessage, **kwargs: Any) -> Any:
        # Responses API input message (user)
        return {
            "type": "message",
            "role": "user",
            "content": to_responses_content(user_message.content),
        }

    @staticmethod
    def to_assistant_message(
        assistant_message: AssistantMessage, **kwargs: Any
    ) -> Any:
        # Responses API input message (assistant). Use EasyInputMessageParam shape.
        content = (
            assistant_message.content if assistant_message.content is not None else "<empty>"
        )
        return {
            "type": "message",
            "role": "assistant",
            "content": to_responses_content(content),
        }

    @staticmethod
    def from_system_message(raw_message: Any, **kwargs: Any) -> SystemMessage:
        return from_api_system_message(raw_message, **kwargs)

    @staticmethod
    def from_user_message(raw_message: Any, **kwargs: Any) -> UserMessage:
        return from_api_user_message(raw_message, **kwargs)

    @staticmethod
    def from_assistant_message(raw_message: Any, **kwargs: Any) -> AssistantMessage:
        return from_api_assistant_message(raw_message, **kwargs)

    @staticmethod
    def to_tool(tool: BaseTool, strict: bool | None = None, **kwargs: Any) -> Any:
        return to_responses_function_tool(tool=tool, strict=strict)

    @staticmethod
    def to_tool_choice(tool_choice: ToolChoice, **kwargs: Any) -> Any:
        return to_responses_tool_choice(tool_choice=tool_choice)

    @staticmethod
    def to_tool_message(tool_message: ToolMessage, **kwargs: Any) -> Any:
        return {
            "type": "function_call_output",
            "call_id": tool_message.tool_call_id,
            "output": tool_message.content,
        }

    @staticmethod
    def from_tool_message(raw_message: Any, **kwargs: Any) -> ToolMessage:
        return from_api_tool_message(raw_message, **kwargs)

    @staticmethod
    def to_content(content: Content, **kwargs: Any) -> Any:
        # For Responses API, return a ResponseInputMessageContentListParam
        return to_responses_content(content)

    @staticmethod
    def from_content(raw_content: Any, **kwargs: Any) -> Content:
        return from_api_content(raw_content, **kwargs)

    @staticmethod
    def to_completion(completion: Completion, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def from_completion(
        raw_completion: OpenAIParsedResponse[Any] | OpenAIResponse,
        name: str | None = None,
        **kwargs: Any,
    ) -> Completion:
        return completion_from_response(raw_completion, name=name)

    @staticmethod
    def from_completion_usage(raw_usage: Any, **kwargs: Any) -> Usage:
        return _usage_from_response(raw_usage)

    @staticmethod
    def to_completion_chunk(chunk: CompletionChunk, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def from_completion_chunk(
        raw_chunk: Any, name: str | None = None, **kwargs: Any
    ) -> CompletionChunk:
        if is_supported_stream_event(raw_chunk):
            chunk = from_api_response_stream_event(raw_chunk, name=name)
            if chunk is not None:
                return chunk
        raise TypeError(
            f"Unsupported streaming item for Responses conversion: {type(raw_chunk)!r}"
        )
