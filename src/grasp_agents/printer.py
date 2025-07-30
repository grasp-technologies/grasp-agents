import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel
from termcolor import colored
from termcolor._types import Color

from grasp_agents.typing.completion_chunk import CompletionChunk
from grasp_agents.typing.events import (
    AnnotationsChunkEvent,
    AnnotationsEndEvent,
    AnnotationsStartEvent,
    CompletionChunkEvent,
    # CompletionEndEvent,
    CompletionStartEvent,
    Event,
    GenMessageEvent,
    MessageEvent,
    ProcPacketOutputEvent,
    ResponseChunkEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    RunResultEvent,
    SystemMessageEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolMessageEvent,
    UserMessageEvent,
    WorkflowResultEvent,
)

from .typing.completion import Usage
from .typing.content import Content, ContentPartText
from .typing.message import (
    AssistantMessage,
    Message,
    Role,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)


ROLE_TO_COLOR: Mapping[Role, Color] = {
    Role.SYSTEM: "magenta",
    Role.USER: "green",
    Role.ASSISTANT: "light_blue",
    Role.TOOL: "blue",
}

AVAILABLE_COLORS: list[Color] = [
    "magenta",
    "green",
    "light_blue",
    "light_cyan",
    "yellow",
    "blue",
    "red",
]

ColoringMode: TypeAlias = Literal["agent", "role"]
CompletionBlockType: TypeAlias = Literal["response", "thinking", "tool_call"]


class Printer:
    def __init__(
        self, color_by: ColoringMode = "role", msg_trunc_len: int = 20000
    ) -> None:
        self.color_by = color_by
        self.msg_trunc_len = msg_trunc_len
        self._current_message: str = ""

    @staticmethod
    def get_role_color(role: Role) -> Color:
        return ROLE_TO_COLOR[role]

    @staticmethod
    def get_agent_color(agent_name: str) -> Color:
        idx = int(
            hashlib.md5(agent_name.encode()).hexdigest(),  # noqa :S324
            16,
        ) % len(AVAILABLE_COLORS)

        return AVAILABLE_COLORS[idx]

    @staticmethod
    def content_to_str(content: Content | str | None, role: Role) -> str:
        if role == Role.USER and isinstance(content, Content):
            content_str_parts: list[str] = []
            for content_part in content.parts:
                if isinstance(content_part, ContentPartText):
                    content_str_parts.append(content_part.data.strip(" \n"))
                elif content_part.data.type == "url":
                    content_str_parts.append(str(content_part.data.url))
                elif content_part.data.type == "base64":
                    content_str_parts.append("<ENCODED_IMAGE>")
            return "\n".join(content_str_parts)

        assert isinstance(content, str | None)

        return (content or "").strip(" \n")

    @staticmethod
    def truncate_content_str(content_str: str, trunc_len: int = 2000) -> str:
        if len(content_str) > trunc_len:
            return content_str[:trunc_len] + "[...]"

        return content_str

    def print_message(
        self,
        message: Message,
        agent_name: str,
        call_id: str,
        usage: Usage | None = None,
    ) -> None:
        if usage is not None and not isinstance(message, AssistantMessage):
            raise ValueError(
                "Usage information can only be printed for AssistantMessage"
            )

        color = (
            self.get_agent_color(agent_name)
            if self.color_by == "agent"
            else self.get_role_color(message.role)
        )
        log_kwargs = {"extra": {"color": color}}

        out = f"<{agent_name}> [{call_id}]\n"

        # Thinking
        if isinstance(message, AssistantMessage) and message.reasoning_content:
            thinking = message.reasoning_content.strip(" \n")
            out += f"<thinking>\n{thinking}\n</thinking>\n"

        # Content
        content = self.content_to_str(message.content or "", message.role)
        if content:
            try:
                content = json.dumps(json.loads(content), indent=2)
            except Exception:
                pass
            content = self.truncate_content_str(content, trunc_len=self.msg_trunc_len)
            if isinstance(message, SystemMessage):
                out += f"<system>\n{content}\n</system>\n"
            elif isinstance(message, UserMessage):
                out += f"<input>\n{content}\n</input>\n"
            elif isinstance(message, AssistantMessage):
                out += f"<response>\n{content}\n</response>\n"
            else:
                out += f"<tool result> [{message.tool_call_id}]\n{content}\n</tool result>\n"

        # Tool calls
        if isinstance(message, AssistantMessage) and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                out += (
                    f"<tool call> {tool_call.tool_name} [{tool_call.id}]\n"
                    f"{tool_call.tool_arguments}\n</tool call>\n"
                )

        # Usage
        if usage is not None:
            usage_str = f"I/O/R/C tokens: {usage.input_tokens}/{usage.output_tokens}"
            usage_str += f"/{usage.reasoning_tokens or '-'}"
            usage_str += f"/{usage.cached_tokens or '-'}"

            out += f"\n------------------------------------\n{usage_str}\n"

        logger.debug(out, **log_kwargs)  # type: ignore

    def print_messages(
        self,
        messages: Sequence[Message],
        agent_name: str,
        call_id: str,
        usages: Sequence[Usage | None] | None = None,
    ) -> None:
        _usages: Sequence[Usage | None] = usages or [None] * len(messages)

        for _message, _usage in zip(messages, _usages, strict=False):
            self.print_message(
                _message, usage=_usage, agent_name=agent_name, call_id=call_id
            )


def stream_text(new_text: str, color: Color) -> None:
    sys.stdout.write(colored(new_text, color))
    sys.stdout.flush()


async def print_event_stream(
    event_generator: AsyncIterator[Event[Any]],
    color_by: ColoringMode = "role",
    trunc_len: int = 10000,
) -> AsyncIterator[Event[Any]]:
    color = Printer.get_role_color(Role.ASSISTANT)

    def _get_color(event: Event[Any], role: Role = Role.ASSISTANT) -> Color:
        if color_by == "agent":
            return Printer.get_agent_color(event.proc_name or "")
        return Printer.get_role_color(role)

    def _print_packet(
        event: ProcPacketOutputEvent | WorkflowResultEvent | RunResultEvent,
    ) -> None:
        color = _get_color(event, Role.ASSISTANT)

        if isinstance(event, WorkflowResultEvent):
            src = "workflow"
        elif isinstance(event, RunResultEvent):
            src = "run"
        else:
            src = "processor"

        text = f"\n<{event.proc_name}> [{event.call_id}]\n"

        if event.data.payloads:
            text += f"<{src} output>\n"
            for p in event.data.payloads:
                if isinstance(p, BaseModel):
                    p_str = p.model_dump_json(indent=2)
                else:
                    try:
                        p_str = json.dumps(p, indent=2)
                    except TypeError:
                        p_str = str(p)
                text += f"{p_str}\n"
            text += f"</{src} output>\n"

        stream_text(text, color)

    async for event in event_generator:
        yield event

        if isinstance(event, CompletionChunkEvent) and isinstance(
            event.data, CompletionChunk
        ):
            color = _get_color(event, Role.ASSISTANT)

            text = ""

            if isinstance(event, CompletionStartEvent):
                text += f"\n<{event.proc_name}> [{event.call_id}]\n"
            elif isinstance(event, ThinkingStartEvent):
                text += "<thinking>\n"
            elif isinstance(event, ResponseStartEvent):
                text += "<response>\n"
            elif isinstance(event, ToolCallStartEvent):
                tc = event.data.tool_call
                text += f"<tool call> {tc.tool_name} [{tc.id}]\n"
            elif isinstance(event, AnnotationsStartEvent):
                text += "<annotations>\n"

            # if isinstance(event, CompletionEndEvent):
            #     text += f"\n</{event.proc_name}>\n"
            if isinstance(event, ThinkingEndEvent):
                text += "\n</thinking>\n"
            elif isinstance(event, ResponseEndEvent):
                text += "\n</response>\n"
            elif isinstance(event, ToolCallEndEvent):
                text += "\n</tool call>\n"
            elif isinstance(event, AnnotationsEndEvent):
                text += "\n</annotations>\n"

            if isinstance(event, ThinkingChunkEvent):
                thinking = event.data.thinking
                if isinstance(thinking, str):
                    text += thinking
                else:
                    text = "\n".join(
                        [block.get("thinking", "[redacted]") for block in thinking]
                    )

            if isinstance(event, ResponseChunkEvent):
                text += event.data.response

            if isinstance(event, ToolCallChunkEvent):
                text += event.data.tool_call.tool_arguments or ""

            if isinstance(event, AnnotationsChunkEvent):
                text += "\n".join(
                    [
                        json.dumps(annotation, indent=2)
                        for annotation in event.data.annotations
                    ]
                )

            stream_text(text, color)

        if isinstance(event, MessageEvent) and not isinstance(event, GenMessageEvent):
            assert isinstance(event.data, (SystemMessage | UserMessage | ToolMessage))

            message = event.data
            role = message.role
            content = Printer.content_to_str(message.content, role=role)
            color = _get_color(event, role)

            text = f"\n<{event.proc_name}> [{event.call_id}]\n"

            if isinstance(event, (SystemMessageEvent, UserMessageEvent)):
                content = Printer.truncate_content_str(content, trunc_len=trunc_len)

            if isinstance(event, SystemMessageEvent):
                text += f"<system>\n{content}\n</system>\n"

            elif isinstance(event, UserMessageEvent):
                text += f"<input>\n{content}\n</input>\n"

            elif isinstance(event, ToolMessageEvent):
                message = event.data
                try:
                    content = json.dumps(json.loads(content), indent=2)
                except Exception:
                    pass
                text += (
                    f"<tool result> [{message.tool_call_id}]\n"
                    f"{content}\n</tool result>\n"
                )

            stream_text(text, color)

        if isinstance(
            event, (ProcPacketOutputEvent, WorkflowResultEvent, RunResultEvent)
        ):
            _print_packet(event)
