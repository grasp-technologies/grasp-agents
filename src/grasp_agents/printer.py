import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel
from termcolor import colored
from termcolor._types import Color

from grasp_agents.typing.events import (
    CompletionChunkEvent,
    Event,
    GenMessageEvent,
    MessageEvent,
    ProcPacketOutputEvent,
    RunResultEvent,
    SystemMessageEvent,
    ToolMessageEvent,
    UserMessageEvent,
    WorkflowResultEvent,
)

from .typing.completion import Usage
from .typing.content import Content, ContentPartText
from .typing.message import AssistantMessage, Message, Role, SystemMessage, UserMessage

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
    def content_to_str(content: Content | str, role: Role) -> str:
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

        assert isinstance(content, str)

        return content.strip(" \n")

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
            out += f"\n<thinking>\n{thinking}\n</thinking>\n"

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
    trunc_len: int = 1000,
) -> AsyncIterator[Event[Any]]:
    prev_chunk_id: str | None = None
    thinking_open = False
    response_open = False
    open_tool_calls: set[str] = set()

    color = Printer.get_role_color(Role.ASSISTANT)

    def _close_blocks(
        _thinking_open: bool, _response_open: bool, color: Color
    ) -> tuple[bool, bool]:
        closing_text = ""
        while open_tool_calls:
            open_tool_calls.pop()
            closing_text += "\n</tool call>\n"

        if _thinking_open:
            closing_text += "\n</thinking>\n"
            _thinking_open = False

        if _response_open:
            closing_text += "\n</response>\n"
            _response_open = False

        if closing_text:
            stream_text(closing_text, color)

        return _thinking_open, _response_open

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
                    for field_info in type(p).model_fields.values():
                        if field_info.exclude:
                            field_info.exclude = False
                    type(p).model_rebuild(force=True)
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

        if isinstance(event, CompletionChunkEvent):
            delta = event.data.choices[0].delta
            chunk_id = event.data.id
            new_completion = chunk_id != prev_chunk_id
            color = _get_color(event, Role.ASSISTANT)

            text = ""

            if new_completion:
                thinking_open, response_open = _close_blocks(
                    thinking_open, response_open, color
                )
                text += f"\n<{event.proc_name}> [{event.call_id}]\n"

            if delta.reasoning_content:
                if not thinking_open:
                    text += "<thinking>\n"
                    thinking_open = True
                text += delta.reasoning_content
            elif thinking_open:
                text += "\n</thinking>\n"
                thinking_open = False

            if delta.content:
                if not response_open:
                    text += "<response>\n"
                    response_open = True
                text += delta.content
            elif response_open:
                text += "\n</response>\n"
                response_open = False

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id and tc.id not in open_tool_calls:
                        open_tool_calls.add(tc.id)  # type: ignore
                        text += f"<tool call> {tc.tool_name} [{tc.id}]\n"

                    if tc.tool_arguments:
                        text += tc.tool_arguments

            stream_text(text, color)
            prev_chunk_id = chunk_id

        else:
            thinking_open, response_open = _close_blocks(
                thinking_open, response_open, color
            )

        if isinstance(event, MessageEvent) and not isinstance(event, GenMessageEvent):
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
