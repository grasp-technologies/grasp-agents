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


def stream_colored_text(new_colored_text: str) -> None:
    sys.stdout.write(new_colored_text)
    sys.stdout.flush()


def get_color(
    agent_name: str = "", role: Role = Role.ASSISTANT, color_by: ColoringMode = "role"
) -> Color:
    if color_by == "agent":
        idx = int(
            hashlib.md5(agent_name.encode()).hexdigest(),  # noqa :S324
            16,
        ) % len(AVAILABLE_COLORS)

        return AVAILABLE_COLORS[idx]
    return ROLE_TO_COLOR[role]


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


def truncate_content_str(content_str: str, trunc_len: int = 2000) -> str:
    if len(content_str) > trunc_len:
        return content_str[:trunc_len] + "[...]"

    return content_str


class Printer:
    def __init__(
        self,
        color_by: ColoringMode = "role",
        msg_trunc_len: int = 20000,
        output_to: Literal["print", "log"] = "print",
        logging_level: Literal["info", "debug", "warning", "error"] = "info",
    ) -> None:
        self.color_by: ColoringMode = color_by
        self.msg_trunc_len = msg_trunc_len
        self._current_message: str = ""
        self._logging_level = logging_level
        self._output_to = output_to

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
        color = get_color(
            agent_name=agent_name, role=message.role, color_by=self.color_by
        )
        log_kwargs = {"extra": {"color": color}}

        out = f"<{agent_name}> [{call_id}]\n"

        # Thinking
        if isinstance(message, AssistantMessage) and message.reasoning_content:
            thinking = message.reasoning_content.strip(" \n")
            out += f"<thinking>\n{thinking}\n</thinking>\n"

        # Content
        content = content_to_str(message.content or "", message.role)
        if content:
            try:
                content = json.dumps(json.loads(content), indent=2)
            except Exception:
                pass
            content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
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

        if self._output_to == "log":
            if self._logging_level == "debug":
                logger.debug(out, **log_kwargs)  # type: ignore
            elif self._logging_level == "info":
                logger.info(out, **log_kwargs)  # type: ignore
            elif self._logging_level == "warning":
                logger.warning(out, **log_kwargs)  # type: ignore
            else:
                logger.error(out, **log_kwargs)  # type: ignore
        else:
            stream_colored_text(colored(out + "\n", color))

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


async def print_event_stream(
    event_generator: AsyncIterator[Event[Any]],
    color_by: ColoringMode = "role",
    trunc_len: int = 10000,
) -> AsyncIterator[Event[Any]]:
    def _make_chunk_text(event: CompletionChunkEvent[CompletionChunk]) -> str:
        color = get_color(
            agent_name=event.proc_name or "", role=Role.ASSISTANT, color_by=color_by
        )
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

        return colored(text, color)

    def _make_message_text(
        event: MessageEvent[SystemMessage | UserMessage | ToolMessage],
    ) -> str:
        message = event.data
        role = message.role
        content = content_to_str(message.content, role=role)

        color = get_color(
            agent_name=event.proc_name or "", role=role, color_by=color_by
        )
        text = f"\n<{event.proc_name}> [{event.call_id}]\n"

        if isinstance(event, (SystemMessageEvent, UserMessageEvent)):
            content = truncate_content_str(content, trunc_len=trunc_len)

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
                f"<tool result> [{message.tool_call_id}]\n{content}\n</tool result>\n"
            )

        return colored(text, color)

    def _make_packet_text(
        event: ProcPacketOutputEvent | WorkflowResultEvent | RunResultEvent,
    ) -> str:
        if isinstance(event, WorkflowResultEvent):
            src = "workflow"
        elif isinstance(event, RunResultEvent):
            src = "run"
        else:
            src = "processor"

        color = get_color(
            agent_name=event.proc_name or "", role=Role.ASSISTANT, color_by=color_by
        )
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

        return colored(text, color)

    # ------ Wrap event generator -------

    async for event in event_generator:
        yield event

        if isinstance(event, CompletionChunkEvent) and isinstance(
            event.data, CompletionChunk
        ):
            stream_colored_text(_make_chunk_text(event))

        if isinstance(event, MessageEvent) and not isinstance(event, GenMessageEvent):
            stream_colored_text(_make_message_text(event))

        if isinstance(
            event, (ProcPacketOutputEvent, WorkflowResultEvent, RunResultEvent)
        ):
            stream_colored_text(_make_packet_text(event))
