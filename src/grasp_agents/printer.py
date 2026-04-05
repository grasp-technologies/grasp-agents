import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel
from termcolor import colored
from termcolor._types import Color

from .types.content import InputImage, InputText
from .types.events import (
    Event,
    LLMStreamEvent,
    ProcPacketOutEvent,
    RunPacketOutEvent,
    SystemMessageEvent,
    ToolMessageEvent,
    UserMessageEvent,
)
from .types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from .types.llm_events import (
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningContentPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseCreated,
)

logger = logging.getLogger(__name__)


ROLE_TO_COLOR: dict[str, Color] = {
    "system": "magenta",
    "developer": "magenta",
    "user": "green",
    "assistant": "light_blue",
    "tool": "blue",
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


def stream_colored_text(new_colored_text: str) -> None:
    sys.stdout.write(new_colored_text)
    sys.stdout.flush()


def get_color(
    agent_name: str = "", role: str = "assistant", color_by: ColoringMode = "role"
) -> Color:
    if color_by == "agent":
        idx = int(
            hashlib.md5(agent_name.encode()).hexdigest(),  # noqa: S324
            16,
        ) % len(AVAILABLE_COLORS)

        return AVAILABLE_COLORS[idx]
    return ROLE_TO_COLOR.get(role, "light_blue")


def truncate_content_str(content_str: str, trunc_len: int = 2000) -> str:
    if len(content_str) > trunc_len:
        return content_str[:trunc_len] + "[...]"

    return content_str


def prettify_json_str(json_str: str) -> str:
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2)
    except Exception:
        return json_str


def _input_message_text(msg: InputMessageItem) -> str:
    parts: list[str] = []
    for part in msg.content_parts:
        if isinstance(part, InputText):
            parts.append(part.text.strip(" \n"))
        elif isinstance(part, InputImage):
            if part.is_url:
                parts.append(part.image_url or "")
            else:
                parts.append("<ENCODED_IMAGE>")
    return "\n".join(parts)


class Printer:
    def __init__(
        self,
        color_by: ColoringMode = "role",
        msg_trunc_len: int = 20000,
        output_to: Literal["stdout", "log"] = "stdout",
        logging_level: Literal["info", "debug", "warning", "error"] = "info",
    ) -> None:
        self.color_by: ColoringMode = color_by
        self.msg_trunc_len = msg_trunc_len
        self._logging_level = logging_level
        self._output_to = output_to

    def _output(self, text: str, color: Color) -> None:
        if self._output_to == "log":
            log_kwargs: dict[str, Any] = {"extra": {"color": color}}
            getattr(logger, self._logging_level)(text, **log_kwargs)
        else:
            stream_colored_text(colored(text + "\n", color))

    def print_message(
        self,
        message: Any,
        agent_name: str,
        call_id: str,
    ) -> None:
        out = f"<{agent_name}> [{call_id}]\n"

        if isinstance(message, InputMessageItem):
            role = message.role
            color = get_color(agent_name=agent_name, role=role, color_by=self.color_by)
            content = _input_message_text(message)
            content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
            if role == "system":
                out += f"<system>\n{content}\n</system>\n"
            else:
                out += f"<input>\n{content}\n</input>\n"

        elif isinstance(message, FunctionToolOutputItem):
            color = get_color(
                agent_name=agent_name, role="tool", color_by=self.color_by
            )
            content = message.output if isinstance(message.output, str) else ""
            try:
                content = json.dumps(json.loads(content), indent=2)
            except Exception:
                pass
            content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
            out += f"<tool result> [{message.call_id}]\n{content}\n</tool result>\n"

        else:
            color = get_color(
                agent_name=agent_name, role="assistant", color_by=self.color_by
            )
            out += f"{message}\n"

        self._output(out, color)

    def print_messages(
        self,
        messages: Sequence[Any],
        agent_name: str,
        call_id: str,
    ) -> None:
        for _message in messages:
            self.print_message(_message, agent_name=agent_name, call_id=call_id)


async def print_event_stream(
    event_generator: AsyncIterator[Event[Any]],
    color_by: ColoringMode = "role",
    trunc_len: int = 10000,
    exclude_packet_events: bool = False,
) -> AsyncIterator[Event[Any]]:
    def _make_stream_event_text(event: LLMStreamEvent) -> str:
        se = event.data
        color = get_color(
            agent_name=event.src_name or "", role="assistant", color_by=color_by
        )
        text = ""

        if isinstance(se, ResponseCreated):
            text += f"\n<{event.src_name}> [{event.call_id}]\n"

        elif isinstance(se, OutputItemAdded):
            item = se.item
            if isinstance(item, ReasoningItem):
                text += "<thinking>\n"
            elif isinstance(item, OutputMessageItem):
                text += "<response>\n"
            elif isinstance(item, FunctionToolCallItem):
                text += f"<tool call> {item.name} [{item.call_id}]\n"

        elif isinstance(se, OutputItemDone):
            item = se.item
            if isinstance(item, ReasoningItem):
                text += "\n</thinking>\n"
            elif isinstance(item, OutputMessageItem):
                text += "\n</response>\n"
            elif isinstance(item, FunctionToolCallItem):
                text += "\n</tool call>\n"

        elif isinstance(
            se,
            OutputMessageTextPartTextDelta
            | ReasoningSummaryPartTextDelta
            | ReasoningContentPartTextDelta
            | FunctionCallArgumentsDelta,
        ):
            text += se.delta

        return colored(text, color)

    def _make_message_text(
        event: SystemMessageEvent | UserMessageEvent | ToolMessageEvent,
    ) -> str:
        data = event.data
        color = get_color(
            agent_name=event.src_name or "", role="assistant", color_by=color_by
        )
        text = f"\n<{event.src_name}> [{event.call_id}]\n"

        if isinstance(event, SystemMessageEvent):
            assert isinstance(data, InputMessageItem)
            content = _input_message_text(data)
            content = truncate_content_str(content, trunc_len=trunc_len)
            color = get_color(
                agent_name=event.src_name or "",
                role="system",
                color_by=color_by,
            )
            text += f"<system>\n{content}\n</system>\n"

        elif isinstance(event, UserMessageEvent):
            assert isinstance(data, InputMessageItem)
            content = _input_message_text(data)
            content = truncate_content_str(content, trunc_len=trunc_len)
            color = get_color(
                agent_name=event.src_name or "",
                role="user",
                color_by=color_by,
            )
            text += f"<input>\n{content}\n</input>\n"

        else:
            assert isinstance(data, FunctionToolOutputItem)
            content = data.output if isinstance(data.output, str) else ""
            try:
                content = json.dumps(json.loads(content), indent=2)
            except Exception:
                pass
            color = get_color(
                agent_name=event.src_name or "",
                role="tool",
                color_by=color_by,
            )
            text += f"<tool result> [{data.call_id}]\n{content}\n</tool result>\n"

        return colored(text, color)

    def _make_packet_text(
        event: ProcPacketOutEvent | RunPacketOutEvent,
    ) -> str:
        src = "run" if isinstance(event, RunPacketOutEvent) else "processor"

        color = get_color(
            agent_name=event.src_name or "", role="assistant", color_by=color_by
        )
        text = f"\n<{event.src_name}> [{event.call_id}]\n"

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
        if isinstance(event, LLMStreamEvent):
            text = _make_stream_event_text(event)
            if text:
                stream_colored_text(text)

        elif isinstance(
            event, SystemMessageEvent | UserMessageEvent | ToolMessageEvent
        ):
            stream_colored_text(_make_message_text(event))

        elif (
            isinstance(event, ProcPacketOutEvent | RunPacketOutEvent)
            and not exclude_packet_events
        ):
            stream_colored_text(_make_packet_text(event))  # type: ignore[arg-type]

        yield event
