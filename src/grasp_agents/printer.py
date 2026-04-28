import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from .types.content import InputImage, InputText
from .types.events import (
    Event,
    LLMStreamEvent,
    ProcPacketOutEvent,
    RunPacketOutEvent,
    SystemMessageEvent,
    ToolOutputItemEvent,
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

_console = Console(
    force_jupyter=False,
    force_terminal=True,
    highlight=False,
    color_system="256",
)

ROLE_TO_STYLE: dict[str, str] = {
    "system": "magenta",
    "developer": "magenta",
    "user": "green",
    "assistant": "bright_blue",
    "tool": "blue",
}

AVAILABLE_STYLES: list[str] = [
    "magenta",
    "green",
    "bright_blue",
    "bright_cyan",
    "yellow",
    "blue",
    "red",
]

ColoringMode: TypeAlias = Literal["agent", "role"]


def stream_colored_text(new_colored_text: str) -> None:
    sys.stdout.write(new_colored_text)
    sys.stdout.flush()


def _styled_str(text: str, style: str) -> str:
    """Return text wrapped in ANSI escape codes via Rich."""
    t = Text(text, style=style)
    with _console.capture() as capture:
        _console.print(t, end="")
    return capture.get()


def get_style(
    agent_name: str = "",
    role: str = "assistant",
    color_by: ColoringMode = "role",
) -> str:
    if color_by == "agent":
        idx = int(
            hashlib.md5(agent_name.encode()).hexdigest(),  # noqa: S324
            16,
        ) % len(AVAILABLE_STYLES)

        return AVAILABLE_STYLES[idx]
    return ROLE_TO_STYLE.get(role, "bright_blue")


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

    def _output(self, text: str, style: str) -> None:
        if self._output_to == "log":
            log_kwargs: dict[str, Any] = {"extra": {"color": style}}
            getattr(logger, self._logging_level)(text, **log_kwargs)
        else:
            stream_colored_text(_styled_str(text + "\n", style))

    def print_message(
        self,
        message: Any,
        agent_name: str,
        exec_id: str,
    ) -> None:
        out = f"<{agent_name}> [{exec_id}]\n"

        if isinstance(message, InputMessageItem):
            role = message.role
            style = get_style(agent_name=agent_name, role=role, color_by=self.color_by)
            content = _input_message_text(message)
            content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
            if role == "system":
                out += f"<system>\n{content}\n</system>\n"
            else:
                out += f"<input>\n{content}\n</input>\n"

        elif isinstance(message, FunctionToolOutputItem):
            style = get_style(
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
            style = get_style(
                agent_name=agent_name,
                role="assistant",
                color_by=self.color_by,
            )
            out += f"{message}\n"

        self._output(out, style)

    def print_messages(
        self,
        messages: Sequence[Any],
        agent_name: str,
        exec_id: str,
    ) -> None:
        for _message in messages:
            self.print_message(_message, agent_name=agent_name, exec_id=exec_id)


async def print_event_stream(
    event_generator: AsyncIterator[Event[Any]],
    color_by: ColoringMode = "role",
    trunc_len: int = 10000,
    exclude_packet_events: bool = False,
) -> AsyncIterator[Event[Any]]:
    def _make_stream_event_text(event: LLMStreamEvent) -> str:
        se = event.data
        style = get_style(
            agent_name=event.source or "",
            role="assistant",
            color_by=color_by,
        )
        text = ""

        if isinstance(se, ResponseCreated):
            text += f"\n<{event.source}> [{event.exec_id}]\n"

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

        return _styled_str(text, style)

    def _make_message_text(
        event: SystemMessageEvent | UserMessageEvent | ToolOutputItemEvent,
    ) -> str:
        data = event.data
        text = f"\n<{event.source}> [{event.exec_id}]\n"

        if isinstance(event, SystemMessageEvent):
            assert isinstance(data, InputMessageItem)
            content = _input_message_text(data)
            content = truncate_content_str(content, trunc_len=trunc_len)
            style = get_style(
                agent_name=event.source or "",
                role="system",
                color_by=color_by,
            )
            text += f"<system>\n{content}\n</system>\n"

        elif isinstance(event, UserMessageEvent):
            assert isinstance(data, InputMessageItem)
            content = _input_message_text(data)
            content = truncate_content_str(content, trunc_len=trunc_len)
            style = get_style(
                agent_name=event.source or "",
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
            style = get_style(
                agent_name=event.source or "",
                role="tool",
                color_by=color_by,
            )
            text += f"<tool result> [{data.call_id}]\n{content}\n</tool result>\n"

        return _styled_str(text, style)

    def _make_packet_text(
        event: ProcPacketOutEvent | RunPacketOutEvent,
    ) -> str:
        src = "run" if isinstance(event, RunPacketOutEvent) else "processor"

        style = get_style(
            agent_name=event.source or "",
            role="assistant",
            color_by=color_by,
        )
        text = f"\n<{event.source}> [{event.exec_id}]\n"

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

        return _styled_str(text, style)

    # ------ Wrap event generator -------

    async for event in event_generator:
        if isinstance(event, LLMStreamEvent):
            text = _make_stream_event_text(event)
            if text:
                stream_colored_text(text)

        elif isinstance(
            event, SystemMessageEvent | UserMessageEvent | ToolOutputItemEvent
        ):
            stream_colored_text(_make_message_text(event))

        elif (
            isinstance(event, ProcPacketOutEvent | RunPacketOutEvent)
            and not exclude_packet_events
        ):
            stream_colored_text(_make_packet_text(event))  # type: ignore[arg-type]

        yield event
