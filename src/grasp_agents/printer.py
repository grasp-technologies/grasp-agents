import hashlib
import json
import logging
import re
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

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
    WebSearchCallItem,
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

# C0 control characters (including ESC) minus tab / newline, plus DEL: any of
# these reaching the terminal can clear the screen, move the cursor, or spoof
# the window title (CSI / OSC injection via untrusted tool output — Rich
# strips BEL but passes ESC through).
_TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_terminal_text(text: str) -> str:
    r"""
    Neutralize terminal control / escape characters in untrusted text.

    Lone ``\r`` becomes ``\n`` (carriage-return line-overwrite spoofing);
    other control characters are dropped, leaving any sequence payload
    visible as plain text.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _TERMINAL_CONTROL_RE.sub("", normalized)


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

type ColoringMode = Literal["agent", "role"]


def stream_colored_text(new_colored_text: str) -> None:
    sys.stdout.write(new_colored_text)
    sys.stdout.flush()


def _styled_str(text: str, style: str) -> str:
    """Return text wrapped in ANSI escape codes via Rich."""
    t = Text(sanitize_terminal_text(text), style=style)
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


def render_payload(payload: Any) -> str:
    r"""
    Stringify a packet payload for display.

    A plain ``str`` is shown verbatim — no wrapping quotes, no ``\uXXXX``
    escaping; a pydantic model as indented JSON; anything else as indented JSON
    (UTF-8, non-ASCII kept), falling back to ``str()``.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, BaseModel):
        return payload.model_dump_json(indent=2)
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except TypeError:
        return str(payload)


def _input_message_text(msg: InputMessageItem) -> str:
    parts: list[str] = []
    for part in msg.content:
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
            getattr(logger, self._logging_level)(
                sanitize_terminal_text(text), **log_kwargs
            )
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
            # Generated (assistant) output — the raw printer shows everything the
            # model produces: its reasoning, its text, and its tool calls.
            style = get_style(
                agent_name=agent_name,
                role="assistant",
                color_by=self.color_by,
            )
            if isinstance(message, ReasoningItem):
                thinking = message.content_text or message.summary_text
                thinking = truncate_content_str(thinking, trunc_len=self.msg_trunc_len)
                out += f"<thinking>\n{thinking}\n</thinking>\n"
            elif isinstance(message, OutputMessageItem):
                content = message.refusal or message.text
                content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
                out += f"<response>\n{content}\n</response>\n"
            elif isinstance(message, FunctionToolCallItem):
                args = prettify_json_str(message.arguments)
                args = truncate_content_str(args, trunc_len=self.msg_trunc_len)
                out += (
                    f"<tool call> {message.name} [{message.call_id}]\n"
                    f"{args}\n</tool call>\n"
                )
            elif isinstance(message, WebSearchCallItem):
                out += f"<web search>\n{message.summary}\n</web search>\n"
            else:
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

    async def stream(
        self,
        events: AsyncIterator[Event[Any]],
        *,
        exclude_packet_events: bool = False,
    ) -> AsyncIterator[Event[Any]]:
        """
        Render an event stream raw, yielding each event through.

        Shows everything the model produces or ingests — thinking, response
        text, tool-call arguments, tool results — streamed token-by-token and
        wrapped in tags, with no formatting. The raw counterpart to
        :meth:`EventConsole.stream` (use that for readable output);
        ``ctx.printer`` is the inline (non-wrapping) variant of the same.
        """

        def _make_stream_event_text(event: LLMStreamEvent) -> str:
            se = event.data
            style = get_style(
                agent_name=event.source or "",
                role="assistant",
                color_by=self.color_by,
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
                else:  # WebSearchCallItem — server-side web search / fetch
                    text += "<web search>\n"

            elif isinstance(se, OutputItemDone):
                item = se.item
                if isinstance(item, ReasoningItem):
                    text += "\n</thinking>\n"
                elif isinstance(item, OutputMessageItem):
                    text += "\n</response>\n"
                elif isinstance(item, FunctionToolCallItem):
                    text += "\n</tool call>\n"
                else:  # WebSearchCallItem
                    text += f"{item.summary}\n</web search>\n"

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
                content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
                style = get_style(
                    agent_name=event.source or "",
                    role="system",
                    color_by=self.color_by,
                )
                text += f"<system>\n{content}\n</system>\n"

            elif isinstance(event, UserMessageEvent):
                assert isinstance(data, InputMessageItem)
                content = _input_message_text(data)
                content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
                style = get_style(
                    agent_name=event.source or "",
                    role="user",
                    color_by=self.color_by,
                )
                text += f"<input>\n{content}\n</input>\n"

            else:
                assert isinstance(data, FunctionToolOutputItem)
                content = data.output if isinstance(data.output, str) else ""
                try:
                    content = json.dumps(json.loads(content), indent=2)
                except Exception:
                    pass
                content = truncate_content_str(content, trunc_len=self.msg_trunc_len)
                style = get_style(
                    agent_name=event.source or "",
                    role="tool",
                    color_by=self.color_by,
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
                color_by=self.color_by,
            )
            text = f"\n<{event.source}> [{event.exec_id}]\n"

            if event.data.payloads:
                text += f"<{src} output>\n"
                for p in event.data.payloads:
                    text += f"{render_payload(p)}\n"
                text += f"</{src} output>\n"

            return _styled_str(text, style)

        async for event in events:
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


async def print_events(
    events: AsyncIterator[Event[Any]],
    *,
    color_by: ColoringMode = "role",
    trunc_len: int = 20000,
    exclude_packet_events: bool = False,
) -> AsyncIterator[Event[Any]]:
    """
    Wrap an event stream with raw :class:`Printer` display, yielding events
    through — the raw counterpart to :func:`grasp_agents.ui.render_events`::

        async for event in print_events(agent.run_stream("hello")):
            ...

    Thin sugar over :meth:`Printer.stream`; use ``EventConsole`` /
    ``render_events`` for readable (panelled, Markdown) output instead.
    """
    printer = Printer(color_by=color_by, msg_trunc_len=trunc_len)
    async for event in printer.stream(
        events, exclude_packet_events=exclude_packet_events
    ):
        yield event
