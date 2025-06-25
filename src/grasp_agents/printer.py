import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

from termcolor._types import Color  # type: ignore[import]

from .typing.completion import Usage
from .typing.content import Content, ContentPartText
from .typing.message import AssistantMessage, Message, Role, ToolMessage

logger = logging.getLogger(__name__)


ColoringMode: TypeAlias = Literal["agent", "role"]

ROLE_TO_COLOR: Mapping[Role, Color] = {
    Role.SYSTEM: "magenta",
    Role.USER: "green",
    Role.ASSISTANT: "light_blue",
    Role.TOOL: "light_cyan",
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


class Printer:
    def __init__(
        self,
        color_by: ColoringMode = "role",
        msg_trunc_len: int = 20000,
        print_messages: bool = False,
    ) -> None:
        self.color_by = color_by
        self.msg_trunc_len = msg_trunc_len
        self.print_messages = print_messages

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

    def print_llm_message(
        self, message: Message, agent_name: str, run_id: str, usage: Usage | None = None
    ) -> None:
        if not self.print_messages:
            return

        if usage is not None and not isinstance(message, AssistantMessage):
            raise ValueError(
                "Usage information can only be printed for AssistantMessage"
            )

        role = message.role
        content_str = self.content_to_str(message.content or "", message.role)

        if self.color_by == "agent":
            color = self.get_agent_color(agent_name)
        elif self.color_by == "role":
            color = self.get_role_color(role)

        log_kwargs = {"extra": {"color": color}}  # type: ignore

        # Print message title

        out = f"\n[agent: {agent_name} | role: {role.value} | run: {run_id}]"

        if isinstance(message, ToolMessage):
            out += f"\n{message.name} | {message.tool_call_id}"

        # Print message content

        if content_str:
            try:
                content_str = json.dumps(json.loads(content_str), indent=2)
            except Exception:
                pass
            content_str_truncated = self.truncate_content_str(
                content_str, trunc_len=self.msg_trunc_len
            )
            out += f"\n{content_str_truncated}"

        logger.debug(out, **log_kwargs)  # type: ignore

        # Print tool calls

        if isinstance(message, AssistantMessage) and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                if self.color_by == "agent":
                    tool_color = self.get_agent_color(agent_name=agent_name)
                elif self.color_by == "role":
                    tool_color = self.get_role_color(role=Role.TOOL)
                logger.debug(
                    f"\n<{agent_name}>[TOOL_CALL]\n{tool_call.tool_name} "
                    f"| {tool_call.id}\n{tool_call.tool_arguments}",
                    extra={"color": tool_color},  # type: ignore
                )

        # Print usage

        if usage is not None:
            usage_str = (
                f"I/O/(R)/(C) tokens: {usage.input_tokens}/{usage.output_tokens}"
            )
            if usage.reasoning_tokens is not None:
                usage_str += f"/{usage.reasoning_tokens}"
            if usage.cached_tokens is not None:
                usage_str += f"/{usage.cached_tokens}"
            logger.debug(
                f"\n------------------------------------\n{usage_str}",
                **log_kwargs,  # type: ignore
            )

    def print_llm_messages(
        self,
        messages: Sequence[Message],
        agent_name: str,
        run_id: str,
        usages: Sequence[Usage | None] | None = None,
    ) -> None:
        if not self.print_messages:
            return

        _usages: Sequence[Usage | None] = usages or [None] * len(messages)

        for _message, _usage in zip(messages, _usages, strict=False):
            self.print_llm_message(
                _message, usage=_usage, agent_name=agent_name, run_id=run_id
            )
