from pydantic import BaseModel

from ..typing.message import AssistantMessage, Role
from ..typing.tool import ToolCall


class Reasoning(BaseModel):
    summaries: list[str]
    content: list[str]


class CompletionItem(BaseModel):
    item: AssistantMessage | ToolCall | Reasoning
    role: Role | None = None
    name: str | None = None
