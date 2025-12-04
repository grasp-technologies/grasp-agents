from uuid import uuid4

from pydantic import BaseModel, Field

from ..typing.message import AssistantMessage, Role
from ..typing.tool import ToolCall


class Reasoning(BaseModel):
    summaries: list[str]
    content: list[str]


class CompletionItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    item: AssistantMessage | ToolCall | Reasoning
    role: Role | None = None
    name: str | None = None
