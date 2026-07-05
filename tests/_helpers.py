"""
Shared, importable test utilities: a queue-driven mock ``LLM``, ``Response``
builders, and simple ``BaseTool`` subclasses.

Pytest *fixtures* (``tools``, ``parallel_tools``, API keys) live in
``conftest.py``; this module holds the plain building blocks that tests
import explicitly. The leading-underscore builder names match the originals so
call sites are unchanged when a local copy is replaced by an import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel, Field

from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.llm.llm import LLM
from grasp_agents.sandbox.environment import ExecutionEnvironment, SnapshotCapable
from grasp_agents.sandbox.policy import SandboxPolicy
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import FunctionToolCallItem, OutputMessageItem
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _text_response(text: str) -> Response:
    return Response(
        model="mock",
        output=[
            OutputMessageItem(
                content=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
        usage=_make_usage(),
    )


def _tool_call_response(name: str, arguments: str, call_id: str) -> Response:
    return Response(
        model="mock",
        output=[
            FunctionToolCallItem(call_id=call_id, name=name, arguments=arguments),
        ],
        usage=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    """LLM stub that pops queued ``Response`` objects and counts calls."""

    model_name: str = "mock"
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_call_count", 0)

    @property
    def call_count(self) -> int:
        return self._call_count  # type: ignore[attr-defined]

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        seq = 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


# --- Simple tools (back the ``tools`` / ``parallel_tools`` fixtures) ---


class AddInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class AddTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="add",
            description="Add two integers and return their sum.",
            **kwargs,
        )

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> int:
        return inp.a + inp.b


class MultiplyInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class MultiplyTool(BaseTool[MultiplyInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="multiply",
            description="Multiply two integers and return their product.",
            **kwargs,
        )

    async def _run(
        self,
        inp: MultiplyInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> int:
        return inp.a * inp.b


# --- Fake environments (SessionContext.environment stand-ins) ---


class FakeSnapshotEnv(ExecutionEnvironment, SnapshotCapable):
    """SnapshotCapable environment over a local backend, recording calls."""

    def __init__(self, root: Path) -> None:
        self._policy = SandboxPolicy(allowed_roots=(root,))
        self._backend = LocalFileBackend(allowed_roots=[root])
        self.snapshots: list[str] = []
        self.restored: list[str] = []

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> LocalFileBackend:
        return self._backend

    @property
    def exec_backend(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def snapshot(self) -> str:
        ref = f"snap-{len(self.snapshots) + 1}"
        self.snapshots.append(ref)
        return ref

    async def restore(self, ref: str) -> None:
        self.restored.append(ref)
