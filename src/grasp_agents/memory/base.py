"""
Working-memory ABC consumed by :class:`Processor`.

This is the per-processor scratchpad — distinct from the cross-session
memdir under :class:`MemoryProvider`. ``LLMAgentMemory`` is the concrete
implementation used by ``LLMAgent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..run_context import RunContext


class Memory(BaseModel, ABC):
    @abstractmethod
    def reset(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
        pass

    @abstractmethod
    def erase(self) -> None:
        pass

    @abstractmethod
    def update(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
        pass

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DummyMemory(Memory):
    def reset(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
        del args, ctx, kwargs

    def erase(self) -> None:
        pass

    def update(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
        del args, ctx, kwargs

    @property
    def is_empty(self) -> bool:
        return True
