from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .run_context import RunContext


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
        pass

    def erase(self) -> None:
        pass

    def update(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
        pass

    @property
    def is_empty(self) -> bool:
        return True
