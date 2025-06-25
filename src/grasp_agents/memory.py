from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from .run_context import RunContext

MemT = TypeVar("MemT", bound="Memory")


class Memory(BaseModel, ABC):
    @abstractmethod
    def reset(
        self, *args: Any, ctx: RunContext[Any] | None = None, **kwargs: Any
    ) -> None:
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
