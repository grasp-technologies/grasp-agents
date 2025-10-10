from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar

R = TypeVar("R")
P = ParamSpec("P")

AsyncCallable: TypeAlias = Callable[P, Coroutine[Any, Any, R]]
AsyncFunction: TypeAlias = Callable[P, Coroutine[Any, Any, R]]
AsyncMethod: TypeAlias = Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]
AsyncFunctionOrMethod: TypeAlias = AsyncFunction[P, R] | AsyncMethod[P, R]
