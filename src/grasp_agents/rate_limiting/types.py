from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

ProcessorFuncSingle: TypeAlias = Callable[Concatenate[T, P], Coroutine[Any, Any, R]]
ProcessorFuncList: TypeAlias = Callable[
    Concatenate[list[T], P], Coroutine[Any, Any, list[R]]
]

ProcessorMethodSingle: TypeAlias = Callable[
    Concatenate[Any, T, P], Coroutine[Any, Any, R]
]
ProcessorMethodList: TypeAlias = Callable[
    Concatenate[Any, list[T], P], Coroutine[Any, Any, list[R]]
]

ProcessorCallableSingle: TypeAlias = (
    ProcessorFuncSingle[T, P, R] | ProcessorMethodSingle[T, P, R]
)

ProcessorCallableList: TypeAlias = (
    ProcessorFuncList[T, P, R] | ProcessorMethodList[T, P, R]
)


RateLimWrapperWithArgsSingle = Callable[
    [ProcessorCallableSingle[T, P, R]], ProcessorCallableSingle[T, P, R]
]


RateLimWrapperWithArgsList = Callable[
    [ProcessorCallableList[T, P, R]], ProcessorCallableList[T, P, R]
]
