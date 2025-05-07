from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

MAX_RPM = 1e10


@dataclass
class RateLimiterState:
    next_request_time: float = 0.0


QueryT = TypeVar("QueryT")
QueryR = TypeVar("QueryR")
QueryP = ParamSpec("QueryP")

RetrievalFuncSingle: TypeAlias = Callable[
    Concatenate[QueryT, QueryP], Coroutine[Any, Any, QueryR]
]
RetrievalFuncList: TypeAlias = Callable[
    Concatenate[list[QueryT], QueryP], Coroutine[Any, Any, list[QueryR]]
]

RetrievalMethodSingle: TypeAlias = Callable[
    Concatenate[Any, QueryT, QueryP], Coroutine[Any, Any, QueryR]
]
RetrievalMethodList: TypeAlias = Callable[
    Concatenate[Any, list[QueryT], QueryP], Coroutine[Any, Any, list[QueryR]]
]

RetrievalCallableSingle: TypeAlias = (
    RetrievalFuncSingle[QueryT, QueryP, QueryR]
    | RetrievalMethodSingle[QueryT, QueryP, QueryR]
)

RetrievalCallableList: TypeAlias = (
    RetrievalFuncList[QueryT, QueryP, QueryR]
    | RetrievalMethodList[QueryT, QueryP, QueryR]
)


RateLimDecoratorWithArgsSingle = Callable[
    [RetrievalCallableSingle[QueryT, QueryP, QueryR]],
    RetrievalCallableSingle[QueryT, QueryP, QueryR],
]


RateLimDecoratorWithArgsList = Callable[
    [RetrievalCallableList[QueryT, QueryP, QueryR]],
    RetrievalCallableList[QueryT, QueryP, QueryR],
]
