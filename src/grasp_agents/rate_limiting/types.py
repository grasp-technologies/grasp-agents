from collections.abc import Callable, Coroutine
from typing import Any, Concatenate

type AsyncCallable[**P, R] = Callable[P, Coroutine[Any, Any, R]]
type AsyncFunction[**P, R] = Callable[P, Coroutine[Any, Any, R]]
type AsyncMethod[**P, R] = Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]
type AsyncFunctionOrMethod[**P, R] = AsyncFunction[P, R] | AsyncMethod[P, R]
