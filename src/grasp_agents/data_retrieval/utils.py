import inspect
from collections.abc import Callable, Coroutine, Sequence
from typing import (
    Any,
)

from .types import (
    QueryP,
    QueryR,
    QueryT,
    RetrievalCallableList,
    RetrievalCallableSingle,
)


def is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(self_candidate, func.__name__)


def split_pos_args(
    call: (RetrievalCallableSingle[QueryT, QueryP, QueryR] | RetrievalCallableList[QueryT, QueryP, QueryR]),
    args: Sequence[Any],
) -> tuple[Any | None, QueryT | list[QueryT], Sequence[Any]]:
    if not args:
        raise ValueError("No positional arguments passed.")
    maybe_self = args[0]
    if is_bound_method(call, maybe_self):
        # Case: Bound instance method with signature (self, inp, *rest)
        if len(args) < 2:
            raise ValueError(
                "Must pass at least `self` and an input (or a list of inputs) " + "for a bound instance method."
            )
        return maybe_self, args[1], args[2:]
    # Case: Standalone function with signature (inp, *rest)
    if not args:
        raise ValueError("Must pass an input (or a list of inputs) " + "for a standalone function.")
    return None, args[0], args[1:]


def partial_retrieval_callable(
    call: Callable[..., Coroutine[Any, Any, QueryR]],
    self_obj: Any,
    *args: QueryP.args,
    **kwargs: QueryP.kwargs,
) -> Callable[[QueryT], Coroutine[Any, Any, QueryR]]:
    async def wrapper(inp: QueryT) -> QueryR:
        if self_obj is not None:
            # `call` is a method
            return await call(self_obj, inp, *args, **kwargs)
        # `call` is a function
        return await call(inp, *args, **kwargs)

    return wrapper


def expected_exec_time_from_max_concurrency_and_rpm(rpm: float, max_concurrency: int) -> float:
    return 60.0 / (rpm / max_concurrency)
