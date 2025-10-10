import inspect
from collections.abc import Callable, Coroutine
from typing import Any

from .types import AsyncFunctionOrMethod, P, R


def is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(
        self_candidate, func.__name__
    )


def split_pos_args(
    call: AsyncFunctionOrMethod[P, R], args: tuple[Any, ...]
) -> tuple[Any | None, tuple[Any, ...]]:
    if not args:
        raise ValueError("No positional arguments passed.")
    maybe_self = args[0]
    if is_bound_method(call, maybe_self):
        # Case: Bound instance method with signature (self, inp, *rest)
        self_arg = args[0]
        remaining_args = args[1:]

        return self_arg, remaining_args

    # Case: Standalone function with signature (inp, *rest)
    if not args:
        raise ValueError(
            "Must pass an input (or a list of inputs) for a standalone function."
        )
    self_arg = None
    remaining_args = args

    return self_arg, remaining_args


def partial_callable(
    call: Callable[..., Coroutine[Any, Any, R]],
    self_obj: Any,
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapper(inp: Any) -> R:
        if self_obj is not None:
            # `call` is a method
            return await call(self_obj, inp, *args, **kwargs)
        # `call` is a function
        return await call(inp, *args, **kwargs)

    return wrapper
