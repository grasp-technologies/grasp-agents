import inspect
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, cast

from .types import AsyncFunctionOrMethod, P, R, T


def is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(
        self_candidate, func.__name__
    )


def split_pos_args(
    call: AsyncFunctionOrMethod[T, P, R], args: tuple[Any, ...]
) -> tuple[Any | None, T | Sequence[T], tuple[Any, ...]]:
    if not args:
        raise ValueError("No positional arguments passed.")
    maybe_self = args[0]
    if is_bound_method(call, maybe_self):
        # Case: Bound instance method with signature (self, inp, *rest)
        if len(args) < 2:
            raise ValueError(
                "Must pass at least `self` and an input (or a list of inputs) "
                "for a bound instance method."
            )
        self_arg = args[0]
        first_arg = cast("T | Sequence[T]", args[1])
        remaining_args = args[2:]

        return self_arg, first_arg, remaining_args

    # Case: Standalone function with signature (inp, *rest)
    if not args:
        raise ValueError(
            "Must pass an input (or a list of inputs) " + "for a standalone function."
        )
    self_arg = None
    first_arg = cast("T | Sequence[T]", args[0])
    remaining_args = args[1:]

    return self_arg, first_arg, remaining_args


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
