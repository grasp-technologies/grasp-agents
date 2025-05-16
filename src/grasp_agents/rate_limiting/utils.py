import inspect
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, overload

from .types import P, ProcessorCallableList, ProcessorCallableSingle, R, T


def is_bound_method(func: Callable[..., Any], self_candidate: Any) -> bool:
    return (inspect.ismethod(func) and (func.__self__ is self_candidate)) or hasattr(
        self_candidate, func.__name__
    )


@overload
def split_pos_args(
    call: ProcessorCallableSingle[T, P, R],
    args: Sequence[Any],
) -> tuple[Any | None, T, Sequence[Any]]: ...


@overload
def split_pos_args(
    call: ProcessorCallableList[T, P, R],
    args: Sequence[Any],
) -> tuple[Any | None, list[T], Sequence[Any]]: ...


def split_pos_args(
    call: (ProcessorCallableSingle[T, P, R] | ProcessorCallableList[T, P, R]),
    args: Sequence[Any],
) -> tuple[Any | None, T | list[T], Sequence[Any]]:
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
        return maybe_self, args[1], args[2:]
    # Case: Standalone function with signature (inp, *rest)
    if not args:
        raise ValueError(
            "Must pass an input (or a list of inputs) " + "for a standalone function."
        )
    return None, args[0], args[1:]


def partial_processor_callable(
    call: Callable[..., Coroutine[Any, Any, R]],
    self_obj: Any,
    *args: Any,
    **kwargs: Any,
) -> Callable[[Any], Coroutine[Any, Any, R]]:
    async def wrapper(inp: Any) -> R:
        if self_obj is not None:
            # `call` is a method
            return await call(self_obj, inp, *args, **kwargs)
        # `call` is a function
        return await call(inp, *args, **kwargs)

    return wrapper
