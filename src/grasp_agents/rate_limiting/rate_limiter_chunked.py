import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from time import monotonic
from typing import Any, Generic, overload

from tqdm.autonotebook import tqdm

from ..utils import asyncio_gather_with_pbar
from .types import (
    P,
    ProcessorCallableList,
    ProcessorCallableSingle,
    R,
    RateLimWrapperWithArgsList,
    RateLimWrapperWithArgsSingle,
    T,
)
from .utils import partial_processor_callable, split_pos_args

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterState:
    next_request_time: float = 0.0


class RateLimiterC(Generic[T, R]):
    def __init__(
        self,
        rpm: float,
        chunk_size: int = 1000,
        max_concurrency: int = 200,
    ):
        self._rpm = rpm
        self._max_concurrency = max_concurrency
        self._chunk_size = chunk_size

        self._lock = asyncio.Lock()
        self._state = RateLimiterState(next_request_time=0.0)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    async def process_input(
        self,
        func_partial: Callable[[T], Coroutine[Any, Any, R]],
        inp: T,
    ) -> R:
        async with self._semaphore:
            async with self._lock:
                now = monotonic()
                if now < self._state.next_request_time:
                    await asyncio.sleep(self._state.next_request_time - now)
                self._state.next_request_time = monotonic() + 1.01 * 60.0 / self._rpm
            return await func_partial(inp)

    async def process_inputs(
        self,
        func_partial: Callable[[T], Coroutine[Any, Any, R]],
        inputs: list[T],
        no_tqdm: bool = False,
    ) -> list[R]:
        results: list[R] = []
        for i in tqdm(
            range(0, len(inputs), self._chunk_size),
            disable=no_tqdm,
            desc="Processing chunks",
        ):
            chunk = inputs[i : i + self._chunk_size]
            corouts = [
                self.process_input(func_partial=func_partial, inp=inp) for inp in chunk
            ]
            chunk_results = await asyncio.gather(*corouts)
            results.extend(chunk_results)

        return results

    @property
    def rpm(self) -> float:
        return self._rpm

    @rpm.setter
    def rpm(self, value: float) -> None:
        self._rpm = value

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def state(self) -> RateLimiterState:
        return self._state


@overload
def limit_rate(
    call: ProcessorCallableSingle[T, P, R],
    rate_limiter: RateLimiterC[T, R] | None = None,
) -> ProcessorCallableSingle[T, P, R]: ...


@overload
def limit_rate(
    call: None = None,
    rate_limiter: RateLimiterC[T, R] | None = None,
) -> RateLimWrapperWithArgsSingle[T, P, R]: ...


def limit_rate(
    call: ProcessorCallableSingle[T, P, R] | None = None,
    rate_limiter: RateLimiterC[T, R] | None = None,
) -> ProcessorCallableSingle[T, P, R] | RateLimWrapperWithArgsSingle[T, P, R]:
    if call is None:
        return functools.partial(limit_rate, rate_limiter=rate_limiter)

    @functools.wraps(call)  # type: ignore
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        inp: T
        self_obj, inp, other_args = split_pos_args(call, args)
        call_partial = partial_processor_callable(call, self_obj, *other_args, **kwargs)

        _rate_limiter = rate_limiter
        if _rate_limiter is None:
            _rate_limiter = getattr(self_obj, "rate_limiter", None)

        if _rate_limiter is None:
            return await call_partial(inp)
        return await _rate_limiter.process_input(func_partial=call_partial, inp=inp)

    return wrapper


@overload
def limit_rate_chunked(
    call: ProcessorCallableList[T, P, R],
    rate_limiter: RateLimiterC[T, R] | None = None,
    no_tqdm: bool | None = None,
) -> ProcessorCallableList[T, P, R]: ...


@overload
def limit_rate_chunked(
    call: None = None,
    rate_limiter: RateLimiterC[T, R] | None = None,
    no_tqdm: bool | None = None,
) -> RateLimWrapperWithArgsList[T, P, R]: ...


def limit_rate_chunked(
    call: ProcessorCallableList[T, P, R] | None = None,
    rate_limiter: RateLimiterC[T, R] | None = None,
    no_tqdm: bool | None = None,
) -> ProcessorCallableList[T, P, R] | RateLimWrapperWithArgsList[T, P, R]:
    if call is None:
        return functools.partial(
            limit_rate_chunked, rate_limiter=rate_limiter, no_tqdm=no_tqdm
        )  # type: ignore

    @functools.wraps(call)  # type: ignore
    async def wrapper(*args: Any, **kwargs: Any) -> list[R]:
        assert call is not None

        self_obj, inputs, other_args = split_pos_args(call, args)
        call_partial = partial_processor_callable(call, self_obj, *other_args, **kwargs)

        _no_tqdm = no_tqdm
        _rate_limiter = rate_limiter
        if _no_tqdm is None:
            _no_tqdm = getattr(self_obj, "no_tqdm", False)
        if _rate_limiter is None:
            _rate_limiter = getattr(self_obj, "rate_limiter", None)

        if _rate_limiter is None:
            return await asyncio_gather_with_pbar(
                *[call_partial(inp) for inp in inputs], no_tqdm=_no_tqdm
            )
        return await _rate_limiter.process_inputs(
            func_partial=call_partial,  # type: ignore
            inputs=inputs,
            no_tqdm=_no_tqdm,
        )

    return wrapper
