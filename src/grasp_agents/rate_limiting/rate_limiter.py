import asyncio
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Generic, overload

from .types import AsyncCallable, P, R
from .utils import split_pos_args

logger = logging.getLogger(__name__)


MAX_RPM = 1e10

RateLimDecorator = Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]


@dataclass
class RateLimiterState:
    next_request_time: float = 0.0


class RateLimiter(Generic[R]):
    def __init__(self, rpm: float, max_concurrency: int = 200):
        self._rpm = rpm
        self._max_concurrency = max_concurrency

        self._lock = asyncio.Lock()
        self._state = RateLimiterState(next_request_time=0.0)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    async def process(self, func_partial: AsyncCallable[..., R]) -> R:
        async with self._semaphore:
            async with self._lock:
                now = monotonic()
                if now < self._state.next_request_time:
                    await asyncio.sleep(self._state.next_request_time - now)
                self._state.next_request_time = monotonic() + 1.01 * 60.0 / self._rpm
            return await func_partial()

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
    def state(self) -> RateLimiterState:
        return self._state


@overload
def limit_rate(
    call: AsyncCallable[P, R], rate_limiter: RateLimiter[R] | None = None
) -> AsyncCallable[P, R]: ...


@overload
def limit_rate(
    call: None = None, rate_limiter: RateLimiter[R] | None = None
) -> RateLimDecorator[P, R]: ...


def limit_rate(
    call: AsyncCallable[P, R] | None = None, rate_limiter: RateLimiter[R] | None = None
) -> AsyncCallable[P, R] | RateLimDecorator[P, R]:
    if call is None:
        return functools.partial(limit_rate, rate_limiter=rate_limiter)

    @functools.wraps(call)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self_obj, _ = split_pos_args(call, args)
        call_partial = partial(call, *args, **kwargs)
        _rate_limiter = rate_limiter or getattr(self_obj, "rate_limiter", None)
        if _rate_limiter is None:
            return await call_partial()

        return await _rate_limiter.process(func_partial=call_partial)

    return wrapper
