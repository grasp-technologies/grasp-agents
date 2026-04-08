from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Generic, TypeVar

from grasp_agents.types.events import Event

logger = getLogger(__name__)


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class PumpError:
    """Records a failed pump in :class:`ConcurrentStream`."""

    index: int
    exception: BaseException


_QueueItem = tuple[int, _T] | PumpError | None


class ConcurrentStream(Generic[_T]):
    """
    Merges multiple async iterators, yielding ``(index, item)`` in arrival order.

    After iteration completes, :attr:`errors` contains one :class:`PumpError`
    per failed pump so callers can react (retry, inject error messages, etc.).
    """

    def __init__(self, generators: list[AsyncIterator[_T]]) -> None:
        self._generators = generators
        self._errors: list[PumpError] = []

    @property
    def errors(self) -> list[PumpError]:
        return self._errors

    @property
    def failed_indices(self) -> list[int]:
        return [e.index for e in self._errors]

    async def __aiter__(self) -> AsyncIterator[tuple[int, _T]]:
        generators = self._generators
        if not generators:
            return

        queue: asyncio.Queue[_QueueItem[_T]] = asyncio.Queue()
        pumps_left = len(generators)
        errors = self._errors

        async def pump(gen: AsyncIterator[_T], idx: int) -> None:
            nonlocal pumps_left
            try:
                async for item in gen:
                    await queue.put((idx, item))

            except asyncio.CancelledError:
                raise

            except Exception as e:
                logger.warning("stream_concurrent pump %d failed: %r", idx, e)
                await queue.put(PumpError(idx, e))

            finally:
                pumps_left -= 1
                if pumps_left == 0:
                    await queue.put(None)

        async with asyncio.TaskGroup() as tg:
            for idx, gen in enumerate(generators):
                tg.create_task(pump(gen, idx))

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                if isinstance(msg, PumpError):
                    errors.append(msg)
                    continue
                yield msg


def stream_concurrent(
    generators: list[AsyncIterator[_T]],
) -> ConcurrentStream[_T]:
    """
    Create a :class:`ConcurrentStream` that merges *generators*.

    Usage::

        stream = stream_concurrent(generators)
        async for idx, item in stream:
            ...
        for err in stream.errors:
            handle(err.index, err.exception)
    """
    return ConcurrentStream(generators)


_F = TypeVar("_F")


class MissingFinalEventError(RuntimeError):
    """Raised when the stream finishes without encountering the required final event type."""


class EventStream(AsyncIterator[Event[Any]], Generic[_F]):
    def __init__(
        self, source: AsyncIterable[Event[Any]], final_type: type[_F] = object
    ) -> None:
        self._aiter: AsyncIterator[Event[Any]] = source.__aiter__()
        self._final_type: type[_F] = final_type
        self._final_event: Event[_F]
        self._final_event_set: bool = False
        self._events: list[Event[Any]] = []

    @property
    def final_type(self) -> type[_F]:
        return self._final_type

    @property
    def events(self) -> list[Event[Any]]:
        return self._events

    def __aiter__(self) -> EventStream[_F]:
        return self

    async def __anext__(self) -> Event[Any]:
        event = await self._aiter.__anext__()
        if isinstance(event.data, self.final_type):
            self._final_event = event
            self._final_event_set = True

        return event

    async def drain(self) -> list[Event[Any]]:
        async for event in self:
            self._events.append(event)
        return self._events

    async def final_event(self) -> Event[_F]:
        async for _ in self:
            pass
        if not self._final_event_set:
            raise MissingFinalEventError(
                f"No event of type Event[{self.final_type.__name__}] was encountered."
            )
        return self._final_event

    async def final_data(self) -> _F:
        return (await self.final_event()).data
