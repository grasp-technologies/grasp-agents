from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from logging import getLogger
from typing import Any, Generic, TypeVar

from grasp_agents.typing.events import Event

logger = getLogger(__name__)


_T = TypeVar("_T")


async def stream_concurrent(
    generators: list[AsyncIterator[_T]],
) -> AsyncIterator[tuple[int, _T]]:
    tasks: list[asyncio.Task[None]] = []
    queue: asyncio.Queue[tuple[int, _T] | None] = asyncio.Queue()
    pumps_left = len(generators)

    async def pump(gen: AsyncIterator[_T], idx: int) -> None:
        nonlocal pumps_left
        try:
            async for item in gen:
                await queue.put((idx, item))

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.warning(f"stream_concurrent pump {idx} failed:\n{e!r}")

        finally:
            pumps_left -= 1
            if pumps_left == 0:
                await queue.put(None)

    async with asyncio.TaskGroup() as tg:
        for idx, gen in enumerate(generators):
            tasks.append(tg.create_task(pump(gen, idx)))

        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg


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
