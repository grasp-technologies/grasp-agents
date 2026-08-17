from collections.abc import AsyncIterator
from typing import Any

import pytest

from grasp_agents.processors.processor import Processor
from grasp_agents.types.errors import ProcRunError
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    ProcStreamingErrorEvent,
)


class _FlakyProcessor(Processor[str, str, None]):
    def __init__(self, name: str, *, fail_times: int, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self._fail_times = fail_times
        self.calls = 0

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ValueError(f"boom {self.calls}")
        for inp in in_args or []:
            yield ProcPayloadOutEvent(data=f"{inp}!", source=self.name, exec_id=exec_id)


@pytest.mark.asyncio
async def test_retry_error_events_carry_generated_exec_id() -> None:
    proc = _FlakyProcessor("flaky", fail_times=1, max_retries=1)

    events = [event async for event in proc.run_stream(in_args="hi")]

    error_events = [e for e in events if isinstance(e, ProcStreamingErrorEvent)]
    packet_events = [e for e in events if isinstance(e, ProcPacketOutEvent)]
    assert len(error_events) == 1
    assert len(packet_events) == 1
    assert error_events[0].exec_id is not None
    assert error_events[0].exec_id == packet_events[0].exec_id
    assert error_events[0].data.exec_id == packet_events[0].exec_id


@pytest.mark.asyncio
async def test_exhausted_retries_share_generated_exec_id() -> None:
    proc = _FlakyProcessor("doomed", fail_times=99, max_retries=1)

    error_events: list[ProcStreamingErrorEvent] = []

    async def consume() -> None:
        async for event in proc.run_stream(in_args="hi"):
            if isinstance(event, ProcStreamingErrorEvent):
                error_events.append(event)

    with pytest.raises(ProcRunError) as exc_info:
        await consume()

    assert exc_info.value.exec_id is not None
    exec_ids = {e.exec_id for e in error_events}
    assert exec_ids == {exc_info.value.exec_id}
