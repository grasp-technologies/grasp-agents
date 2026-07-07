"""
Standalone tests for the actor runtime (``grasp_agents.runtime``), exercised with
trivial string actors — no LLM, no ``Processor`` — so the engine is verified in
isolation from either frontend.

Covers the three termination policies (terminal / quiescence / daemon), the
single-drainer + one-group-per-turn delivery, error propagation, the bounded
in-process queue, and quiescence detection over a mailbox-style transport.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from grasp_agents.runtime import ActorDriver, InProcessTransport
from grasp_agents.runtime.transport import CLOSED, MAX_QUEUE_SIZE, Closed, Transport


@dataclass(frozen=True)
class Msg:
    """A minimal single-destination envelope."""

    destination: str | None
    text: str
    msg_id: int = field(default=0)


# --------------------------------------------------------------------------- #
# A mailbox-style transport test double: ``has_pending`` stays true until ack,
# which is what makes quiescence detection correct (mirrors a real mailbox).
# --------------------------------------------------------------------------- #


class FakeMailbox(Transport[Msg]):
    def __init__(self, poll: float = 0.005) -> None:
        super().__init__()
        self._boxes: dict[str, list[Msg]] = {}
        self._poll = poll
        self._closed = asyncio.Event()

    def register(self, recipient: str) -> None:
        self._boxes.setdefault(recipient, [])

    async def post(self, envelope: Msg) -> None:
        if envelope.destination is not None:
            self._boxes.setdefault(envelope.destination, []).append(envelope)

    async def consume(self, recipient: str) -> Msg | Closed:
        while not self._closed.is_set():
            box = self._boxes.get(recipient)
            if box:
                return box[0]
            await asyncio.sleep(self._poll)
        return CLOSED

    async def ack(self, recipient: str, envelope: Msg) -> None:
        box = self._boxes.get(recipient)
        if box:
            self._boxes[recipient] = [m for m in box if m.msg_id != envelope.msg_id]

    async def has_pending(self, recipient: str) -> bool:
        return bool(self._boxes.get(recipient))

    async def shutdown(self) -> None:
        self._closed.set()


# --------------------------------------------------------------------------- #
# Terminal mode (the orchestration shape): chain to a final result.
# --------------------------------------------------------------------------- #


async def _run_terminal(
    handlers: dict[str, Callable[[ActorDriver[Msg], Msg], object]],
    seed: Msg,
) -> tuple[object, list[object]]:
    """Drive a terminal-mode run and return (final_result, streamed_events)."""
    driver: ActorDriver[Msg] = ActorDriver(
        InProcessTransport[Msg](), termination="terminal"
    )
    streamed: list[object] = []

    def make(name: str) -> Callable[[Msg], object]:
        async def handler(envelope: Msg) -> None:
            await handlers[name](driver, envelope)

        return handler

    async with driver:
        for name in handlers:
            driver.register_handler(name, make(name))
        await driver.post(seed)
        async for event in driver.stream_events():
            streamed.append(event)

    return await driver.final_result(), streamed


@pytest.mark.asyncio
async def test_terminal_linear_chain_returns_final_result() -> None:
    async def a(driver: ActorDriver[Msg], msg: Msg) -> None:
        await driver.push_to_stream(f"A:{msg.text}")
        await driver.post(Msg(destination="B", text=f"{msg.text}->A"))

    async def b(driver: ActorDriver[Msg], msg: Msg) -> None:
        await driver.push_to_stream(f"B:{msg.text}")
        await driver.finalize(f"{msg.text}->B")

    result, streamed = await _run_terminal(
        {"A": a, "B": b}, Msg(destination="A", text="s")
    )
    assert result == "s->A->B"
    assert streamed == ["A:s", "B:s->A"]


@pytest.mark.asyncio
async def test_terminal_fan_out_runs_recipients_concurrently() -> None:
    started = asyncio.Event()
    both_in_flight = asyncio.Event()
    seen: list[str] = []

    async def a(driver: ActorDriver[Msg], msg: Msg) -> None:
        del msg
        await driver.post(Msg(destination="B", text="x", msg_id=1))
        await driver.post(Msg(destination="C", text="y", msg_id=2))

    async def slow(driver: ActorDriver[Msg], msg: Msg) -> None:
        seen.append(msg.text)
        if len(seen) == 1:
            started.set()
            # Park until the sibling is also in flight — proves concurrency.
            await asyncio.wait_for(both_in_flight.wait(), timeout=1.0)
        else:
            both_in_flight.set()
        await driver.post(Msg(destination="D", text=msg.text))

    async def d(driver: ActorDriver[Msg], msg: Msg) -> None:
        if len(seen) == 2:
            await driver.finalize("done")

    result, _ = await _run_terminal(
        {"A": a, "B": slow, "C": slow, "D": d}, Msg(destination="A", text="go")
    )
    assert result == "done"
    assert sorted(seen) == ["x", "y"]


@pytest.mark.asyncio
async def test_terminal_handler_error_propagates_and_retrieves_future() -> None:
    async def boom(driver: ActorDriver[Msg], msg: Msg) -> None:
        del driver, msg
        raise RuntimeError("handler boom")

    with pytest.raises(BaseExceptionGroup):
        await _run_terminal({"A": boom}, Msg(destination="A", text="s"))


@pytest.mark.asyncio
async def test_terminal_crashing_handler_future_is_retrieved() -> None:
    # Mirrors the event-bus contract: the crash arrives via the TaskGroup, and the
    # final-result future's identical exception is retrieved on exit so a GC'd
    # future can't trip asyncio's "exception was never retrieved" warning.
    driver: ActorDriver[Msg] = ActorDriver(
        InProcessTransport[Msg](), termination="terminal"
    )

    async def boom(envelope: Msg) -> None:
        del envelope
        raise RuntimeError("boom")

    async def drive() -> None:
        async with driver:
            driver.register_handler("p", boom)
            await driver.post(Msg(destination="p", text="x"))
            async for _ in driver.stream_events():
                pass

    with pytest.raises(BaseExceptionGroup):
        await drive()

    fut = driver._final_result_fut
    assert fut is not None
    assert fut.done()
    assert fut._log_traceback is False
    assert isinstance(fut.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_teardown_does_not_hang_on_producer_parked_on_full_stream() -> None:
    # A handler floods the stream past its bound (MAX_QUEUE_SIZE) with nothing
    # draining it, so it parks inside push_to_stream. The run then shuts down on the
    # clean path (finalize). Teardown must wake the parked producer and complete —
    # without the shutdown-aware push, __aexit__ would wait on it forever.
    transport = InProcessTransport[Msg]()
    driver: ActorDriver[Msg] = ActorDriver(transport, termination="terminal")

    async def flooder(envelope: Msg) -> None:
        del envelope
        for i in range(MAX_QUEUE_SIZE + 50):
            await driver.push_to_stream(f"e{i}")

    async def drive() -> None:
        async with driver:
            driver.register_handler("flood", flooder)
            await driver.post(Msg(destination="flood", text="go"))
            # Let the flooder fill the (undrained) stream queue and park, then shut
            # down — never draining stream_events (a consumer that stopped pulling).
            await asyncio.sleep(0.05)
            await driver.finalize("done")

    await asyncio.wait_for(drive(), timeout=5.0)
    assert await driver.final_result() == "done"


# --------------------------------------------------------------------------- #
# Quiescence mode (bounded peer collaboration).
# --------------------------------------------------------------------------- #


async def _run_quiescence(
    handlers: dict[str, Callable[[ActorDriver[Msg], Msg], object]],
    seeds: list[Msg],
    *,
    max_activations: int | None = None,
) -> ActorDriver[Msg]:
    transport = FakeMailbox()
    driver: ActorDriver[Msg] = ActorDriver(
        transport, termination="quiescence", max_activations=max_activations
    )

    def make(name: str) -> Callable[[Msg], object]:
        async def handler(envelope: Msg) -> None:
            await handlers[name](driver, envelope)

        return handler

    async with driver:
        for name in handlers:
            driver.register_handler(name, make(name))
        for seed in seeds:
            await driver.post(seed)
        await driver.settle()
        async for _ in driver.stream_events():
            pass

    return driver


@pytest.mark.asyncio
async def test_quiescence_ping_pong_stops_when_idle() -> None:
    counts = {"alice": 0, "bob": 0}

    async def alice(driver: ActorDriver[Msg], msg: Msg) -> None:
        counts["alice"] += 1
        if msg.text == "kick":
            await driver.post(Msg(destination="bob", text="ping", msg_id=10))
        # On bob's reply, alice sends nothing → the team goes idle.

    async def bob(driver: ActorDriver[Msg], msg: Msg) -> None:
        counts["bob"] += 1
        await driver.post(Msg(destination="alice", text="pong", msg_id=20))

    driver = await _run_quiescence(
        {"alice": alice, "bob": bob}, [Msg(destination="alice", text="kick", msg_id=1)]
    )
    assert counts == {"alice": 2, "bob": 1}
    assert driver.activation_count == 3


@pytest.mark.asyncio
async def test_quiescence_one_group_per_turn() -> None:
    seen: list[str] = []

    async def solo(driver: ActorDriver[Msg], msg: Msg) -> None:
        del driver
        seen.append(msg.text)

    driver = await _run_quiescence(
        {"solo": solo},
        [
            Msg(destination="solo", text="first", msg_id=1),
            Msg(destination="solo", text="second", msg_id=2),
        ],
    )
    # Two pending messages → two separate activations, never one merged turn.
    assert seen == ["first", "second"]
    assert driver.activation_count == 2


@pytest.mark.asyncio
async def test_quiescence_empty_seed_stops_immediately() -> None:
    async def solo(driver: ActorDriver[Msg], msg: Msg) -> None:
        del driver, msg

    driver = await asyncio.wait_for(
        _run_quiescence({"solo": solo}, []), timeout=1.0
    )
    assert driver.activation_count == 0


@pytest.mark.asyncio
async def test_quiescence_max_activations_bounds_run() -> None:
    counts = {"alice": 0, "bob": 0}

    async def alice(driver: ActorDriver[Msg], msg: Msg) -> None:
        del msg
        counts["alice"] += 1
        await driver.post(Msg(destination="bob", text="ping", msg_id=counts["alice"]))
        # Widen the window: bob's eager consumer has ample time to fetch "ping"
        # before the budget shutdown fires — so bob's NOT running proves the budget
        # gate blocks the activation, not lucky timing.
        await asyncio.sleep(0.05)

    async def bob(driver: ActorDriver[Msg], msg: Msg) -> None:
        del msg
        counts["bob"] += 1
        await driver.post(Msg(destination="alice", text="pong", msg_id=100))

    driver = await _run_quiescence(
        {"alice": alice, "bob": bob},
        [Msg(destination="alice", text="kick", msg_id=1)],
        max_activations=1,
    )
    # Budget of 1 → only alice's first activation; bob never runs.
    assert driver.activation_count == 1
    assert counts == {"alice": 1, "bob": 0}


# --------------------------------------------------------------------------- #
# Daemon mode (never self-terminates).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_daemon_keeps_running_past_quiescence() -> None:
    transport = FakeMailbox()
    driver: ActorDriver[Msg] = ActorDriver(transport, termination="daemon")
    handled: list[str] = []

    async def solo(envelope: Msg) -> None:
        handled.append(envelope.text)

    async def drain() -> None:
        async with driver:
            driver.register_handler("solo", solo)
            await driver.settle()
            async for _ in driver.stream_events():
                pass

    async def until(pred: Callable[[], bool]) -> None:
        for _ in range(300):
            if pred():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("condition not met within timeout")

    consumer = asyncio.create_task(drain())
    try:
        await transport.post(Msg(destination="solo", text="first", msg_id=1))
        await until(lambda: handled == ["first"])  # handled; would quiesce if bounded
        await transport.post(Msg(destination="solo", text="second", msg_id=2))
        await until(lambda: handled == ["first", "second"])  # daemon kept running
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer


# --------------------------------------------------------------------------- #
# In-process transport: bounded queue + backpressure.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_in_process_transport_queue_is_bounded() -> None:
    transport = InProcessTransport[Msg]()
    transport.register("x")
    assert transport._queues["x"].maxsize == MAX_QUEUE_SIZE


@pytest.mark.asyncio
async def test_in_process_transport_has_pending() -> None:
    transport = InProcessTransport[Msg]()
    transport.register("x")
    assert not await transport.has_pending("x")
    await transport.post(Msg(destination="x", text="m"))
    assert await transport.has_pending("x")
    got = await transport.consume("x")
    assert got is not None
    assert got.text == "m"
    assert not await transport.has_pending("x")
