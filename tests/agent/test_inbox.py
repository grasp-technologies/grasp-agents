"""
The per-agent :class:`AgentInbox` as a view over a pluggable
:class:`MessageTransport`: in-memory FIFO, the parked-idle signal, and durable
delivery (the same message survives a restart over a shared checkpoint store, and a
consumed one is not re-delivered), and acks that commit inside the checkpoint's
transaction when the mailbox lives in that same store.
"""

from __future__ import annotations

import pytest

from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.inbox import AgentInbox
from grasp_agents.mailbox import CheckpointMailboxTransport, InMemoryMailboxTransport
from grasp_agents.runtime import Transport
from grasp_agents.types.message import CONTROL_PRIORITY, TeamMessage
from tests._helpers import _text_response
from tests.agent.test_human_turn_anchors import _human, _run_resident_until
from tests.durability.test_deferred_effects import _RollbackStore, _Store
from tests.durability.test_sessions import _make_agent

pytestmark = pytest.mark.asyncio


def _msg(text: str, *, to: str = "curator") -> TeamMessage:
    return TeamMessage.from_text(sender="user", to=to, text=text)


async def test_in_memory_fifo_and_waiting_signal() -> None:
    inbox = AgentInbox(recipient="curator")
    assert not await inbox.has_pending()
    assert not inbox.is_waiting

    await inbox.post(_msg("a"))
    await inbox.post(_msg("b"))
    assert await inbox.has_pending()

    first = await inbox.poll()
    second = await inbox.poll()
    assert first is not None
    assert first.text == "a"
    assert second is not None
    assert second.text == "b"
    assert await inbox.poll() is None
    assert not await inbox.has_pending()

    with inbox.waiting():
        assert inbox.is_waiting
    assert not inbox.is_waiting


async def test_shared_transport_routes_to_the_right_recipient() -> None:
    # One transport, two members' views — a message addressed to alice lands only
    # in alice's view (the same seam serves every member).
    transport = InMemoryMailboxTransport()
    alice = AgentInbox(transport=transport, recipient="alice")
    bob = AgentInbox(transport=transport, recipient="bob")

    await alice.post(_msg("for bob", to="bob"))
    assert not await alice.has_pending()
    assert await bob.has_pending()
    got = await bob.poll()
    assert got is not None
    assert got.text == "for bob"


async def test_durable_transport_survives_restart() -> None:
    store = InMemoryCheckpointStore()

    producer = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    await producer.post(_msg("first"))
    await producer.post(_msg("second"))

    # A fresh inbox over a fresh transport on the same store (a restart) sees both.
    resumed = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    texts = []
    while (msg := await resumed.poll()) is not None:
        texts.append(msg.text)
    assert texts == ["first", "second"]


async def test_take_does_not_ack_until_released() -> None:
    # ``take`` consumes without acking: the message stays pending until the
    # caller releases it, so a crash before then re-delivers rather than drops it.
    inbox = AgentInbox(recipient="curator")
    await inbox.post(_msg("hi"))

    taken = await inbox.take()
    assert taken is not None
    assert taken.text == "hi"
    assert await inbox.has_pending()  # still pending — not yet acked

    await inbox.ack(taken)
    assert not await inbox.has_pending()


async def test_take_skips_already_processed_redelivery() -> None:
    # A re-delivery whose ack only partly landed (the message lingers in the
    # inbox but is already marked processed) is acked-and-skipped, not re-run.
    store = InMemoryCheckpointStore()
    inbox = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    done = _msg("done")
    await inbox.post(done)
    taken = await inbox.take()
    assert taken is not None
    await inbox.ack(taken)  # writes processed/<id>

    # The same message lands back in the inbox (a crash inside ack), plus a new one.
    await inbox.post(done)
    await inbox.post(_msg("fresh"))
    got = await inbox.take()
    assert got is not None
    assert got.text == "fresh"  # "done" was recognized as processed and skipped
    await inbox.ack(got)
    assert not await inbox.has_pending()


async def test_control_priority_drains_before_earlier_normal_message() -> None:
    # A control-plane (priority) message drains ahead of an earlier-queued normal
    # one — in both transports — so human input / wakeups preempt peer mail.
    transports: list[Transport[TeamMessage]] = [
        InMemoryMailboxTransport(),
        CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s"),
    ]
    for transport in transports:
        inbox = AgentInbox(transport=transport, recipient="curator")
        await inbox.post(_msg("normal"))
        await inbox.post(
            TeamMessage.from_text(
                sender="user", to="curator", text="urgent", priority=CONTROL_PRIORITY
            )
        )
        first = await inbox.poll()
        second = await inbox.poll()
        assert first is not None
        assert first.text == "urgent"
        assert second is not None
        assert second.text == "normal"
        assert await inbox.poll() is None


async def test_durable_consume_is_not_redelivered_after_restart() -> None:
    store = InMemoryCheckpointStore()
    inbox = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    await inbox.post(_msg("once"))
    await inbox.post(_msg("twice"))
    consumed = await inbox.poll()
    assert consumed is not None
    assert consumed.text == "once"

    resumed = AgentInbox(
        transport=CheckpointMailboxTransport(store, session_key="s"),
        recipient="curator",
    )
    remaining = []
    while (msg := await resumed.poll()) is not None:
        remaining.append(msg.text)
    assert remaining == ["twice"]


async def test_take_mints_consumption_seq_and_restore_never_lowers() -> None:
    # The inbox mints ``TeamMessage.seq`` at take (it is the recipient's sole
    # consumer, so take order is absorption order); the counter lives on the
    # shared transport, so it survives the per-run inbox, and a restored
    # watermark only ever seeds it forward — burned seqs stay burned.
    transport = InMemoryMailboxTransport()
    inbox = AgentInbox(transport=transport, recipient="curator")
    await inbox.post(_msg("a"))
    await inbox.post(_msg("b"))

    first = await inbox.poll()
    second = await inbox.poll()
    assert first is not None
    assert second is not None
    assert (first.seq, second.seq) == (1, 2)
    assert inbox.last_consumption_seq == 2

    inbox.seed_consumption_seq(0)
    assert inbox.last_consumption_seq == 2
    inbox.seed_consumption_seq(9)

    # A fresh per-run inbox over the same transport keeps counting from there.
    reattached = AgentInbox(transport=transport, recipient="curator")
    await reattached.post(_msg("c"))
    third = await reattached.poll()
    assert third is not None
    assert third.seq == 10


# --- Acks and the checkpoint transaction ---


def _is_ack_write(key: str) -> bool:
    return "processed" in key.split("/")


class _AckRecordingStore(_Store):
    """Records, for each mailbox ack, whether it landed inside a transaction."""

    def __init__(self) -> None:
        super().__init__()
        self.ack_writes: list[bool] = []

    async def save(self, key: str, data: bytes) -> None:
        if _is_ack_write(key):
            self.ack_writes.append(self.in_transaction)
        await super().save(key, data)


class _AckRecordingRollbackStore(_RollbackStore):
    """The transactional counterpart of :class:`_AckRecordingStore`."""

    def __init__(self) -> None:
        super().__init__()
        self.ack_writes: list[bool] = []

    async def save(self, key: str, data: bytes) -> None:
        if _is_ack_write(key):
            self.ack_writes.append(self.in_transaction)
        await super().save(key, data)


async def test_acks_commit_inside_a_transactional_checkpoint() -> None:
    store = _AckRecordingRollbackStore()
    agent, ctx = _make_agent([_text_response("re: hi")], session_key="s1", store=store)
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent.attach_inbox()

    await _run_resident_until(agent, transport, [_human("hi")])

    assert store.ack_writes == [True]


async def test_acks_follow_a_non_transactional_checkpoint() -> None:
    store = _AckRecordingStore()
    agent, ctx = _make_agent([_text_response("re: hi")], session_key="s1", store=store)
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent.attach_inbox()

    await _run_resident_until(agent, transport, [_human("hi")])

    assert store.ack_writes == [False]


async def test_a_rolled_back_checkpoint_keeps_the_message_leased() -> None:
    store = _RollbackStore()
    agent, ctx = _make_agent([], session_key="s1", store=store)
    transport = CheckpointMailboxTransport(store, session_key="s1")
    ctx.transport = transport
    agent.attach_inbox()
    inbox = agent.agent_ctx.inbox
    assert inbox is not None

    message = _human("hi")
    await transport.post(message)
    assert await inbox.take() is not None

    store.fail_next_commit = True
    with pytest.raises(RuntimeError, match="commit failed"):
        await agent.save_checkpoint()
    # The ack rolled back with the checkpoint: still pending, and still leased,
    # so it is not taken a second time.
    assert await transport.has_pending("test_agent")
    assert not await transport.was_processed("test_agent", message.message_id)
    assert await inbox.take() is None

    await agent.save_checkpoint()
    assert await transport.was_processed("test_agent", message.message_id)
    assert not await transport.has_pending("test_agent")
    assert not inbox._leased  # pyright: ignore[reportPrivateUsage]
