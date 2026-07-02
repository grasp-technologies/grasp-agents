"""
Mailbox transports as runtime ``Transport[TeamMessage]`` implementations: post /
consume / ack round-trip, ordering, isolation, payloads, shutdown, and the default
resolution. One transport seam — no separate mailbox type, no adapter.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from grasp_agents.agent_team.tools import default_transport
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.mailbox import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
)
from grasp_agents.runtime import CLOSED, Transport
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import InputText
from grasp_agents.types.message import TeamMessage


@pytest.fixture(params=["in_memory", "checkpoint"])
def transport(request: pytest.FixtureRequest) -> Transport[TeamMessage]:
    if request.param == "checkpoint":
        return CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    return InMemoryMailboxTransport()


@pytest.mark.asyncio
async def test_post_consume_ack_roundtrip(transport: Transport[TeamMessage]) -> None:
    assert await transport.has_pending("bob") is False

    await transport.post(TeamMessage.from_text(sender="alice", to="bob", text="hi"))
    assert await transport.has_pending("bob") is True

    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert msg.sender == "alice"
    assert msg.text == "hi"
    assert await transport.has_pending("bob") is True  # consume must not remove it

    await transport.ack("bob", msg)
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_consume_orders_oldest_first(transport: Transport[TeamMessage]) -> None:
    # Validates the priority-then-id *sort*, with ids supplied already in order. It
    # does NOT prove real ``_new_message_id`` ordering is strict — same-instant ids
    # tie on their random suffix (ordering is best-effort by design).
    for i in range(3):
        await transport.post(
            TeamMessage.from_text(
                sender="a", to="bob", text=f"m{i}", message_id=f"{i:04d}-x"
            )
        )
    # One group per call, oldest first; ack to advance to the next.
    got: list[str] = []
    for _ in range(3):
        msg = await transport.consume("bob")
        assert isinstance(msg, TeamMessage)
        got.append(msg.text)
        await transport.ack("bob", msg)
    assert got == ["m0", "m1", "m2"]
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_mailboxes_are_isolated_per_recipient(
    transport: Transport[TeamMessage],
) -> None:
    await transport.post(TeamMessage.from_text(sender="a", to="bob", text="for bob"))
    assert await transport.has_pending("bob") is True
    assert await transport.has_pending("carol") is False


@pytest.mark.asyncio
async def test_message_payloads_roundtrip(transport: Transport[TeamMessage]) -> None:
    await transport.post(
        TeamMessage(sender="a", routing=[["bob"]], payloads=[InputText(text="hi")])
    )
    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert isinstance(msg.payloads[0], InputText)
    assert msg.text == "hi"


@pytest.mark.asyncio
async def test_checkpoint_transport_marks_processed_after_ack() -> None:
    # The durable transport records an acked message under ``processed/`` so a
    # redelivery (crash inside ack) can be deduped — at-most-once on top of
    # at-least-once delivery.
    transport = CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await transport.post(msg)
    assert await transport.was_processed("bob", msg.message_id) is False

    consumed = await transport.consume("bob")
    assert isinstance(consumed, TeamMessage)
    # Consumed but not acked yet: not deduped, so a redelivery still re-runs.
    assert await transport.was_processed("bob", msg.message_id) is False

    await transport.ack("bob", consumed)
    assert await transport.was_processed("bob", msg.message_id) is True


@pytest.mark.asyncio
async def test_in_memory_transport_never_marks_processed() -> None:
    # Ephemeral mailboxes lose everything on a crash, so there is no redelivery
    # to dedupe — ``was_processed`` is always False (the base-class default).
    transport = InMemoryMailboxTransport()
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await transport.post(msg)
    consumed = await transport.consume("bob")
    assert isinstance(consumed, TeamMessage)
    await transport.ack("bob", consumed)
    assert await transport.was_processed("bob", msg.message_id) is False


async def _deliver(transport: CheckpointMailboxTransport, msg: TeamMessage) -> None:
    """Post → consume → ack, leaving a ``processed/`` record for ``msg``."""
    await transport.post(msg)
    consumed = await transport.consume(msg.routing[0][0])
    assert isinstance(consumed, TeamMessage)
    await transport.ack(consumed.routing[0][0], consumed)


@pytest.mark.asyncio
async def test_prune_processed_reclaims_old_records() -> None:
    # A delivered message's processed/ record is reclaimable once it is past the
    # retention window; a fresh one is kept (its dedup guard may still be needed).
    transport = CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await _deliver(transport, msg)
    assert await transport.was_processed("bob", msg.message_id) is True

    # Within the window: nothing reclaimed.
    assert await transport.prune_processed(older_than=timedelta(hours=1)) == 0
    assert await transport.was_processed("bob", msg.message_id) is True

    # Past the window: reclaimed, and the dedup guard goes with it.
    assert await transport.prune_processed(older_than=timedelta(seconds=-1)) == 1
    assert await transport.was_processed("bob", msg.message_id) is False


@pytest.mark.asyncio
async def test_prune_processed_keeps_still_deliverable() -> None:
    # The crash window: a message lingers in inbox/ (the ack's delete had not
    # landed) while its processed/ copy exists. GC must not reap the guard a
    # redelivery still needs — even past the retention window.
    transport = CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await _deliver(transport, msg)
    await transport.post(msg)  # redelivered: same id back in inbox/

    assert await transport.prune_processed(older_than=timedelta(seconds=-1)) == 0
    assert await transport.was_processed("bob", msg.message_id) is True


@pytest.mark.asyncio
async def test_prune_processed_pins_kept_ids() -> None:
    # A pinned id (e.g. a permanent entry seed) survives GC regardless of age.
    transport = CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    seed = TeamMessage.from_text(sender="a", to="bob", text="seed", message_id="SEED-1")
    other = TeamMessage.from_text(sender="a", to="bob", text="x", message_id="OTHER-1")
    await _deliver(transport, seed)
    await _deliver(transport, other)

    pruned = await transport.prune_processed(
        older_than=timedelta(seconds=-1), keep=lambda mid: mid.startswith("SEED-")
    )
    assert pruned == 1
    assert await transport.was_processed("bob", "SEED-1") is True
    assert await transport.was_processed("bob", "OTHER-1") is False


@pytest.mark.asyncio
async def test_prune_processed_reaps_stale_corrupt() -> None:
    # A dead-lettered corrupt record is reclaimed once past the corrupt window and
    # kept within it (folded into the same sweep as processed/ records).
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="x", message_id="0000-bad")
    await transport.post(msg)
    inbox = await store.list_keys("s/mailbox/bob/inbox/")
    await store.save(inbox[0], b"{ not json")
    assert (
        await transport._fetch_next("bob") is None
    )  # dead-letters the bad record  # pyright: ignore[reportPrivateUsage]
    assert any(
        k.endswith("0000-bad") for k in await store.list_keys("s/mailbox/bob/corrupt/")
    )

    # Within the corrupt window: kept.
    assert (
        await transport.prune_processed(
            older_than=timedelta(hours=1), corrupt_older_than=timedelta(hours=1)
        )
        == 0
    )
    # Past it: reaped.
    assert (
        await transport.prune_processed(
            older_than=timedelta(hours=1), corrupt_older_than=timedelta(seconds=-1)
        )
        == 1
    )
    assert not await store.list_keys("s/mailbox/bob/corrupt/")


@pytest.mark.asyncio
async def test_corrupt_inbox_record_is_dead_lettered_not_wedged() -> None:
    # A torn inbox record must not wedge the mailbox: the scan dead-letters it
    # (preserving it under corrupt/) and continues to the next deliverable message.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    good = TeamMessage.from_text(sender="a", to="bob", text="ok", message_id="9999-ok")
    bad = TeamMessage.from_text(sender="a", to="bob", text="x", message_id="0000-bad")
    await transport.post(good)
    await transport.post(bad)

    # Corrupt the record that sorts first (``0000-bad``) in place.
    inbox = await store.list_keys("s/mailbox/bob/inbox/")
    bad_key = next(k for k in inbox if k.endswith(bad.message_id))
    await store.save(bad_key, b"{ not valid json")

    # consume skips past the corrupt one and returns the good message.
    msg = await transport.consume("bob")
    assert isinstance(msg, TeamMessage)
    assert msg.message_id == good.message_id

    # The corrupt record is gone from inbox/ (unwedged) and preserved under corrupt/.
    assert all(
        not k.endswith(bad.message_id)
        for k in await store.list_keys("s/mailbox/bob/inbox/")
    )
    corrupt = await store.list_keys("s/mailbox/bob/corrupt/")
    assert any(k.endswith(bad.message_id) for k in corrupt)


@pytest.mark.asyncio
async def test_schema_mismatch_record_raises_not_dead_lettered() -> None:
    # A version we cannot interpret is a loud failure, not corruption: it must
    # propagate rather than be silently dead-lettered.
    import json

    from grasp_agents.durability.checkpoints import CheckpointSchemaError

    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="x", message_id="0000-v")
    await transport.post(msg)
    inbox = await store.list_keys("s/mailbox/bob/inbox/")
    # Take the real (structurally valid) record and bump only its version, so the
    # model validator — not field validation — is what rejects it.
    raw = await store.load(inbox[0])
    assert raw is not None
    bumped = json.loads(raw)
    bumped["schema_version"] = 999999
    await store.save(inbox[0], json.dumps(bumped).encode())
    with pytest.raises(CheckpointSchemaError):
        await transport.consume("bob")


@pytest.mark.asyncio
async def test_shutdown_unblocks_consume(transport: Transport[TeamMessage]) -> None:
    # An empty mailbox blocks consume; shutdown must release it with CLOSED.
    await transport.shutdown()
    assert await transport.consume("bob") is CLOSED


def test_default_transport_requires_checkpoint_store() -> None:
    with pytest.raises(ValueError, match="checkpoint_store"):
        default_transport(SessionContext[None](state=None))


def test_default_transport_uses_checkpoint_store() -> None:
    ctx = SessionContext[None](state=None, checkpoint_store=InMemoryCheckpointStore())
    assert isinstance(default_transport(ctx), CheckpointMailboxTransport)
