"""
Mailbox transports as runtime ``Transport[TeamMessage]`` implementations: post /
consume / ack round-trip, ordering, isolation, payloads, shutdown, and the default
resolution. One transport seam — no separate mailbox type, no adapter.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.mailbox import (
    CheckpointMailboxTransport,
    InMemoryMailboxTransport,
    resolve_session_transport,
)
from grasp_agents.runtime import CLOSED, Transport
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import InputText
from grasp_agents.types.message import CONTROL_PRIORITY, LEAD_PRIORITY, TeamMessage


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
async def test_consume_orders_control_then_lead_then_peers(
    transport: Transport[TeamMessage],
) -> None:
    # The three priority tiers: control-plane mail (human input, rewind notices)
    # drains first, then the lead's mail, then ordinary peer messages — even when
    # posted in the opposite order (ids supplied already in arrival order).
    for i, (sender, priority) in enumerate(
        [("peer", 0), ("lead", LEAD_PRIORITY), ("user", CONTROL_PRIORITY)]
    ):
        await transport.post(
            TeamMessage.from_text(
                sender=sender,
                to="bob",
                text=sender,
                priority=priority,
                message_id=f"{i:04d}-x",
            )
        )
    got: list[str] = []
    for _ in range(3):
        msg = await transport.consume("bob")
        assert isinstance(msg, TeamMessage)
        got.append(msg.text)
        await transport.ack("bob", msg)
    assert got == ["user", "lead", "peer"]


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


async def _absorb(transport: Transport[TeamMessage], recipient: str) -> str:
    """Consume + mint the consumption seq + ack, as the recipient's inbox does."""
    consumed = await transport.consume(recipient)
    assert isinstance(consumed, TeamMessage)
    consumed.seq = transport.mint_consumption_seq(recipient)
    await transport.ack(recipient, consumed)
    return consumed.text


@pytest.mark.asyncio
async def test_unprocess_after_moves_consumed_suffix_back_to_pending(
    transport: Transport[TeamMessage],
) -> None:
    for i in range(3):
        await transport.post(
            TeamMessage.from_text(
                sender="a", to="bob", text=f"m{i}", message_id=f"{i:04d}-x"
            )
        )
    await _absorb(transport, "bob")
    boundary = transport.last_consumption_seq("bob")
    await _absorb(transport, "bob")
    await _absorb(transport, "bob")
    assert await transport.has_pending("bob") is False

    # Rewind to the boundary after the first absorption: m1 and m2 return to
    # pending (seq cleared for re-minting); m0 stays consumed.
    assert await transport.unprocess_after("bob", boundary) == 2
    assert await transport.has_pending("bob") is True
    # Re-absorptions mint fresh seqs above the burned ones, order preserved.
    assert [await _absorb(transport, "bob") for _ in range(2)] == ["m1", "m2"]
    assert transport.last_consumption_seq("bob") == 5

    # Nothing above the watermark → nothing to move.
    assert await transport.unprocess_after("bob", 10) == 0


@pytest.mark.asyncio
async def test_unprocess_after_uses_consumption_order_not_arrival(
    transport: Transport[TeamMessage],
) -> None:
    # Control mail drains out of arrival order, so "consumed after the
    # boundary" must follow the stamped consumption seq: the control message
    # arrived LAST but was absorbed FIRST (before the boundary), so it stays
    # consumed while the earlier-arrived normal message returns to pending.
    await transport.post(
        TeamMessage.from_text(sender="a", to="bob", text="normal", message_id="0001-x")
    )
    await transport.post(
        TeamMessage.from_text(
            sender="a",
            to="bob",
            text="control",
            message_id="0002-x",
            priority=CONTROL_PRIORITY,
        )
    )
    assert await _absorb(transport, "bob") == "control"
    boundary = transport.last_consumption_seq("bob")
    assert await _absorb(transport, "bob") == "normal"

    assert await transport.unprocess_after("bob", boundary) == 1
    assert await _absorb(transport, "bob") == "normal"
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_unprocess_after_skips_untracked_acks(
    transport: Transport[TeamMessage],
) -> None:
    # A triggered worker's driver acks without stamping a consumption seq;
    # such a message is not restorable (its redelivery is the orchestrator's
    # job) and must never bounce back into the mailbox.
    await transport.post(TeamMessage.from_text(sender="a", to="bob", text="hi"))
    consumed = await transport.consume("bob")
    assert isinstance(consumed, TeamMessage)
    await transport.ack("bob", consumed)

    assert await transport.unprocess_after("bob", 0) == 0
    assert await transport.has_pending("bob") is False


@pytest.mark.asyncio
async def test_unprocess_after_rerun_is_idempotent() -> None:
    # A crash mid-rollback is healed by retrying the rollback: re-running
    # unprocess for the same watermark finds the records already moved home.
    transport = CheckpointMailboxTransport(InMemoryCheckpointStore(), session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await transport.post(msg)
    await _absorb(transport, "bob")

    assert await transport.unprocess_after("bob", 0) == 1
    assert await transport.unprocess_after("bob", 0) == 0
    # The moved record is a normal pending message again — deliverable, and no
    # longer deduped as already-processed.
    assert await transport.was_processed("bob", msg.message_id) is False
    assert await _absorb(transport, "bob") == "hi"


@pytest.mark.asyncio
async def test_redelivery_dedupe_ack_does_not_clobber_seq() -> None:
    # A crash inside ack can leave both the processed record and the inbox
    # copy; the consumer's dedupe path then re-acks with an UNSTAMPED copy.
    # First ack wins: the stored consumption seq must survive, or a later
    # rollback would skip this message.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    msg = TeamMessage.from_text(sender="a", to="bob", text="hi")
    await transport.post(msg)
    consumed = await transport.consume("bob")
    assert isinstance(consumed, TeamMessage)
    consumed.seq = transport.mint_consumption_seq("bob")
    await transport.ack("bob", consumed)

    # Simulate the redelivery: the inbox copy lingers, gets re-acked unstamped.
    await transport.post(msg)
    await transport.ack("bob", msg)
    assert msg.seq == 0  # the redelivered copy really was unstamped

    assert await transport.unprocess_after("bob", 0) == 1  # seq survived


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


def test_resolve_session_transport_defaults_and_installs() -> None:
    # The session owns its one mailbox: resolution creates it on first use —
    # durable when the session has a store, else in-memory — installs it on
    # ``ctx.transport``, and every later resolution returns that same instance.
    ctx = SessionContext[None](state=None)
    resolved = resolve_session_transport(ctx)
    assert isinstance(resolved, InMemoryMailboxTransport)
    assert ctx.transport is resolved
    assert resolve_session_transport(ctx) is resolved

    durable_ctx = SessionContext[None](
        state=None, checkpoint_store=InMemoryCheckpointStore()
    )
    resolved_durable = resolve_session_transport(durable_ctx)
    assert isinstance(resolved_durable, CheckpointMailboxTransport)


@pytest.mark.asyncio
async def test_resolve_warns_on_durability_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Mail safety needs the mailbox and the session state to share a
    # durability fate; a hand-set ctx.transport can break the pairing, so
    # resolution warns — once per transport, in either direction.
    with caplog.at_level(logging.WARNING, logger="grasp_agents.mailbox"):
        # Durable mailbox on a storeless session: consumed mail's handling
        # would evaporate with the process, unredelivered.
        ctx = SessionContext[None](state=None)
        ctx.transport = CheckpointMailboxTransport(
            InMemoryCheckpointStore(), session_key="s1"
        )
        assert resolve_session_transport(ctx) is ctx.transport
        assert resolve_session_transport(ctx) is ctx.transport

        # Ephemeral mailbox on a durable session: pending mail would die
        # while the session resumes without it.
        durable_ctx = SessionContext[None](
            state=None, checkpoint_store=InMemoryCheckpointStore()
        )
        durable_ctx.transport = InMemoryMailboxTransport()
        resolve_session_transport(durable_ctx)

        # Aligned, hand-set configs stay silent.
        aligned = SessionContext[None](state=None)
        aligned.transport = InMemoryMailboxTransport()
        resolve_session_transport(aligned)

    mismatch_logs = [r for r in caplog.records if "ctx.transport" in r.getMessage()]
    assert len(mismatch_logs) == 2  # one per mismatched transport, not per resolve
    assert "durable" in mismatch_logs[0].getMessage()
    assert "ephemeral" in mismatch_logs[1].getMessage()
