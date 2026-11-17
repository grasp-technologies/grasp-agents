"""Side-effect-free readers over a persisted session (``session_history``)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from grasp_agents.durability import (
    AgentCheckpoint,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    MessageRecord,
    MessageStatus,
    SessionCheckpoint,
    StepWatermark,
    TaskRecord,
    read_agent_histories,
    read_fs_snapshot_refs,
    read_pending_messages,
    read_task_records,
)
from grasp_agents.types.items import InputItem, InputMessageItem
from grasp_agents.types.message import CONTROL_PRIORITY, TeamMessage

_SK = "s1"


def _head(name: str, count: int) -> bytes:
    checkpoint = AgentCheckpoint(
        session_key=_SK,
        processor_name=name,
        current=StepWatermark(message_count=count),
    )
    return checkpoint.model_dump_json(exclude={"messages"}).encode()


def _msgs(*texts: str) -> list[InputItem]:
    return [InputMessageItem.from_text(t) for t in texts]


def _mail_record(message: TeamMessage, **kwargs: object) -> bytes:
    return (
        MessageRecord(session_key=_SK, message=message, **kwargs)  # type: ignore[arg-type]
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
async def test_reads_committed_prefix_without_trimming() -> None:
    store = InMemoryCheckpointStore()
    key = f"{_SK}/agent/lead"
    await store.append_messages(key, _msgs("a", "b", "c"))
    await store.save(key, _head("lead", 2))

    histories = await read_agent_histories(store, _SK)

    assert [h.name for h in histories] == ["lead"]
    texts = [m.text for m in histories[0].messages if isinstance(m, InputMessageItem)]
    assert texts == ["a", "b"]  # the uncommitted third record is excluded…
    # …but, unlike resume, never trimmed off the stored log.
    assert len(await store.read_messages(key)) == 3


@pytest.mark.asyncio
async def test_nested_subagents_carry_their_root_and_sort_shallow_first() -> None:
    store = InMemoryCheckpointStore()
    nested = f"{_SK}/agent/lead/tc_1/scout"
    await store.append_messages(nested, _msgs("x"))
    await store.save(nested, _head("scout", 1))
    top = f"{_SK}/agent/lead"
    await store.append_messages(top, _msgs("a"))
    await store.save(top, _head("lead", 1))

    histories = await read_agent_histories(store, _SK)

    assert [(h.name, h.root) for h in histories] == [
        ("lead", "lead"),
        ("scout", "lead"),
    ]


@pytest.mark.asyncio
async def test_task_records_attribute_the_launching_agent() -> None:
    store = InMemoryCheckpointStore()
    record = TaskRecord(
        session_key=_SK,
        task_id="bg1",
        tool_call_id="call_1",
        tool_name="index_sources",
    )
    await store.save(f"{_SK}/task/lead/tc_call_1", record.model_dump_json().encode())

    tasks = await read_task_records(store, _SK)

    assert [(agent, r.task_id) for agent, r in tasks] == [("lead", "bg1")]


@pytest.mark.asyncio
async def test_pending_messages_in_drain_order_scoped_to_a_recipient() -> None:
    store = InMemoryCheckpointStore()
    # Human mail sits in a smaller (higher-priority) lane than peer mail, so it
    # drains first even with a later id.
    peer = TeamMessage.from_text(sender="writer", to="lead", text="draft")
    human = TeamMessage.from_text(
        sender="user", to="lead", text="hello", priority=CONTROL_PRIORITY
    )
    for lane, message in (("99", peer), ("97", human)):
        await store.save(
            f"{_SK}/mailbox/lead/inbox/{lane}/{message.message_id}",
            _mail_record(message),
        )
    # Consumed mail (a processed/ record) is not pending.
    done = TeamMessage.from_text(sender="user", to="lead", text="old")
    await store.save(
        f"{_SK}/mailbox/lead/processed/{done.message_id}",
        _mail_record(done, status=MessageStatus.DELIVERED),
    )
    other = TeamMessage.from_text(sender="user", to="writer", text="ping")
    await store.save(
        f"{_SK}/mailbox/writer/inbox/97/{other.message_id}", _mail_record(other)
    )

    everyone = await read_pending_messages(store, _SK)
    assert [m.text for m in everyone] == ["hello", "draft", "ping"]

    lead_only = await read_pending_messages(store, _SK, recipient="lead")
    assert [m.text for m in lead_only] == ["hello", "draft"]


def _refs_head(
    session_key: str, name: str, current: str | None, *bounds: str | None
) -> bytes:
    checkpoint = AgentCheckpoint(
        session_key=session_key,
        processor_name=name,
        current=StepWatermark(fs_snapshot_ref=current),
        step_watermarks=[
            StepWatermark(step=step, fs_snapshot_ref=ref)
            for step, ref in enumerate(bounds)
        ],
    )
    return checkpoint.model_dump_json(exclude={"messages"}).encode()


def _session(session_key: str, ref: str | None) -> bytes:
    return (
        SessionCheckpoint(session_key=session_key, fs_snapshot_ref=ref)
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_fs_snapshot_refs_of_one_session_or_every_session(
    kind: str, tmp_path: Path
) -> None:
    store: CheckpointStore = (
        InMemoryCheckpointStore() if kind == "memory" else FileCheckpointStore(tmp_path)
    )
    await store.save(f"{_SK}/session", _session(_SK, "snap-3"))
    await store.save(
        f"{_SK}/agent/lead",
        _refs_head(_SK, "lead", "snap-3", None, "snap-1", "snap-2"),
    )
    await store.save(f"{_SK}/agent/lead/tc_c1", _refs_head(_SK, "tc_c1", None))
    # A session key may hold slashes; its records are found all the same.
    await store.save("org/user-7/session", _session("org/user-7", "snap-9"))
    await store.save("org/user-7/agent/a", _refs_head("org/user-7", "a", "snap-8"))

    assert await read_fs_snapshot_refs(store, _SK) == {"snap-1", "snap-2", "snap-3"}
    assert await read_fs_snapshot_refs(store) == {
        "snap-1",
        "snap-2",
        "snap-3",
        "snap-8",
        "snap-9",
    }


@pytest.mark.asyncio
async def test_an_unreadable_record_fails_the_fs_snapshot_ref_read() -> None:
    # Its refs would be missing from a sweep's keep set.
    store = InMemoryCheckpointStore()
    await store.save(f"{_SK}/session", _session(_SK, "snap-1"))
    await store.save(f"{_SK}/agent/broken", b"{not json")

    with pytest.raises(ValidationError):
        await read_fs_snapshot_refs(store, _SK)
    with pytest.raises(ValidationError):
        await read_fs_snapshot_refs(store)
