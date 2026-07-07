"""Side-effect-free readers over a persisted session (``session_history``)."""

import pytest

from grasp_agents.durability import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    MessageRecord,
    MessageStatus,
    StepWatermark,
    TaskRecord,
    read_agent_histories,
    read_pending_messages,
    read_task_records,
)
from grasp_agents.types.items import InputItem, InputMessageItem
from grasp_agents.types.message import CONTROL_PRIORITY, TeamMessage

_SK = "sess"


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
