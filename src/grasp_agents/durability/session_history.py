"""
Side-effect-free readers over a persisted session.

Unlike the resume path (which trims torn log tails, re-arms background tasks,
and re-injects notices), these only *look*: a UI relaunching over an existing
session — or any inspection tool — can rebuild what happened without mutating
the store or disturbing a live session sharing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .checkpoints import AgentCheckpoint, CheckpointKind
from .message_record import MessageRecord, MessageStatus
from .store_keys import make_store_key, task_prefix
from .task_record import TaskRecord

if TYPE_CHECKING:
    from grasp_agents.types.items import InputItem
    from grasp_agents.types.message import TeamMessage

    from .checkpoint_store import CheckpointStore


@dataclass(frozen=True)
class AgentHistory:
    """One agent's committed transcript, as persisted."""

    # The agent's display name (the last store-key segment) — for a nested
    # sub-agent this matches the ``event.source`` its live events carry.
    name: str
    # The top-level agent this transcript sits under (equal to ``name`` unless
    # nested) — lets a reader scope a shared store to one member's tree.
    root: str
    # The full store key of the agent's checkpoint head.
    key: str
    messages: list[InputItem]


async def read_agent_histories(
    store: CheckpointStore, session_key: str
) -> list[AgentHistory]:
    """
    Every agent's committed transcript under ``session_key``, shallow-first.

    Trusts only each head's commit watermark — uncommitted / torn trailing log
    records are excluded (but, unlike resume, never trimmed on disk). Agents
    whose head is missing or unreadable are skipped.
    """
    prefix = make_store_key(session_key, CheckpointKind.AGENT) + "/"
    keys = sorted(await store.list_keys(prefix), key=lambda k: (k.count("/"), k))
    histories: list[AgentHistory] = []
    for key in keys:
        head = await store.load_json(key, AgentCheckpoint, subject=f"head at {key}")
        if head is None:
            continue
        raw = await store.read_messages(key, version=head.current.log_version)
        segments = key[len(prefix) :].split("/")
        histories.append(
            AgentHistory(
                name=segments[-1],
                root=segments[0],
                key=key,
                messages=raw[: head.current.message_count],
            )
        )
    return histories


async def read_task_records(
    store: CheckpointStore, session_key: str
) -> list[tuple[str, TaskRecord]]:
    """
    All background-task records under ``session_key`` as
    ``(launching agent, record)``, oldest first.

    The launching agent is the record key's segment just before the
    ``tc_<call_id>`` leaf — the name its live events carry as ``source``.
    """
    prefix = task_prefix(session_key)
    tasks: list[tuple[str, TaskRecord]] = []
    for key in await store.list_keys(prefix):
        segments = key[len(prefix) :].split("/")
        if len(segments) < 2:
            continue
        record = await store.load_json(key, TaskRecord, subject=f"task record at {key}")
        if record is not None:
            tasks.append((segments[-2], record))
    tasks.sort(key=lambda t: t[1].created_at)
    return tasks


async def read_pending_messages(
    store: CheckpointStore, session_key: str, *, recipient: str | None = None
) -> list[TeamMessage]:
    """
    Not-yet-consumed mailbox messages, in each recipient's drain order
    (priority lanes, then time-ordered id). ``recipient`` narrows to one
    mailbox; the default scans them all.
    """
    base = make_store_key(session_key, CheckpointKind.MAILBOX)
    prefix = f"{base}/{recipient}/inbox/" if recipient else f"{base}/"
    pending: list[TeamMessage] = []
    for key in sorted(await store.list_keys(prefix)):
        if "/inbox/" not in key:
            continue
        record = await store.load_json(
            key, MessageRecord, subject=f"mailbox record at {key}"
        )
        if record is not None and record.status is MessageStatus.PENDING:
            pending.append(record.message)
    return pending
