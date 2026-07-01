from grasp_agents.types.message import TeamMessage

from .checkpoints import PersistedRecord
from .task_record import TaskStatus


class MessageRecord(PersistedRecord):
    """
    A persisted async message addressed to an agent — the messaging sibling of
    :class:`TaskRecord`.

    Shares the same checkpoint-store substrate: the store-key convention
    (``"<session_key>/mailbox/<recipient>/inbox|processed/<message_id>"``), the
    :class:`TaskStatus` lifecycle (``PENDING`` undelivered → ``DELIVERED``), and
    schema versioning. The :class:`TeamMessage` is nested directly (it is fully
    serializable) — sender / recipient / id all live on it, so the record adds
    only the delivery status on top.
    """

    message: TeamMessage
    status: TaskStatus = TaskStatus.PENDING
