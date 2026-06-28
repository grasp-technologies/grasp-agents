from datetime import UTC, datetime

from pydantic import Field

from .checkpoints import PersistedRecord
from .task_record import TaskStatus


class MessageRecord(PersistedRecord):
    """
    A persisted async message addressed to an agent — the messaging sibling of
    :class:`TaskRecord`.

    Shares the same checkpoint-store substrate: the store-key convention
    (``"<session_key>/mailbox/<recipient>/inbox|processed/<message_id>"``), the
    :class:`TaskStatus` lifecycle (``PENDING`` undelivered → ``DELIVERED``), and
    schema versioning. ``body`` is the serialized message, kept opaque so the
    durability layer stays agnostic of the messaging payload type.
    """

    message_id: str
    sender: str
    recipient: str
    body: str  # the serialized message (e.g. a TeamMessage JSON)

    status: TaskStatus = TaskStatus.PENDING

    created_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
