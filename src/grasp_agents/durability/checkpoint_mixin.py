"""Mixin providing shared checkpoint key/load/save for holders of session state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from .checkpoints import AgentCheckpoint, CheckpointKind, ProcessorCheckpoint
from .store_keys import make_store_key

if TYPE_CHECKING:
    from ..run_context import RunContext

logger = logging.getLogger(__name__)


class CheckpointPersistMixin:
    """
    Adds checkpoint key composition + load/save to a session-scoped object.

    Subclasses set ``_checkpoint_kind`` (``None`` disables persistence)
    and maintain ``_path`` / ``_checkpoint_number``.
    """

    _checkpoint_kind: ClassVar[CheckpointKind | None] = None

    _path: list[str]
    _checkpoint_number: int

    # Append-only message-log bookkeeping. Only meaningful for the agent
    # checkpoint (the only message-bearing one); harmless defaults elsewhere.
    # ``_persisted_message_count`` is the offset already on the log;
    # ``_log_dirty`` is set when the transcript is rebuilt from scratch, so the
    # next checkpoint rewrites the log instead of appending a stale delta.
    _persisted_message_count: int = 0
    _log_dirty: bool = False

    def _checkpoint_store_key(self, ctx: RunContext[Any]) -> str | None:
        if ctx.checkpoint_store is None or self._checkpoint_kind is None:
            return None
        return make_store_key(ctx.session_key, self._checkpoint_kind, self._path)

    async def _deserialize_checkpoint[CpT: ProcessorCheckpoint](
        self,
        ctx: RunContext[Any],
        checkpoint_type: type[CpT],
    ) -> CpT | None:
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return None

        checkpoint = await store.load_json(
            key, checkpoint_type, subject=f"checkpoint at {key}"
        )
        if checkpoint is None:
            return None

        self._checkpoint_number = checkpoint.checkpoint_number
        return checkpoint

    async def _serialize_checkpoint(
        self,
        ctx: RunContext[Any],
        checkpoint: ProcessorCheckpoint,
    ) -> None:
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return

        checkpoint.checkpoint_number = self._checkpoint_number + 1
        await store.save(key, checkpoint.model_dump_json().encode("utf-8"))
        self._checkpoint_number += 1

    async def _serialize_agent_checkpoint(
        self,
        ctx: RunContext[Any],
        checkpoint: AgentCheckpoint,
    ) -> None:
        """
        Persist an :class:`AgentCheckpoint` as message-log + head.

        Appends only the messages added since the last checkpoint (or rewrites
        the whole log after a from-scratch transcript), then overwrites the
        small head blob with ``messages`` excluded. Log is written before the
        head so the head's ``message_count`` only ever undercounts a crash —
        never points past what is on the log.
        """
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return

        messages = checkpoint.messages
        if self._log_dirty:
            await store.rewrite_messages(key, messages)
            self._log_dirty = False
        else:
            new_messages = messages[self._persisted_message_count :]
            if new_messages:
                await store.append_messages(key, new_messages)

        checkpoint.checkpoint_number = self._checkpoint_number + 1
        checkpoint.message_count = len(messages)
        head_blob = checkpoint.model_dump_json(exclude={"messages"}).encode("utf-8")
        await store.save(key, head_blob)

        self._persisted_message_count = len(messages)
        self._checkpoint_number += 1

    async def _deserialize_agent_checkpoint(
        self,
        ctx: RunContext[Any],
    ) -> AgentCheckpoint | None:
        """
        Reconstruct an :class:`AgentCheckpoint` from head + message-log.

        Trusts only the head's watermark: any records beyond ``message_count``
        (an interrupted save) or a torn final line are dropped. When such a tail
        exists the log is rewritten to the committed prefix so later appends
        extend a clean file — a one-off write at resume, off the per-turn hot
        path; a clean log is left untouched.
        """
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return None

        head = await store.load_json(
            key, AgentCheckpoint, subject=f"checkpoint at {key}"
        )
        if head is None:
            return None

        raw = await store.read_messages(key)
        committed = raw[: head.message_count]
        head.messages = committed
        # Drop an uncommitted / torn tail (records past the head's watermark)
        # so later appends extend a clean file. Skip when the log is already
        # exactly the committed prefix — the common, clean resume.
        if len(raw) != len(committed):
            await store.rewrite_messages(key, committed)

        self._checkpoint_number = head.checkpoint_number
        self._persisted_message_count = len(committed)
        return head
