"""Mixin providing shared checkpoint key/load/save for holders of session state."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from .checkpoints import AgentCheckpoint, CheckpointKind, ProcessorCheckpoint
from .store_keys import make_store_key

if TYPE_CHECKING:
    from grasp_agents.session_context import SessionContext

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

    def _checkpoint_store_key(self, ctx: SessionContext[Any]) -> str | None:
        if ctx.checkpoint_store is None or self._checkpoint_kind is None:
            return None
        return make_store_key(ctx.session_key, self._checkpoint_kind, self._path)

    async def has_checkpoint(self, ctx: SessionContext[Any]) -> bool:
        """
        True if a checkpoint head exists in the store for this object's key.

        A cheap existence probe (no JSON parse): distinguishes a session that can
        be resumed from one that was interrupted before it ever checkpointed.
        """
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return False
        return await store.load(key) is not None

    async def _deserialize_checkpoint[CpT: ProcessorCheckpoint](
        self,
        ctx: SessionContext[Any],
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
        ctx: SessionContext[Any],
        checkpoint: ProcessorCheckpoint,
    ) -> None:
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return

        t0 = time.monotonic()
        checkpoint.checkpoint_number = self._checkpoint_number + 1
        await store.save(key, checkpoint.model_dump_json().encode("utf-8"))
        self._checkpoint_number += 1
        logger.debug(
            "checkpoint saved (#%d) in %.0fms",
            self._checkpoint_number,
            (time.monotonic() - t0) * 1000,
        )


class AgentCheckpointPersistMixin(CheckpointPersistMixin):
    """
    Extends checkpoint persistence with the agent's message-log plane.

    An :class:`AgentCheckpoint` is stored as a small head blob plus an
    append-only message log; this mixin owns the log bookkeeping and the
    three write/read paths over it (append/rewrite on save, truncate on
    rollback, trim-and-load on resume).
    """

    # ``_persisted_messages`` holds the exact message objects already on the
    # log (strong refs, so identity comparison stays sound): a save appends the
    # delta while they remain a leading prefix of the transcript, else rewrites
    # the whole log. ``_log_version`` is the log file the current head points
    # at — rewrites bump it (see :meth:`_serialize_agent_checkpoint`).
    _persisted_messages: tuple[Any, ...] = ()
    _log_version: int = 0

    async def _serialize_agent_checkpoint(
        self,
        ctx: SessionContext[Any],
        checkpoint: AgentCheckpoint,
    ) -> None:
        """
        Persist an :class:`AgentCheckpoint` as message-log + head.

        Appends only the messages added since the last checkpoint, then
        overwrites the small head blob with ``messages`` excluded. Log is
        written before the head so the head's ``message_count`` only ever
        undercounts a crash — never points past what is on the log.

        The whole log is rewritten instead when the already-persisted messages
        are no longer a leading prefix of the transcript — a resume tail-strip,
        a failed-run revert, or a builder that reseeds / prunes history
        (divergence detected by object identity + length). Per-turn context
        management does NOT take this path: it shapes the transient model-facing
        view (see ``ViewProjector``) and leaves this append-only log — and the
        rollback boundaries indexing it — intact. A hook that mutates a
        persisted message's fields in place is NOT detected (and so not
        re-persisted); replace the object instead.

        A rewrite goes to a **new log version**, with the head re-pointed
        only after the rewrite lands: a crash in between leaves the old
        head + old log pair intact, instead of pairing the old head's
        watermark with a log it no longer describes. The superseded
        version is removed best-effort once the new head is durable (a
        crash can orphan one file — harmless, cleaned by ``delete``).
        """
        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            return

        t0 = time.monotonic()
        messages = checkpoint.messages
        persisted = self._persisted_messages
        prefix_intact = len(messages) >= len(persisted) and all(
            m is p for m, p in zip(messages, persisted, strict=False)
        )
        rewrite = not prefix_intact
        # A new version is only needed when an existing head + log pair
        # would be superseded; the first save of a fresh session rewrites
        # its own (empty) version in place.
        supersedes = bool(persisted) or self._checkpoint_number > 0
        log_version = self._log_version + (1 if rewrite and supersedes else 0)
        if rewrite:
            await store.rewrite_messages(key, messages, version=log_version)
        else:
            new_messages = messages[len(persisted) :]
            if new_messages:
                await store.append_messages(key, new_messages, version=log_version)

        checkpoint.checkpoint_number = self._checkpoint_number + 1
        checkpoint.current.message_count = len(messages)
        checkpoint.current.log_version = log_version
        head_blob = checkpoint.model_dump_json(exclude={"messages"}).encode("utf-8")
        await store.save(key, head_blob)

        if rewrite and self._log_version != log_version:
            try:
                await store.rewrite_messages(key, [], version=self._log_version)
            except Exception:
                logger.warning(
                    "Failed to remove superseded message-log version %d at %s",
                    self._log_version,
                    key,
                    exc_info=True,
                )

        self._log_version = log_version
        self._persisted_messages = tuple(messages)
        self._checkpoint_number += 1
        logger.debug(
            "checkpoint saved (#%d, %d msgs%s) in %.0fms",
            self._checkpoint_number,
            len(messages),
            ", full rewrite" if rewrite else "",
            (time.monotonic() - t0) * 1000,
        )

    async def _serialize_rollback_checkpoint(
        self,
        ctx: SessionContext[Any],
        checkpoint: AgentCheckpoint,
    ) -> None:
        """
        Persist a rewound :class:`AgentCheckpoint`: head-first, then truncate
        the durable message-log to match.

        Head-first — lower the head's watermark before shrinking the log — so
        a crash in between leaves the head pointing *within* the log (resume
        re-trims the leftover tail), never past it. The log is truncated in
        place, so the head keeps the live log version (the checkpoint's
        ``current`` may be cut from an older boundary). Without a checkpoint
        store (live-only rollback) only the append bookkeeping is re-anchored
        to the truncated transcript.
        """
        messages = checkpoint.messages
        checkpoint.current.message_count = len(messages)
        checkpoint.current.log_version = self._log_version

        store = ctx.checkpoint_store
        key = self._checkpoint_store_key(ctx)
        if store is None or key is None:
            self._persisted_messages = tuple(messages)
            return

        checkpoint.checkpoint_number = self._checkpoint_number + 1
        head_blob = checkpoint.model_dump_json(exclude={"messages"}).encode("utf-8")
        await store.save(key, head_blob)
        await store.truncate_messages(
            key, message_count=len(messages), version=self._log_version
        )
        self._persisted_messages = tuple(messages)
        self._checkpoint_number += 1

    async def _deserialize_agent_checkpoint(
        self,
        ctx: SessionContext[Any],
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

        t0 = time.monotonic()
        head = await store.load_json(
            key, AgentCheckpoint, subject=f"checkpoint at {key}"
        )
        if head is None:
            return None

        raw = await store.read_messages(key, version=head.current.log_version)
        committed = raw[: head.current.message_count]
        head.messages = committed
        # Drop an uncommitted / torn tail (records past the head's watermark)
        # so later appends extend a clean file. Skip when the log is already
        # exactly the committed prefix — the common, clean resume. (Same
        # version: the content is a prefix of what is there, so a crash
        # mid-rewrite cannot invalidate the head's watermark.)
        if len(raw) != len(committed):
            await store.rewrite_messages(
                key, committed, version=head.current.log_version
            )

        self._checkpoint_number = head.checkpoint_number
        self._log_version = head.current.log_version
        self._persisted_messages = tuple(committed)
        logger.info(
            "checkpoint resumed (#%d, %d msgs) in %.0fms",
            head.checkpoint_number,
            len(committed),
            (time.monotonic() - t0) * 1000,
        )
        return head
