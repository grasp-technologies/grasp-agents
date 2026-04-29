"""Mixin providing shared checkpoint key/load/save for holders of session state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from .checkpoints import CheckpointKind, ProcessorCheckpoint
from .store_keys import make_store_key

if TYPE_CHECKING:
    from ..run_context import RunContext

logger = logging.getLogger(__name__)

CpT = TypeVar("CpT", bound=ProcessorCheckpoint)


class CheckpointPersistMixin:
    """
    Adds checkpoint key composition + load/save to a session-scoped object.

    Subclasses set ``_checkpoint_kind`` (``None`` disables persistence)
    and maintain ``_path`` / ``_checkpoint_number``.
    """

    _checkpoint_kind: ClassVar[CheckpointKind | None] = None

    _path: list[str]
    _checkpoint_number: int

    def _checkpoint_store_key(self, ctx: RunContext[Any]) -> str | None:
        if ctx.checkpoint_store is None or self._checkpoint_kind is None:
            return None
        return make_store_key(
            ctx.session_key, self._checkpoint_kind, self._path
        )

    async def _deserialize_checkpoint(
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
