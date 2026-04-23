"""
Shared ``load + validate`` helper used by every persisted-data reader.

The guarantees are:

- Schema-version mismatches (``CheckpointSchemaError``) are re-raised so
  fail-fast still happens.
- Any other deserialization error is logged at WARN and collapses to
  ``None`` so the caller falls back to fresh state instead of crashing.
- Missing keys return ``None`` (protocol contract).

Used by both :class:`~..processors.processor.Processor` checkpoint
loads and :class:`~..agent.background_tasks.BackgroundTaskManager` task-
record scans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from .checkpoints import CheckpointSchemaError

if TYPE_CHECKING:
    import logging

    from .checkpoint_store import CheckpointStore

_M = TypeVar("_M", bound=BaseModel)


async def load_json(
    store: CheckpointStore,
    key: str,
    model_type: type[_M],
    *,
    subject: str,
    logger: logging.Logger,
) -> _M | None:
    """
    Load ``key`` from ``store`` and validate it as ``model_type``.

    Returns ``None`` on missing key or corrupt payload (logged at WARN).
    Re-raises :class:`CheckpointSchemaError` so schema-version drift
    does not silently degrade to fresh-state.
    """
    data = await store.load(key)
    if data is None:
        return None
    try:
        return model_type.model_validate_json(data)
    except CheckpointSchemaError:
        raise
    except Exception:
        logger.warning(
            "Corrupt %s at %s, treating as missing", subject, key, exc_info=True
        )
        return None
