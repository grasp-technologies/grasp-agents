import logging
from typing import TypeVar

from pydantic import BaseModel

from .checkpoints import CheckpointSchemaError

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=BaseModel)


class CheckpointStore:
    """
    Key-value persistence backend for session checkpoints.

    Subclasses implement raw byte-level :meth:`save` / :meth:`load` /
    :meth:`delete` / :meth:`list_keys`; :meth:`load_json` is provided.
    """

    async def save(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    async def load(self, key: str) -> bytes | None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    async def load_json(
        self,
        key: str,
        model_type: type[_M],
        *,
        subject: str | None = None,
    ) -> _M | None:
        """
        Load ``key`` and validate as ``model_type``.

        Missing key → ``None``. Schema-version mismatch propagates as
        :class:`CheckpointSchemaError`. Any other deserialization error
        is logged at WARN and returns ``None``.
        """
        data = await self.load(key)
        if data is None:
            return None

        try:
            return model_type.model_validate_json(data)
        except CheckpointSchemaError:
            raise
        except Exception:
            logger.warning(
                "Corrupt %s at %s, treating as missing",
                subject or model_type.__name__,
                key,
                exc_info=True,
            )
            return None


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint store for testing and short-lived sessions."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def save(self, key: str, data: bytes) -> None:
        self._data[key] = data

    async def load(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self._data if k.startswith(prefix)]
