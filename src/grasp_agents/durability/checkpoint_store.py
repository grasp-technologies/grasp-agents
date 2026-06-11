import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, TypeAdapter

from ..types.items import InputItem
from .checkpoints import CheckpointSchemaError

logger = logging.getLogger(__name__)

_INPUT_ITEM_ADAPTER: TypeAdapter[InputItem] = TypeAdapter(InputItem)


def encode_messages(messages: Sequence[InputItem]) -> bytes:
    """Frame messages as newline-terminated JSONL (one ``InputItem`` per line)."""
    return b"".join(m.model_dump_json().encode("utf-8") + b"\n" for m in messages)


def decode_message_log(blob: bytes) -> list[InputItem]:
    """
    Parse a JSONL message log, tolerating a torn tail.

    Stops at the first line that fails to parse (a partial trailing record
    from an interrupted append, or any corruption beyond it) and returns the
    valid prefix — so an interrupted write costs at most the last record.
    """
    messages: list[InputItem] = []
    for line in blob.split(b"\n"):
        if not line:
            continue
        try:
            messages.append(_INPUT_ITEM_ADAPTER.validate_json(line))
        except ValueError:
            break
    return messages


class CheckpointStore(ABC):
    """
    Key-value persistence backend for session checkpoints.

    Subclasses implement the raw byte-level :meth:`save` / :meth:`load` /
    :meth:`delete` / :meth:`list_keys` and the message-log
    :meth:`append_messages` / :meth:`read_messages` / :meth:`rewrite_messages`;
    :meth:`load_json` is provided.
    """

    @abstractmethod
    async def save(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    async def load(self, key: str) -> bytes | None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, prefix: str) -> list[str]: ...

    async def load_json[M: BaseModel](
        self,
        key: str,
        model_type: type[M],
        *,
        subject: str | None = None,
    ) -> M | None:
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

    # --- Append-only message log (the agent transcript) ---
    #
    # Keyed to the checkpoint head: ``append_messages`` extends it incrementally
    # (no full-transcript rewrite per turn), ``read_messages`` returns the whole
    # log, ``rewrite_messages`` replaces it (used on resume to drop an
    # uncommitted / torn tail). ``read_messages`` must tolerate a torn final
    # record (see :func:`decode_message_log`); ``encode_messages`` /
    # ``decode_message_log`` give backends a ready JSONL framing.

    @abstractmethod
    async def append_messages(
        self, key: str, messages: Sequence[InputItem]
    ) -> None: ...

    @abstractmethod
    async def read_messages(self, key: str) -> list[InputItem]: ...

    @abstractmethod
    async def rewrite_messages(
        self, key: str, messages: Sequence[InputItem]
    ) -> None: ...


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint store for testing and short-lived sessions."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._logs: dict[str, list[InputItem]] = {}

    async def save(self, key: str, data: bytes) -> None:
        self._data[key] = data

    async def load(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._logs.pop(key, None)

    async def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self._data if k.startswith(prefix)]

    async def append_messages(self, key: str, messages: Sequence[InputItem]) -> None:
        if messages:
            self._logs.setdefault(key, []).extend(messages)

    async def read_messages(self, key: str) -> list[InputItem]:
        return list(self._logs.get(key, []))

    async def rewrite_messages(self, key: str, messages: Sequence[InputItem]) -> None:
        if messages:
            self._logs[key] = list(messages)
        else:
            self._logs.pop(key, None)
