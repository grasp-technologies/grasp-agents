from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointStore(Protocol):
    """Key-value persistence backend for session checkpoints."""

    async def save(self, key: str, data: bytes) -> None: ...
    async def load(self, key: str) -> bytes | None: ...
    async def delete(self, key: str) -> None: ...
    async def list_keys(self, prefix: str) -> list[str]: ...


class InMemoryCheckpointStore:
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
