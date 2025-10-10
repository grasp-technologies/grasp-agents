import asyncio
from collections.abc import AsyncIterator, Coroutine
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

from tqdm.autonotebook import tqdm

logger = getLogger(__name__)


def read_txt(file_path: str | Path, encoding: str = "utf-8") -> str:
    return Path(file_path).read_text(encoding=encoding)


def read_contents_from_file(
    file_path: str | Path,
    binary_mode: bool = False,
) -> str | bytes:
    try:
        if binary_mode:
            return Path(file_path).read_bytes()
        return Path(file_path).read_text()
    except FileNotFoundError:
        logger.exception(f"File {file_path} not found.")
        return ""


def get_prompt(prompt_text: str | None, prompt_path: str | Path | None) -> str | None:
    if prompt_text is None:
        return read_contents_from_file(prompt_path) if prompt_path is not None else None  # type: ignore[arg-type]

    return prompt_text


async def asyncio_gather_with_pbar(
    *corouts: Coroutine[Any, Any, Any],
    no_tqdm: bool = False,
    desc: str | None = None,
) -> list[Any]:
    # TODO: optimize
    pbar = tqdm(total=len(corouts), desc=desc, disable=no_tqdm)

    async def run_and_update(coro: Coroutine[Any, Any, Any]) -> Any:
        result = await coro
        pbar.update(1)
        return result

    wrapped_tasks = [run_and_update(c) for c in corouts]
    results = await asyncio.gather(*wrapped_tasks)
    pbar.close()

    return results


def get_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


_T = TypeVar("_T")


async def stream_concurrent(
    generators: list[AsyncIterator[_T]],
) -> AsyncIterator[tuple[int, _T]]:
    queue: asyncio.Queue[tuple[int, _T] | None] = asyncio.Queue()
    pumps_left = len(generators)

    async def pump(gen: AsyncIterator[_T], idx: int) -> None:
        nonlocal pumps_left
        try:
            async for item in gen:
                await queue.put((idx, item))
        finally:
            pumps_left -= 1
            if pumps_left == 0:
                await queue.put(None)

    for idx, gen in enumerate(generators):
        asyncio.create_task(pump(gen, idx))

    while True:
        msg = await queue.get()
        if msg is None:
            break
        yield msg


def is_method_overridden(
    method_name: str, self: object, base_cls: type[Any] | None = None
) -> bool:
    """
    Check if a method is overridden in a subclass compared to a base class
    or if it is defined directly in the instance's __dict__ (e.g. via a decorator).
    """
    child_cls = type(self)
    if not hasattr(child_cls, method_name):
        raise AttributeError(f"{child_cls} has no method named {method_name}")

    set_via_decorator = method_name in self.__dict__

    overriden_in_child = False
    if base_cls is not None:
        overriden_in_child = hasattr(base_cls, method_name) and getattr(
            child_cls, method_name
        ) is not getattr(base_cls, method_name)

    return set_via_decorator or overriden_in_child
