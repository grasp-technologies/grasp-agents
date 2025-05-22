import ast
import asyncio
import json
import re
from collections.abc import Coroutine, Mapping
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import TypeAdapter, ValidationError
from tqdm.autonotebook import tqdm

logger = getLogger(__name__)

_JSON_START_RE = re.compile(r"[{\[]")

T = TypeVar("T")


def extract_json_substring(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in _JSON_START_RE.finditer(text):
        start = match.start()
        try:
            _, end = decoder.raw_decode(text, idx=start)
            return text[start:end]
        except ValueError:
            continue

    return None


def parse_json_or_py_string(
    s: str, return_none_on_failure: bool = False
) -> dict[str, Any] | list[Any] | None:
    s_fmt = re.sub(r"```[a-zA-Z0-9]*\n|```", "", s).strip()
    try:
        return ast.literal_eval(s_fmt)
    except (ValueError, SyntaxError):
        try:
            return json.loads(s_fmt)
        except json.JSONDecodeError as exc:
            if return_none_on_failure:
                return None
            raise ValueError(
                "Invalid JSON/Python string - Both ast.literal_eval and json.loads "
                f"failed to parse the following response:\n{s}"
            ) from exc


def parse_json_or_py_substring(
    json_str: str, return_none_on_failure: bool = False
) -> dict[str, Any] | list[Any] | None:
    return parse_json_or_py_string(
        extract_json_substring(json_str) or "", return_none_on_failure
    )


@overload
def validate_obj_from_json_or_py_string(
    s: str,
    adapter: TypeAdapter[T],
    from_substring: bool = False,
) -> T: ...


@overload
def validate_obj_from_json_or_py_string(
    s: str,
    adapter: Mapping[str, TypeAdapter[T]],
    from_substring: bool = False,
) -> T | str: ...


def validate_obj_from_json_or_py_string(
    s: str,
    adapter: TypeAdapter[T] | Mapping[str, TypeAdapter[T]],
    from_substring: bool = False,
) -> T | str:
    _selected_adapter: TypeAdapter[T] | None = None
    if isinstance(adapter, Mapping):
        for _marker, _adapter in adapter.items():
            if _marker in s:
                _selected_adapter = _adapter
        if _selected_adapter is None:
            return s
    else:
        _selected_adapter = adapter

    try:
        if from_substring:
            parsed = parse_json_or_py_substring(s, return_none_on_failure=True)
        else:
            parsed = parse_json_or_py_string(s, return_none_on_failure=True)
        if parsed is None:
            parsed = s
        return _selected_adapter.validate_python(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Invalid JSON or Python string:\n{s}\n"
            f"Expected type: {_selected_adapter._type}",  # type: ignore[arg-type]
        ) from exc


def extract_xml_list(text: str) -> list[str]:
    pattern = re.compile(r"<(chunk_\d+)>(.*?)</\1>", re.DOTALL)

    chunks: list[str] = []
    for match in pattern.finditer(text):
        content = match.group(2).strip()
        chunks.append(content)
    return chunks


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
        logger.error(f"File {file_path} not found.")
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
    return datetime.now().strftime("%Y%m%d_%H%M%S")
