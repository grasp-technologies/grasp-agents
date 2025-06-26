import ast
import asyncio
import json
import re
from collections.abc import Coroutine, Mapping
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, TypeVar, get_args, get_origin, overload

from pydantic import TypeAdapter, ValidationError
from tqdm.autonotebook import tqdm

from .errors import OutputValidationError, StringParsingError

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
        except json.JSONDecodeError:
            continue

    return None


def parse_json_or_py_string(
    s: str, return_none_on_failure: bool = False, strip_language_markdown: bool = True
) -> dict[str, Any] | list[Any] | None:
    s_orig = s
    if strip_language_markdown:
        s = re.sub(r"```[a-zA-Z0-9]*\n|```", "", s).strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            return json.loads(s)
        except json.JSONDecodeError as exc:
            if return_none_on_failure:
                return None
            raise StringParsingError(
                "Invalid JSON/Python string - Both ast.literal_eval and json.loads "
                f"failed to parse the following response:\n{s_orig}"
            ) from exc


def parse_json_or_py_substring(
    json_str: str,
    return_none_on_failure: bool = False,
    strip_language_markdown: bool = True,
) -> dict[str, Any] | list[Any] | None:
    return parse_json_or_py_string(
        extract_json_substring(json_str) or "",
        return_none_on_failure=return_none_on_failure,
        strip_language_markdown=strip_language_markdown,
    )


@overload
def validate_obj_from_json_or_py_string(
    s: str,
    adapter: TypeAdapter[T],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> T: ...


@overload
def validate_obj_from_json_or_py_string(
    s: str,
    adapter: Mapping[str, TypeAdapter[T]],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> T | str: ...


def validate_obj_from_json_or_py_string(
    s: str,
    adapter: TypeAdapter[T] | Mapping[str, TypeAdapter[T]],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> T | str:
    _selected_adapter: TypeAdapter[T] | None = None
    _selected_tag: str | None = None
    s_orig = s

    if isinstance(adapter, Mapping):
        for _tag, _adapter in adapter.items():
            match = re.search(rf"<{_tag}>\s*(.*?)\s*</{_tag}>", s, re.DOTALL)
            if not match:
                continue
            s = match.group(1).strip()
            _selected_adapter = _adapter
            _selected_tag = _tag
            break
        if _selected_adapter is None:
            return s
    else:
        _selected_adapter = adapter

    _type = _selected_adapter._type  # type: ignore[attr-defined]
    type_origin = get_origin(_type)
    type_args = get_args(_type)
    is_str_type = (_type is str) or (
        type_origin is Annotated and type_args and type_args[0] is str
    )

    try:
        if not is_str_type:
            if from_substring:
                parsed = parse_json_or_py_substring(
                    s,
                    return_none_on_failure=True,
                    strip_language_markdown=strip_language_markdown,
                )
            else:
                parsed = parse_json_or_py_string(
                    s,
                    return_none_on_failure=True,
                    strip_language_markdown=strip_language_markdown,
                )
            if parsed is None:
                parsed = s
        else:
            parsed = s
        return _selected_adapter.validate_python(parsed)
    except ValidationError as exc:
        err_message = f"Invalid JSON or Python string:\n{s_orig}"
        if _selected_tag:
            err_message += f"\nExpected type {_type} within tag <{_selected_tag}>"
        else:
            err_message += f"\nExpected type {_type}"
        raise OutputValidationError(err_message) from exc


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
