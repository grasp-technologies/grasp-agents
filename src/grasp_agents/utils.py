import ast
import asyncio
import json
import re
from collections.abc import AsyncIterator, Coroutine, Mapping
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, TypeVar, get_args, get_origin

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from tqdm.autonotebook import tqdm

from .errors import JSONSchemaValidationError, PyJSONStringParsingError

logger = getLogger(__name__)

_JSON_START_RE = re.compile(r"[{\[]")


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
    s: str,
    from_substring: bool = False,
    return_none_on_failure: bool = False,
    strip_language_markdown: bool = True,
) -> dict[str, Any] | list[Any] | None:
    s_orig = s

    if strip_language_markdown:
        s = re.sub(r"```[a-zA-Z0-9]*\n|```", "", s).strip()

    if from_substring:
        s = extract_json_substring(s) or ""

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            return json.loads(s)
        except json.JSONDecodeError as exc:
            err_message = (
                "Both ast.literal_eval and json.loads "
                f"failed to parse the following JSON/Python string:\n{s_orig}"
            )
            if return_none_on_failure:
                logger.warning(err_message)
                return None
            raise PyJSONStringParsingError(s_orig, message=err_message) from exc


def is_str_type(t: Any) -> bool:
    type_origin = get_origin(t)
    type_args = get_args(t)

    return (t is str) or (
        (type_origin is Annotated) and len(type_args) > 0 and type_args[0] is str
    )


def validate_obj_from_json_or_py_string(
    s: str,
    schema: Any,
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> Any:
    try:
        if is_str_type(schema):
            parsed = s
        else:
            parsed = parse_json_or_py_string(
                s,
                return_none_on_failure=True,
                from_substring=from_substring,
                strip_language_markdown=strip_language_markdown,
            )
        return TypeAdapter(schema).validate_python(parsed)
    except PydanticValidationError as exc:
        raise JSONSchemaValidationError(s, schema) from exc


def validate_tagged_objs_from_json_or_py_string(
    s: str,
    schema_by_xml_tag: Mapping[str, Any],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> Mapping[str, Any]:
    validated_obj_per_tag: dict[str, Any] = {}
    _schema: Any = None
    _tag: str | None = None

    try:
        for _tag, _schema in schema_by_xml_tag.items():
            match = re.search(rf"<{_tag}>\s*(.*?)\s*</{_tag}>", s, re.DOTALL)
            if not match:
                continue
            tagged_substring = match.group(1).strip()
            validated_obj_per_tag[_tag] = validate_obj_from_json_or_py_string(
                tagged_substring,  # type: ignore[assignment]
                schema=_schema,
                from_substring=from_substring,
                strip_language_markdown=strip_language_markdown,
            )
    except JSONSchemaValidationError as exc:
        err_message = (
            f"Failed to validate substring within tag <{_tag}> against JSON schema:"
            f"\n{s}\nExpected type: {_schema}"
        )
        raise JSONSchemaValidationError(s, _schema, message=err_message) from exc

    return validated_obj_per_tag


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
