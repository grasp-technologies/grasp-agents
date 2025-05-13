import ast
import asyncio
import functools
import json
import re
from collections.abc import Callable, Coroutine
from copy import deepcopy
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, GetCoreSchemaHandler, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from pydantic_core import core_schema
from tqdm.autonotebook import tqdm

logger = getLogger(__name__)


def merge_pydantic_models(*models: type[BaseModel]) -> type[BaseModel]:
    fields_dict: dict[str, FieldInfo] = {}
    for model in models:
        for field_name, field_info in model.model_fields.items():
            if field_name in fields_dict:
                raise ValueError(
                    f"Field conflict detected: '{field_name}' exists in multiple models"
                )
            fields_dict[field_name] = field_info

    return create_model("MergedModel", __module__=__name__, **fields_dict)  # type: ignore


def filter_fields(data: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    return {key: data[key] for key in model.model_fields if key in data}


def read_txt(file_path: str) -> str:
    return Path(file_path).read_text()


def format_json_string(text: str) -> str:
    decoder = json.JSONDecoder()
    text = text.replace("\n", "")
    length = len(text)
    i = 0
    while i < length:
        ch = text[i]
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(text[i:])
                return text[i : i + end]
            except ValueError:
                pass
        i += 1

    return text


def read_json_string(
    json_str: str, return_none_on_failure: bool = False
) -> dict[str, Any] | list[Any] | None:
    try:
        json_response = ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        try:
            json_response = json.loads(json_str)
        except json.JSONDecodeError as exc:
            if return_none_on_failure:
                return None
            raise ValueError(
                "Invalid JSON - Both ast.literal_eval and json.loads "
                f"failed to parse the following response:\n{json_str}"
            ) from exc

    return json_response


def extract_json(
    json_str: str, return_none_on_failure: bool = False
) -> dict[str, Any] | list[Any] | None:
    return read_json_string(format_json_string(json_str), return_none_on_failure)


def extract_xml_list(text: str) -> list[str]:
    pattern = re.compile(r"<(chunk_\d+)>(.*?)</\1>", re.DOTALL)

    chunks: list[str] = []
    for match in pattern.finditer(text):
        content = match.group(2).strip()
        chunks.append(content)
    return chunks


def make_conditional_parsed_output_type(
    response_format: type, marker: str = "<DONE>"
) -> type:
    class ParsedOutput:
        """
        * Accepts any **str**.
        * If the string contains `marker`, it must contain a valid JSON for
        `response_format` → we return that a response_format instance.
        * Otherwise we leave the string untouched.
        """

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            def validator(v: Any) -> Any:
                if isinstance(v, str) and marker in v:
                    v_json_str = format_json_string(v)
                    response_format_adapter = TypeAdapter[Any](response_format)

                    return response_format_adapter.validate_json(v_json_str)

                return v

            return core_schema.no_info_after_validator_function(
                validator, core_schema.any_schema()
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetCoreSchemaHandler
        ):
            return handler(core_schema)

    return ParsedOutput


T = TypeVar("T", bound=Callable[..., Any])


def forbid_state_change(method: T) -> T:
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        before = deepcopy(self.__dict__)
        result = method(self, *args, **kwargs)
        after = self.__dict__
        if before != after:
            raise RuntimeError(
                f"Method '{method.__name__}' modified the instance state."
            )
        return result

    return cast("T", wrapper)


def read_contents_from_file(
    file_path: str | Path,
    binary_mode: bool = False,
) -> str | bytes:
    """Reads and returns contents of file"""
    try:
        if binary_mode:
            with open(file_path, "rb") as file:
                return file.read()
        else:
            with open(file_path) as file:
                return file.read()
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return ""


def get_prompt(prompt_text: str | None, prompt_path: str | Path | None) -> str | None:
    if prompt_text is None:
        prompt = (
            read_contents_from_file(prompt_path) if prompt_path is not None else None
        )
    else:
        prompt = prompt_text

    return prompt  # type: ignore[assignment]


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
