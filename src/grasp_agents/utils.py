import ast
import asyncio
from datetime import datetime
import functools
import json
import re
from collections.abc import Callable, Coroutine
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from tqdm.autonotebook import tqdm


def read_contents_from_file(
    file_path: str | Path,
    binary_mode: bool = False,
) -> str:
    """Reads and returns contents of file"""
    try:
        if binary_mode:
            with open(file_path, "rb") as file:
                return file.read()
        else:
            with open(file_path) as file:
                return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return ""


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

    return ""


def read_json_string(json_str: str) -> dict[str, Any] | list[Any]:
    try:
        json_response = ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        try:
            json_response = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON - Both ast.literal_eval and json.loads "
                f"failed to parse the following response:\n{json_str}"
            ) from exc

    return json_response  # type: ignore


def extract_json(json_str: str) -> dict[str, Any] | list[Any]:
    return read_json_string(format_json_string(json_str))


def extract_xml_list(text: str) -> list[str]:
    pattern = re.compile(r"<(chunk_\d+)>(.*?)</\1>", re.DOTALL)

    chunks: list[str] = []
    for match in pattern.finditer(text):
        content = match.group(2).strip()
        chunks.append(content)
    return chunks


def get_prompt(prompt_text: str | None, prompt_path: str | Path | None) -> str | None:
    if prompt_text is None:
        prompt = (
            read_contents_from_file(prompt_path) if prompt_path is not None else None
        )
    else:
        prompt = prompt_text

    return prompt


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
