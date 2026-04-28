import ast
import json
import re
from logging import getLogger
from typing import Annotated, Any, TypeVar, get_args, get_origin

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from ..types.errors import JSONSchemaValidationError, PyJSONStringParsingError

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


T = TypeVar("T")


def validate_obj_from_json_or_py_string(
    s: str,
    schema: type[T],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> T:
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
