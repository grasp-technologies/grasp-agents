import ast
import json
import re
from collections.abc import Mapping
from logging import getLogger
from typing import Annotated, Any, NoReturn, TypeVar, get_args, get_origin

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from ..errors import JSONSchemaValidationError, PyJSONStringParsingError

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


def validate_tagged_objs_from_json_or_py_string(
    s: str,
    schema_by_xml_tag: Mapping[str, type[T]],
    from_substring: bool = False,
    strip_language_markdown: bool = True,
) -> Mapping[str, T]:
    validated_obj_per_tag: dict[str, T] = {}
    _schema: type[T] | None = None
    _tag: str | None = None

    def _raise(
        tag: str,
        schema: Any,
        *,
        not_found: bool = False,
        base_exc: Exception | None = None,
    ) -> NoReturn:
        if not_found:
            msg = (
                f"Failed to find valid tagged section <{tag}>...</{tag}> "
                f"in the string:\n{s}"
            )
        else:
            msg = (
                f"Failed to validate substring within tag <{tag}> against JSON schema:"
                f"\n{s}\nExpected type: {schema}"
            )
        raise JSONSchemaValidationError(s, schema, message=msg) from base_exc

    for _tag, _schema in schema_by_xml_tag.items():
        if f"<{_tag}>" in s:
            match = re.search(rf"<{_tag}>\s*(.*?)\s*</{_tag}>", s, re.DOTALL)
            if match is None:
                _raise(_tag, _schema, not_found=True, base_exc=None)
            try:
                tagged_substring = match.group(1).strip()
                validated_obj_per_tag[_tag] = validate_obj_from_json_or_py_string(
                    tagged_substring,  # type: ignore[assignment]
                    schema=_schema,
                    from_substring=from_substring,
                    strip_language_markdown=strip_language_markdown,
                )
            except JSONSchemaValidationError as exc:
                _raise(_tag, _schema, base_exc=exc)

    return validated_obj_per_tag
