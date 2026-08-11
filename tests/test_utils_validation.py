"""Parsing LLM output into JSON/Python values, and the fence handling."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from grasp_agents.types.errors import (
    JSONSchemaValidationError,
    PyJSONStringParsingError,
)
from grasp_agents.utils.validation import (
    parse_json_or_py_string,
    validate_obj_from_json_or_py_string,
)


class TestFenceStripping:
    """
    Only a fence wrapping the whole payload is markup; one anywhere else is
    content, and removing it rewrites the model's answer.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            '```json\n{"a": 1}\n```',
            "```\n{'a': 1}\n```",  # no language tag
            '  \n```json\n{"a": 1}\n```  \n',  # padded
            '```json\n{"a": 1}```',  # close on the content line
            '```json\r\n{"a": 1}\r\n```',  # CRLF
            '```json   \n{"a": 1}\n   ```',  # padding around the fences
        ],
    )
    def test_wrapping_fence_is_stripped(self, raw: str) -> None:
        assert parse_json_or_py_string(raw) == {"a": 1}

    def test_fence_inside_a_string_value_survives(self) -> None:
        """The regression: this used to parse, one code block lighter."""
        raw = '{"code": "```python\\nprint(1)\\n```"}'

        assert parse_json_or_py_string(raw) == {"code": "```python\nprint(1)\n```"}

    def test_wrapping_fence_stripped_without_touching_an_inner_one(self) -> None:
        raw = '```json\n{"code": "```py\\nx\\n```"}\n```'

        assert parse_json_or_py_string(raw) == {"code": "```py\nx\n```"}

    @pytest.mark.parametrize(
        "raw",
        [
            'Here you go:\n```json\n{"a": 1}\n```',
            '```json\n{"a": 1}\n```\nHope that helps!',
        ],
    )
    def test_fence_amid_prose_is_left_alone(self, raw: str) -> None:
        """Neither half is parseable on its own — fail rather than guess."""
        with pytest.raises(PyJSONStringParsingError):
            parse_json_or_py_string(raw)

    def test_prose_around_a_fence_still_parses_from_a_substring(self) -> None:
        raw = 'Here you go:\n```json\n{"a": 1}\n```'

        assert parse_json_or_py_string(raw, from_substring=True) == {"a": 1}

    @pytest.mark.parametrize(
        "raw",
        [
            "```python x=1``` ```java y=5```",  # no newline after the tag
            "```python\nx=1\n```\n```java\ny=5\n```",
            '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```',
        ],
    )
    def test_several_sibling_blocks_fail_loudly_quoting_the_original(
        self, raw: str
    ) -> None:
        """
        Sibling blocks are not one payload, so nothing here is parseable.
        What matters is that the caller is shown what the model actually
        said rather than a half-stripped rewrite of it.
        """
        with pytest.raises(PyJSONStringParsingError) as exc_info:
            parse_json_or_py_string(raw)

        assert raw in str(exc_info.value)

    def test_stripping_can_be_disabled(self) -> None:
        with pytest.raises(PyJSONStringParsingError):
            parse_json_or_py_string(
                '```json\n{"a": 1}\n```', strip_language_markdown=False
            )


class TestValidateFromString:
    def test_fenced_output_validates_against_a_schema(self) -> None:
        class M(BaseModel):
            a: int

        assert validate_obj_from_json_or_py_string(
            '```json\n{"a": 1}\n```', schema=M
        ) == M(a=1)

    def test_str_schema_keeps_the_payload_verbatim(self) -> None:
        """A str output is the answer, not a container — fences are content."""
        raw = '```json\n{"a": 1}\n```'

        assert validate_obj_from_json_or_py_string(raw, schema=str) == raw

    def test_unparseable_output_raises_schema_error(self) -> None:
        class M(BaseModel):
            a: int

        with pytest.raises(JSONSchemaValidationError):
            validate_obj_from_json_or_py_string("not json at all", schema=M)
