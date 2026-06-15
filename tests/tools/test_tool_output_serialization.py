"""
Tool-output serialization: a string result passes through verbatim (no
spurious quotes the model would then read), while non-strings serialize to JSON
data rather than a Python ``repr``.
"""

from __future__ import annotations

from pydantic import BaseModel

from grasp_agents.types.items import FunctionToolOutputItem


class _Result(BaseModel):
    name: str = "x"
    count: int = 3


def test_string_output_not_quoted() -> None:
    item = FunctionToolOutputItem.from_tool_result(call_id="c1", output="plain text")
    assert item.output == "plain text"


def test_already_json_string_not_double_escaped() -> None:
    item = FunctionToolOutputItem.from_tool_result(call_id="c1", output='{"a": 1}')
    assert item.output == '{"a": 1}'


def test_model_output_is_json_not_repr() -> None:
    item = FunctionToolOutputItem.from_tool_result(call_id="c1", output=_Result())
    assert '"name": "x"' in item.output
    assert '"count": 3' in item.output
    assert "_Result(" not in item.output
