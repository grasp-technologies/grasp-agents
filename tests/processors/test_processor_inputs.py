import pytest

from grasp_agents.processors.processor import Processor
from grasp_agents.types.errors import ProcInputValidationError


class IntProcessor(Processor[int, int, None]):
    pass


class StrProcessor(Processor[str, str, None]):
    pass


class ListProcessor(Processor[list[int], list[int], None]):
    pass


@pytest.mark.parametrize(
    ("proc", "in_args", "expected"),
    [
        (IntProcessor(name="p"), 0, [0]),
        (StrProcessor(name="p"), "", [""]),
        (ListProcessor(name="p"), list[int](), [list[int]()]),
    ],
)
def test_falsy_single_argument_is_one_argument(
    proc: Processor[object, object, None], in_args: object, expected: list[object]
) -> None:
    assert proc.validate_inputs(exec_id="e", in_args=in_args) == expected


def test_list_of_arguments_still_fans_out() -> None:
    proc = IntProcessor(name="p")
    assert proc.validate_inputs(exec_id="e", in_args=[0, 1, 2]) == [0, 1, 2]


def test_empty_argument_list_is_rejected() -> None:
    with pytest.raises(ProcInputValidationError, match="at least one argument"):
        IntProcessor(name="p").validate_inputs(exec_id="e", in_args=[])


@pytest.mark.asyncio
async def test_run_with_a_falsy_argument() -> None:
    out = await IntProcessor(name="p").run(in_args=0)
    assert list(out.payloads) == [0]
