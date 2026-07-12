import pytest

from grasp_agents.utils.errors import format_error_chain


def _raise_root() -> None:
    msg = "bad request"
    raise ValueError(msg)


def _raise_wrapped() -> None:
    try:
        _raise_root()
    except ValueError as err:
        msg = "run failed"
        raise RuntimeError(msg) from err


def test_single_error() -> None:
    assert format_error_chain(ValueError("boom")) == "ValueError: boom"


def test_empty_message_shows_type_only() -> None:
    assert format_error_chain(ValueError()) == "ValueError"


def test_explicit_cause_chain() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        _raise_wrapped()
    assert format_error_chain(exc_info.value) == (
        "RuntimeError: run failed\nCaused by: ValueError: bad request"
    )


def test_three_level_chain() -> None:
    def raise_outer() -> None:
        try:
            _raise_wrapped()
        except RuntimeError as err:
            msg = "outer"
            raise OSError(msg) from err

    with pytest.raises(OSError) as exc_info:
        raise_outer()
    assert format_error_chain(exc_info.value) == (
        "OSError: outer"
        "\nCaused by: RuntimeError: run failed"
        "\nCaused by: ValueError: bad request"
    )


def test_implicit_context_is_included() -> None:
    def raise_during_handling() -> None:
        try:
            _raise_root()
        except ValueError:
            msg = "during handling"
            raise RuntimeError(msg)  # noqa: B904 — implicit context is the point

    with pytest.raises(RuntimeError) as exc_info:
        raise_during_handling()
    assert format_error_chain(exc_info.value) == (
        "RuntimeError: during handling\nCaused by: ValueError: bad request"
    )


def test_suppressed_context_is_excluded() -> None:
    def raise_clean() -> None:
        try:
            _raise_root()
        except ValueError:
            msg = "clean"
            raise RuntimeError(msg) from None

    with pytest.raises(RuntimeError) as exc_info:
        raise_clean()
    assert format_error_chain(exc_info.value) == "RuntimeError: clean"


def test_cyclic_chain_terminates() -> None:
    a = ValueError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert format_error_chain(a) == "ValueError: a\nCaused by: RuntimeError: b"
