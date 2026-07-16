import pytest

from grasp_agents.utils.errors import format_error_chain, root_cause


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


def test_root_cause_no_chain_returns_self() -> None:
    err = ValueError("boom")
    assert root_cause(err) is err


def test_root_cause_walks_to_deepest() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        _raise_wrapped()
    root = root_cause(exc_info.value)
    assert type(root) is ValueError
    assert str(root) == "bad request"


def test_root_cause_stops_at_suppressed_context() -> None:
    def raise_clean() -> None:
        try:
            _raise_root()
        except ValueError:
            msg = "clean"
            raise RuntimeError(msg) from None

    with pytest.raises(RuntimeError) as exc_info:
        raise_clean()
    assert type(root_cause(exc_info.value)) is RuntimeError


def test_root_cause_cyclic_chain_terminates() -> None:
    a = ValueError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert root_cause(a) in {a, b}
