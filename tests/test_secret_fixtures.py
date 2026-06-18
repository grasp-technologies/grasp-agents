"""
The API-key fixtures must never leak a key into a pytest traceback.

pytest renders a failing test's fixture arguments and assertion operands via
``repr`` (``saferepr``). The key fixtures therefore return a ``str`` subclass
whose ``repr`` is redacted, while the value stays usable for authentication.
"""

from __future__ import annotations

import pytest

from tests.conftest import _require_env_key, _SecretStr


def test_secret_str_redacts_repr_but_keeps_value() -> None:
    secret = _SecretStr("sk-supersecret-0123456789")
    # The exact path that printed the key before: pytest shows fixture args /
    # assert operands via repr.
    assert repr(secret) == "'***'"
    # ...yet it is still the real key everywhere it is actually used (auth).
    assert secret == "sk-supersecret-0123456789"
    assert str(secret) == "sk-supersecret-0123456789"
    assert isinstance(secret, str)


def test_saferepr_redacts() -> None:
    from _pytest._io.saferepr import saferepr

    assert saferepr(_SecretStr("sk-supersecret-0123456789")) == "'***'"


def test_require_env_key_returns_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_TEST_KEY", "sk-fake-value")
    key = _require_env_key("FAKE_TEST_KEY")
    assert key == "sk-fake-value"  # usable
    assert repr(key) == "'***'"  # redacted
    assert type(key) is _SecretStr
