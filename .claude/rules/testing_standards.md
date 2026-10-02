# Testing Standards

Tests live in `tests/`, run with `uv run pytest`, and are configured in
`[tool.pytest.ini_options]` in [`pyproject.toml`](../../pyproject.toml).

## Async tests must be marked

`asyncio_mode = "strict"`. **Every async test needs an explicit
`pytest.mark.asyncio`; an unmarked `async def test_*` is not run — it is
skipped with a warning, so the suite stays green while the behavior ships
untested.** The convention here is a module-level marker at the top of the
file, not a decorator per test:

```python
pytestmark = pytest.mark.asyncio
```

When the whole module also needs real credentials, both markers go in the list:

```python
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
```

## Tests that need real API keys or the network

`addopts = "-m 'not integration' --ignore=tests/kits"`, so integration tests are
excluded from the default run. **Any test that calls a live provider, spends
API credit, or needs a real key must be marked `@pytest.mark.integration`** — an
unmarked one fails for every contributor without that key and in CI.

Conversely, a test that does *not* need credentials must not be marked
`integration`: it then never runs in the default suite, which is the same
silent-skip failure as a missing asyncio marker.

## Secrets in tests

API keys come from the fixtures in `tests/conftest.py` (`openai_api_key`,
`anthropic_api_key`, `google_api_key`). **Every API-key fixture must return its
key through `_require_env_key` so a leaked key can never reach a traceback** —
pytest renders fixture arguments and assertion operands via `repr`, and
`_SecretStr` is what keeps the value out of CI logs. Never read a key straight
from `os.environ` into a test, and never hardcode a real key.

## Shared building blocks

Pytest **fixtures** live in `tests/conftest.py`; the plain building blocks —
the queue-driven mock `LLM`, `Response` builders, simple `BaseTool` subclasses
— live in `tests/_helpers.py` and are imported explicitly. **If a helper is
already in `tests/_helpers.py`, import it rather than copying a local variant
into a new test module.**

Fake the **provider boundary** (the mock `LLM` and its queued responses), not
the framework internals: a test that patches an agent's private attributes
asserts the implementation instead of the behavior.

## Golden snapshots

The TUI/console tests compare rendered output against `.expected` files.
**Regenerate them with `uv run pytest <path> --update-golden`; never hand-edit
an `.expected` file** — the point of the snapshot is that a human reviewed the
diff the code produced.
