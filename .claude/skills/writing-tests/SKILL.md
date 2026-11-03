---
name: writing-tests
description: How to write and run tests in grasp-agents — where a test goes, the shared MockLLM / Response builders / sample tools in tests/_helpers.py vs fixtures in conftest.py, async conventions, driving an agent or AgentLoop deterministically, and live integration tests (marker, cheapest models and their API quirks, telling sandbox/network artifacts from real failures). Use before adding or changing any test, or when a live test fails.
---

# Writing tests

## Where a test goes

`tests/` mirrors `src/grasp_agents/`: put a test in the subdir of the package it
exercises (`tests/agent/`, `tests/durability/`, `tests/sandbox/`, …). Provider tests
go in `tests/{anthropic,gemini,litellm,openai_completions,openai_responses}/`, and
end-to-end live runs go in `tests/integration/`.
- Name files by subsystem (`test_context_window.py`), never by fix, PR or task.
- Keep history and audit references out of test docstrings.
- If a test reads a sibling file via `__file__`, keep the file next to the test.

## Shared code

| Need | Use |
|---|---|
| Deterministic LLM | `tests._helpers.MockLLM(responses_queue=[...])` pops one `Response` per call and exposes `call_count` |
| Canned responses | `_text_response(text)`, `_tool_call_response(name, arguments_json, call_id)`, `_make_usage()` |
| Sample tools | `AddTool`, `MultiplyTool` (and the `tools` / `parallel_tools` fixtures) |
| Retry paths | `FailFirstLLM` |
| A bare `AgentContext` / `AgentLoop` | `_make_agent_ctx(...)`, `_make_agent_loop(...)` |
| Snapshot-capable environment | `FakeSnapshotEnv(root)` |
| Sandbox capability probes | `process_is_sandboxed()`, `srt_can_run()`, `exec_confinement()` |
| Team tests | `_agent(name, responses)`, `_send(to, message, call_id)`, `_until(pred)` |

- Importable helpers live in `tests/_helpers.py`, and pytest fixtures live in
  `tests/conftest.py`. Import absolutely (`from tests._helpers import MockLLM`);
  `from ..x` is banned (ruff `TID252`).
- A few test modules keep a deliberately divergent local `MockLLM` (indexing
  instead of popping, a default fallback answer). Read the body before replacing
  one with the shared helper.
- `conftest.py` has an autouse fixture that resets the process-default
  `SessionContext` around every test. Bare-constructed agents therefore never
  leak state, usage or stores between tests. Pass an explicit `ctx=` (or use
  `with SessionContext(...)`) when a test needs a shared session.
- `LLM.__deepcopy__` returns `self`. Replicas made by `ParallelProcessor` share
  one `MockLLM` queue, so seed one response per expected call.

```python
import pytest

from grasp_agents import LLMAgent
from tests._helpers import AddTool, MockLLM, _text_response, _tool_call_response


@pytest.mark.asyncio
async def test_agent_calls_tool_then_answers() -> None:
    llm = MockLLM(
        responses_queue=[
            _tool_call_response("add", '{"a": 2, "b": 3}', "call_1"),
            _text_response("5"),
        ]
    )
    agent = LLMAgent[str, str, None](name="calc", llm=llm, tools=[AddTool()])
    out = await agent.run("2 + 3?")
    assert out.payloads[0] == "5"
    assert llm.call_count == 2
```

The tool-call name must match the tool's registered `name` (here `"add"`).

## Async

- `asyncio_mode = "strict"`: mark async tests with `@pytest.mark.asyncio`. Don't use
  `anyio` marks or fixtures.
- In a module that mixes sync and async tests, decorate the async tests one by one.
  A module-level `pytestmark` would also mark the sync tests and produce warnings.
- Bound anything that waits (`asyncio.timeout`, or `_until` for polling), so a
  broken test fails instead of hanging.

## Integration (live) tests

- Mark them `@pytest.mark.integration`. `addopts` deselects them by default.
- Run them by file, in the foreground:
  `uv run pytest -m integration tests/<path> --tb=short`.
- API keys come from the environment or a local `.env` (gitignored). Key fixtures
  (`anthropic_api_key`, `openai_api_key`, `google_api_key`) skip the test when the
  key is missing, and they must go through `_require_env_key` so a failing test
  never prints a key.
- **Use the cheapest tier** that exhibits the behaviour under test, because wire
  formats are the same across tiers of a family. Current picks and their quirks:

| Provider | Model | Quirks |
|---|---|---|
| OpenAI | `gpt-6-luna` | On Chat Completions it rejects function tools unless `reasoning_effort="none"`. Through `LiteLLM`, also pass `allowed_openai_params=["reasoning_effort"]` as a constructor field, not in `llm_settings`. On Responses it emits a reasoning item by default; use `medium` effort and no forced `tool_choice` when a test needs reasoning |
| Anthropic | `claude-haiku-5-5` | Rejects `thinking.type="enabled"`. Use `{"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}` |
| Gemini | `gemini-3.5-flash-lite` | Low thinking effort yields no reasoning on simple prompts, so use `medium` |

- Some tests pin a specific model on purpose (feature-support pairs, a nonexistent
  model for error paths). Keep those pins.
- **Diagnosing failures.** HTTP 429 with `insufficient_quota` is an account
  problem, not a bug. HTTP 400 for an invalid model or parameter is a real bug. In a
  sandboxed or proxied environment, `Operation not permitted`,
  `CONNECT tunnel failed, response 403` and `invalid peer certificate` are
  environment artifacts.
- Model behaviour is flaky in known ways. For example, a coordinator may re-call a
  running background tool. Assert on framework guarantees, not on a model choosing
  well.
- After a live run, read what the agent actually did (tool-call arguments, final
  answer) before writing assertions. See the `dev-checks` skill, step 5.
