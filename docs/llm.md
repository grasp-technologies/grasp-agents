# LLM layer

The LLM layer turns the framework's provider-neutral items into one provider's
API calls and back. Each provider ships its own converters; there is no shared
intermediate wire format and no lowest-common-denominator settings object.
On top of the providers sit the retry layer (`RetryPolicy`), the model cascade
(`FallbackLLM`), typed errors, cost and capability metadata, and rate limiting.

## Key types

| Name | Module | Role |
|---|---|---|
| `LLM`, `LLMSettings` | `llm/llm.py` | Abstract base: retry + validation pipeline, `generate_response` / `generate_response_stream` |
| `CloudLLM`, `CloudLLMSettings`, `APIProvider` | `llm/cloud_llm.py` | Base for API-backed providers: endpoint resolution, error mapping, cost stamping, rate limiting, foreign-reasoning handling |
| `OpenAIResponsesLLM` | `llm_providers/openai_responses/` | OpenAI Responses API |
| `OpenAILLM` | `llm_providers/openai_completions/` | OpenAI Chat Completions, and any endpoint speaking it |
| `AnthropicLLM` | `llm_providers/anthropic/` | Anthropic Messages API (`anthropic` extra) |
| `GeminiLLM` | `llm_providers/gemini/` | Google GenAI SDK (`gemini` extra) |
| `LiteLLM` | `llm_providers/litellm/` | Long-tail providers through `litellm` |
| `FallbackLLM` | `llm/fallback_llm.py` | Tries member LLMs in order |
| `RetryPolicy` | `llm/resilience.py` | API and validation retry configuration |
| `ModelCapabilities`, `get_model_capabilities`, `get_context_window`, `count_tokens`, `count_input_tokens`, `estimate_input_tokens` | `llm/model_info.py`, `llm/token_counting.py` | Metadata and token counting from LiteLLM's model database |
| `Llm*Error`, `LlmErrorTuple` | `types/llm_errors.py` | Typed provider errors |
| `RecoveryHint`, `classify_error`, `register_recovery_hint` | `types/recovery.py` | What action recovers from an error |
| `RateLimiter` | `rate_limiting/` | Requests-per-minute pacing + concurrency cap |

`LLM`, `FallbackLLM`, `LLMSettings` and `RetryPolicy` are root exports. Import
providers from `grasp_agents.llm_providers`; the names there load lazily, so a
provider's SDK is imported only when its class is first accessed, and a missing
extra raises an `ImportError` naming it:

```python
from grasp_agents.llm_providers import AnthropicLLM, OpenAIResponsesLLM

llm = OpenAIResponsesLLM(model_name="gpt-5", llm_settings={"reasoning": {"effort": "low"}})
```

| Extra | Adds |
|---|---|
| (base install) | `OpenAIResponsesLLM`, `OpenAILLM`, `LiteLLM` |
| `anthropic` / `gemini` | `AnthropicLLM` / `GeminiLLM` |
| `bedrock` | Claude on AWS Bedrock (SigV4 signing) |
| `vertex` | Claude and Gemini on Google Vertex AI |
| `all-llm-providers` | all of the above |

Azure OpenAI needs no extra.

## Items and events

Internally every conversation is a list of OpenResponses-shaped items
(`types/items.py`): input/output messages, function tool calls and outputs,
reasoning items, `WebSearchCallItem`, and `UnknownItem`. A `Response`
(`types/response.py`) holds output items plus `usage`. Streaming yields
`LlmEvent`s (`types/llm_events.py`) shaped like OpenAI Responses stream events,
plus two framework events: `ResponseRetrying` and `ResponseFallback`.

`UnknownItem` preserves a provider item type the framework does not model,
field for field, so it round-trips through the transcript. Only
`OpenAIResponsesLLM` sends these back; the other providers skip them.

## Endpoints: `platform=` vs `api_provider=`

Two separate axes on `CloudLLM`:

- `platform=` selects a cloud platform that has its own dedicated client.
  Its client args go in a typed `platform_config`. Unset values fall back to
  the platform's own credential chain and environment variables.
- `api_provider=APIProvider(name=, base_url=, api_key=)` points the vendor's
  own SDK client at any endpoint that speaks the same API (a gateway, a
  self-hosted server). Without it, the vendor's endpoint and API-key env var
  are used.

| Provider | `platform` values | `platform_config` type | API-key env vars |
|---|---|---|---|
| `OpenAIResponsesLLM`, `OpenAILLM` | `"azure"`, `"bedrock"`, `"bedrock_mantle"` | `AzureClientConfig` / `BedrockOpenAIClientConfig` | `OPENAI_API_KEY` |
| `AnthropicLLM` | `"bedrock"`, `"bedrock_mantle"`, `"vertex"` | `BedrockClientConfig` / `VertexClientConfig` | `ANTHROPIC_API_KEY` |
| `GeminiLLM` | `"vertex"` | `GeminiVertexClientConfig` | `GOOGLE_API_KEY`, `GEMINI_API_KEY` |
| `LiteLLM` | — (routes by model name) | — | litellm's own resolution |

Amazon Bedrock serves both the OpenAI and the Anthropic API on two endpoints,
the Bedrock runtime (`"bedrock"`) and Bedrock Mantle (`"bedrock_mantle"`). The
OpenAI classes reach both through the OpenAI SDK's `AsyncBedrockOpenAI`, at
`/openai/v1` on each; `AnthropicLLM` through `AsyncAnthropicBedrock` and
`AsyncAnthropicBedrockMantle`. On the runtime a model is called by an inference
profile (`"global.openai.gpt-6-luna"`), on Mantle by its plain ID
(`"openai.gpt-6-luna"`).

```python
from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.llm_providers import AnthropicLLM, OpenAILLM

claude = AnthropicLLM(
    model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
    platform="bedrock",
    platform_config={"aws_region": "us-east-1"},
)
gateway = OpenAILLM(
    model_name="google/gemma-4-31b-it",
    api_provider=APIProvider(name="openrouter", base_url="https://openrouter.ai/api/v1", api_key=key),
)
```

Every provider also takes `http_client`, `default_headers` and an
`extra_<vendor>_client_params` escape hatch passed verbatim to the SDK client
(`LiteLLM` does not support a custom `http_client`).

**Cost identity follows the endpoint.** Pricing and capability lookups use the
provider actually serving the model: the platform (aliased to LiteLLM's name,
e.g. `vertex` → `vertex_ai`), else an `api_provider` that has a `base_url`,
else the vendor (`CloudLLM._resolve_litellm_provider`). Pass `litellm_provider=`
to pin it.

## Settings

Each provider has a settings `TypedDict` (`OpenAIResponsesLLMSettings`,
`OpenAILLMSettings`, `AnthropicLLMSettings`, `GeminiLLMSettings`,
`LiteLLMSettings`) built from the provider SDK's own parameter types. They are
validated at construction with pydantic under `extra="allow"`: declared keys
are type-checked, undeclared keys are forwarded to the provider untouched.
Per-call overrides come from a before-LLM hook mutating `extra_llm_settings`
(see [Hooks](hooks.md)).

**Structured output.** By default the agent's `llm_output_schema` and tool
argument schemas are validated client-side only. Set
`apply_output_schema_via_provider=True` to also send the output schema to the
provider (this also turns on `apply_tool_call_schema_via_provider`). With
`apply_tool_call_schema_via_provider=True`, tool schemas go out in the
provider's strict form (OpenAI strict JSON schema, Anthropic `strict=True`
with `transform_schema`, Gemini `VALIDATED` function-calling mode).

**Provider-native web search** is a setting: `web_search` (Responses,
Anthropic), `web_fetch` (Anthropic), `google_search` / `url_context` (Gemini).
Calls surface as `WebSearchCallItem`s.

## The generation pipeline

`LLM.generate_response_stream` (and the non-streaming twin) wraps each call in
two retry layers:

1. **API retries** (inner): transient errors, classified by `RecoveryHint`
   (`TRANSIENT`, `RATE_LIMITED`), are retried with exponential backoff and
   uniform jitter. A rate-limit `retry_after` is a floor, capped by
   `max_retry_after`.
2. **Validation retries** (outer): a response whose tool calls name an unknown
   tool or carry invalid arguments (`LLMToolCallValidationError`), or whose
   text fails `output_schema` (`LLMResponseValidationError`), is re-sampled
   immediately, up to `validation_retries`.

```python
from grasp_agents import RetryPolicy

RetryPolicy(api_retries=3, initial_delay=1.0, max_delay=60.0, exponential_base=2.0,
            jitter=0.5, max_retry_after=300.0, validation_retries=0)
```

`retry_policy=None` disables both layers. Provider SDK client retries default
to 0 so they never multiply with this layer.

**Streaming contract.**

- The terminal event (`ResponseCompleted` / `ResponseIncomplete`) is yielded
  only after the response passed validation. A consumer that acts on the
  terminal event never acts on a response that is about to be retried.
- Each retry yields `ResponseRetrying(attempt=, error=)` before the next
  attempt; each cascade step yields `ResponseFallback(failed_model=,
  fallback_model=, error_type=, attempt=)`. Each layer counts its own attempts.
- A failure after the terminal event neither retries nor raises; the
  delivered response is kept.
- A stream that ends with no terminal event, or with a `response.failed`
  event, raises a retryable `LlmInternalServerError`.
- Usage of attempts a retry replaced is kept in `Response.superseded_usage`,
  separate from `usage`. The session `UsageTracker` counts both.

## FallbackLLM

```python
from grasp_agents import FallbackLLM

llm = FallbackLLM(primary=primary_llm, fallbacks=(backup_a, backup_b))
```

Retries come first: each member runs its whole pipeline (its own API retries,
validation and validation retries) before the cascade moves on. The cascade
advances on exhausted or deterministic API errors and on
`LLMResponseValidationError` once the member's retries are spent. A content
filter block also advances it. A context-window overflow does not: it is raised
at once, without trying the other members, because the input has to shrink
(the agent loop compacts and retries). When every member fails, the raised error
prefers a retryable one (shortest implied wait).

Constructor rules (`FallbackLLM.__post_init__`): `retry_policy`,
`llm_settings` and `litellm_provider` must be `None`; set them on the members.
`model_name` defaults to the primary's. `capabilities` is the conservative
merge: a feature only if every member has it, the smallest known token windows.
Each member stamps its own cost on the responses it serves.

## Errors

Providers map every SDK exception to a type in `types/llm_errors.py`
(`CloudLLM._map_api_error`); these subclass the OpenAI SDK's error types, so
`except openai.RateLimitError` also works. Retry and fallback act only on
`LlmErrorTuple` — an unmapped exception is fatal.

| Hint (`classify_error`) | Errors | Retried |
|---|---|---|
| `TRANSIENT` | timeout, connection, 5xx, 409, other status errors | yes |
| `RATE_LIMITED` | `LlmRateLimitError` | yes |
| `QUOTA_EXCEEDED` | `LlmQuotaExceededError` | no |
| `NEEDS_COMPACTION` | `LlmContextWindowError` | no (with compaction enabled, the agent loop forces a fold and retries once; see [Context window](context-window.md)) |
| `REAUTH_REQUIRED` | 401, 403 | no |
| `CONTENT_REFUSED` | `LlmContentFilterError` | no |
| `INVALID_REQUEST` | 400, 404, 422 | no |
| `INVALID_RESPONSE` | `LlmResponseSchemaError` (our types rejected the provider's response) | no |

`LlmContextWindowError` is OpenAI's `context_length_exceeded` code, Anthropic's
400 "prompt is too long: … tokens > … maximum" and 413, Gemini's 400 "The input
token count (…) exceeds the maximum number of tokens allowed (…)", OpenAI's own
overflow wording when a gateway drops the code, and LiteLLM's
`ContextWindowExceededError`. Anthropic and Gemini give the overflow no
structured marker, so it is matched on the message
(`_http_helpers.is_context_window_message`).

Give your own exception a `recovery_hint` class attribute, or call
`register_recovery_hint(exc_type, hint)` for types you don't own.

**Content filters vs refusals.** A provider block (normalized to
`incomplete_details.reason == "content_filter"`) raises `LlmContentFilterError`
in every provider: there is no output to act on. A refusal the model wrote is
only logged as a warning; its text enters the transcript so the agent can
correct course, and a structured-output call re-samples through
`validation_retries`.

**Logging policy.** The layer that swallows or translates an exception logs it
with `exc_info`; layers that re-raise do not log. `utils/errors.py` →
`format_error_chain` renders the full cause chain for logs and model-facing
error strings. See [Observability](observability.md).

## Prompt caching

`CacheControl(ttl=None | "5m" | "1h")` (`types/content.py`, root export) on a
content part asks the provider to cache the prompt prefix up to and including
that part. `SystemPromptSection(cache_control=...)` puts it on a section's
block. Anthropic sends it as `cache_control`; OpenAI (Responses and Chat
Completions) as an explicit `prompt_cache_breakpoint`; the others ignore it.
OpenAI's TTL is set per request through the `prompt_cache_options` setting.
The MCP instructions section carries a breakpoint by default.

`AnthropicLLM` caches the conversation itself through the top-level
`cache_control` setting (`AnthropicLLMSettings`). It defaults to
`{"type": "ephemeral"}` (5 minutes) on every platform; set `{"type": "ephemeral",
"ttl": "1h"}` for an hour, or `None` to turn it off. Bedrock rejects the setting
for Claude Opus 4.6 and older models with a 400: set `None` for those. Every request then writes a cache entry ending at its last
block, and the next request in an agent loop, compaction's summary request
included, reads it. Anthropic finds an earlier entry only within about 20 blocks
of a request's end, so a turn that adds more blocks than that (many parallel
tool calls) misses once. A request no later request extends pays the
cache-write premium for nothing. OpenAI and Gemini cache shared prefixes on
their own; on OpenAI the `prompt_cache_retention` setting (`"in_memory"` or
`"24h"`) chooses how long, and the provider picks when it is unset.

## Model metadata, tokens, cost

`llm.capabilities` (`ModelCapabilities`: `function_calling`, `vision`,
`output_schema`, `prompt_caching`, `reasoning`, `web_search`, `audio_input`,
`max_input_tokens`, `max_output_tokens`) comes from LiteLLM's model database for
the resolved provider. An unknown model gets permissive defaults (every
feature `True`, no limits) and a warning. `count_input_tokens(model, items)`
counts a view with LiteLLM's tokenizer and estimates files and encrypted
reasoning separately, falling back to `estimate_input_tokens(items)`, a
tokenizer-free character estimate (what compaction uses). `CloudLLM`
stamps `usage.cost` from LiteLLM pricing when the response is produced.

## Rate limiting

`CloudLLM(rate_limiter=RateLimiter(rpm=60, max_concurrency=200))` paces API
calls (minimum spacing of 60/rpm seconds) and caps concurrency. One limiter
can be shared between LLM instances. For streaming it paces stream acquisition.

## Reasoning across providers

Items a model produces are stamped when a backend verifies their payload
(reasoning items; messages and tool calls carrying a thought signature) with
the model's `native_provider_name`, the vendor (`"openai"`, `"anthropic"`,
`"gemini"`) whatever platform serves it, and its `platform` (`None` for the
vendor's own API, else for example `"bedrock"`, `"bedrock_mantle"`,
`"vertex"`). Both are compared: a platform encrypts reasoning with its own
keys, and one platform serves several vendors. The OpenAI Responses and Chat
Completions classes on one platform match. An `api_provider` endpoint has no
platform, so reach a cloud platform through `platform=`.

Before a request, `CloudLLM` drops reasoning items stamped by another vendor
or platform, and untagged reasoning items carrying an encrypted payload (stored before
stamping, so no backend is known to have signed them). It strips foreign
thought signatures from messages and tool calls. The transcript is never
modified, so switching models or endpoints mid-conversation, or a `FallbackLLM`
cascade, does not send a payload the backend would reject; the cost is the
dropped reasoning. Gemini thought signatures arriving through `LiteLLM` are
normalized in `llm/thought_signatures.py`.

## Adding a provider

Subclass `CloudLLM` (a frozen dataclass) and:

1. Set the ClassVars: `_settings_type` (your settings `TypedDict`, decorated
   with `@with_config(ConfigDict(extra="allow"))`), `_native_provider_name`,
   `_native_api_key_env_vars`, `_cloud_platforms`, `_litellm_provider_aliases`.
2. In `__post_init__` (after `super().__post_init__()`), build the client for
   the selected `platform` or the resolved `api_provider`, and set
   `litellm_provider` to `self._resolve_litellm_provider()`.
3. Implement `_make_api_input` (items → API input, tools, tool choice;
   per-call keys under `extra_settings`), `_get_api_response`, `_get_api_stream`,
   `_convert_api_response`, `_convert_api_stream` (yield `LlmEvent`s ending in a
   terminal event carrying the full `Response`), and `_map_api_error` (map
   *every* SDK exception, including content-filter detection).
4. Add the class to `llm_providers/__init__.py`'s lazy map and, if it needs an
   SDK, an extra in `pyproject.toml`.

`_get_api_response` / `_get_api_stream` are wrapped with the rate limiter
automatically (`CloudLLM.__init_subclass__`).

## Contracts

1. The framework retry layer is the only retry system; SDK client retries
   default to 0 (`*_client_max_retries`).
2. A terminal stream event is emitted only after validation
   (`LLM.generate_response_stream`).
3. Retry and fallback act only on `LlmErrorTuple`; providers must map every SDK
   exception (`CloudLLM._map_api_error`).
4. `FallbackLLM` takes no `retry_policy`, `llm_settings` or `litellm_provider`.
5. `platform` and `api_provider` are mutually exclusive; `platform_config`
   requires a `platform` (`CloudLLM._validate_platform`).
6. Credentials (`api_provider`, `platform_config`) are excluded from `repr`.
7. `LLM.__deepcopy__` returns `self`: copies of an agent share one LLM instance.

## Limitations

- No automatic server-side conversation state: the full transcript is sent
  each turn. `OpenAIResponsesLLM` honors `previous_response_id` / `conversation`
  passed per call (e.g. from a before-LLM hook) and then sends only new items.
- There is no cooldown tracking: a `FallbackLLM` member that just failed is
  tried first again on the next call.
- Retry jitter is uniform; there is no decorrelated-jitter or composable
  retry-policy API.

## See also

- [Agent loop](agent-loop.md) — how the loop consumes responses and tool-call
  validation errors.
- [Observability](observability.md) — usage, cost, logging gates.
- Code: `llm/`, `llm_providers/`, `types/llm_errors.py`, `types/recovery.py`,
  `types/llm_events.py`.
- Example: `src/grasp_agents/examples/notebooks/advanced_patterns.ipynb`.
