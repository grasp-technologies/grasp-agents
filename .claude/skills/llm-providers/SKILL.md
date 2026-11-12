---
name: llm-providers
description: How to change or add an LLM provider in grasp-agents — the CloudLLM subclass contract (ClassVars, settings TypedDict with extra="allow", API and conversion layers, error mapping), per-provider converters to and from the internal OpenResponses-shaped items and events, the streaming/terminal-event contract, cross-provider reasoning and thought signatures, platforms vs api_provider, and which tests to run. Use before touching anything under llm/ or llm_providers/, or when exposing a new SDK parameter.
---

# LLM providers

Read `docs/llm.md` first. It describes the public behavior that this skill
explains how to preserve.

## Layout of a provider

Each provider lives in `llm_providers/<name>/` and owns its converters. There is no
shared intermediate format beyond our own `types/` items and events:

| File | Role |
|---|---|
| `<name>_llm.py` | The `CloudLLM` subclass, its `*LLMSettings` TypedDict, platform configs |
| `response_to_provider_inputs.py` | Our input items → the vendor request (messages, tool results, reasoning replay) |
| `provider_output_to_response.py` | Vendor response → `Response` (output items, usage) |
| `llm_event_converters.py` | Vendor stream → our `LlmEvent`s |
| `tool_converters.py` | `BaseTool` schemas → vendor tool definitions (strict mode where supported) |
| `error_mapping.py` | Vendor exceptions → typed `LlmError`s with a recovery hint, including content-filter detection |

Providers are imported lazily: `llm_providers/__init__.py` maps public names to
submodules (`_SUBMODULE_BY_NAME`). Add new public names there and to that
subpackage's `__all__`. Don't re-export providers from the root package. Vendor SDKs
are optional extras in `pyproject.toml`.

## The `CloudLLM` contract (`llm/cloud_llm.py`)

1. **ClassVars:**
   - `_settings_type`: the provider's settings TypedDict.
   - `_native_provider_name`: the API family, e.g. `"anthropic"`.
   - `_native_api_key_env_vars`: key env vars, in precedence order.
   - `_cloud_platforms`: platforms with a dedicated client, configured via
     `platform_config`.
   - `_litellm_provider_aliases`: names LiteLLM knows differently, used for
     cost lookups.
2. **Settings.** Decorate the TypedDict with `@with_config(ConfigDict(extra="allow"))`.
   Declared keys are type-checked at construction; undeclared keys are forwarded to
   the SDK untouched. Mirror the vendor SDK's own parameter names and shapes. Don't
   invent a lowest-common-denominator schema. Expose a new SDK parameter by
   declaring it, not by special-casing it.
3. **The API layer:** `_get_api_response` / `_get_api_stream`. The conversion
   layer: `_convert_api_response` / `_convert_api_stream`. Input preparation:
   `_make_api_input`. Error mapping: `_map_api_error`. Retries, fallback, rate
   limiting, validation and the terminal event live in the base classes, so don't
   re-implement them per provider.
4. **`platform=` vs `api_provider=`.**
   - `platform=` is a cloud that needs a dedicated client or endpoint
     (Bedrock, Bedrock Mantle, Vertex, Azure).
   - `api_provider=` points the vendor's own SDK at any endpoint speaking that API.
     Its `name` is the pricing and capability identity.
   - Model-name prefix routing does not exist. Don't add it back.

## Conversion rules that are easy to break

- **Items are final when emitted.** Once an `OutputItemDone` event is yielded, never
  mutate that item. Attach provider-specific fields (signatures, ids) while the item
  is still open.
- **The terminal event comes last, and only after validation.** Retries and
  fallbacks surface as `ResponseRetrying` / `ResponseFallback` events. The
  usage of superseded attempts is kept in `Response.superseded_usage`.
- **A cut-off stream must not be reported as `completed`.** It raises the typed
  stream-end error.
- **Unknown vendor items** round-trip as `UnknownItem` rather than being dropped.
- **Foreign reasoning.** Reasoning items are tagged with the producing model's
  `origin`: the vendor on its own API, `"<vendor>:<platform>"` on a cloud
  platform. `native_provider_name` stays the vendor on any platform. A
  consumer drops reasoning from another origin, and untagged reasoning carrying
  an encrypted payload, before sending.
- **Thought signatures** (`llm/thought_signatures.py`):
  - Gemini signs function calls and the message, never the reasoning item.
  - LiteLLM encodes the signature into the tool-call id (`…__thought__<sig>`).
    Split it off into `provider_specific_fields` on both the stream and
    non-stream paths.
  - Unsigned function calls sent to Gemini 3+ get the documented placeholder
    signature.
  - Don't patch signatures at end of stream.
- **Strip internal-only part fields** before serializing to the wire, so that
  framework metadata never reaches a vendor request.
- **Image tool outputs.** The Chat Completions API can't carry images in tool
  results. Plots from `RunPython`/`RunCell` need Responses, Anthropic or Gemini.

## Tests to run

- The provider's unit tests in `tests/<provider>/`: converters, stream events,
  error mapping, cloud platforms.
- `tests/llm/` for the base pipeline (retries, fallback, validation, terminal
  events).
- Live: `tests/<provider>/test_integration.py -m integration` with the cheap model
  (see the `writing-tests` skill).
- After touching LiteLLM/Gemini signature handling, also run
  `tests/litellm/test_integration.py::TestLiteLLMGeminiThoughtSignatures` and the
  offline replay test `tests/litellm/test_stream_converter.py::TestThoughtSignatureReplay`.
  The replay test feeds our converted inputs to LiteLLM's pure Gemini message
  transformer to check signature placement without network access.
- After touching reasoning replay in any provider, run
  `tests/integration/test_foreign_reasoning_live.py -m integration`. It replays
  reasoning across every producer and consumer pair (about a minute on cheap
  models).

## Adding a provider

Copy the structure of the closest existing provider, then:
1. Fill in the ClassVars and the settings TypedDict.
2. Write the converters and error mapping.
3. Register lazy imports, add the extra, and add `tests/<provider>/` mirroring an
   existing provider's test files.
4. Document it in `docs/llm.md`, in the same change.

Prefer `LiteLLM` for long-tail providers. A native provider is worth its
maintenance cost only when it exposes features LiteLLM flattens.
