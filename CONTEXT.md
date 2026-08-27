# grasp-agents

Provider-agnostic LLM agent framework: one internal item vocabulary (messages,
reasoning, tool calls) converted to and from each provider's wire format, with
retry/fallback layered on top.

## Language

**Reasoning origin**:
The format identity of the API dialect that produced a reasoning item — e.g.
`openai_responses`, `anthropic`, `gemini`, `openai_completions`,
`litellm:<backend>`. Identifies the only dialect the item may be replayed to.
_Avoid_: provider (overloaded: connection config, vendor, converter package), vendor, model

**Foreign reasoning item**:
A reasoning item whose origin differs from the dialect a request is being built
for. Foreign items are dropped at request build; they can never be sanitized
into acceptance — both OpenAI and Anthropic cryptographically verify reasoning
payloads.

**Untagged reasoning item**:
A reasoning item with no origin, from a history persisted before origins
existed. Kept by every dialect — legacy data keeps its pre-origin behavior;
only provably foreign items are dropped.
