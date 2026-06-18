"""
Concrete LLM providers, one subpackage each. Import from the specific provider
you need — several require optional extras:

* :mod:`.openai_responses` — ``OpenAIResponsesLLM`` (OpenAI Responses API)
* :mod:`.openai_completions` — ``OpenAILLM`` (Chat Completions; also Gemini /
  OpenRouter OpenAI-compatible endpoints)
* :mod:`.anthropic` — ``AnthropicLLM`` (needs the ``anthropic`` extra)
* :mod:`.gemini` — ``GeminiLLM`` (needs the ``gemini`` extra)
* :mod:`.litellm` — ``LiteLLM`` (long-tail providers via ``litellm``)

Kept import-free so this package never eagerly pulls a provider whose extra
isn't installed.
"""
