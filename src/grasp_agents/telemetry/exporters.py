from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# Set of LLM provider names used in OpenTelemetry attributes
# See https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#gen-ai-provider-name
LLM_PROVIDER_NAMES = {
    "anthropic",
    "aws.bedrock",
    "azure.ai.inference",
    "azure.ai.openai",
    "cohere",
    "deepseek",
    "gcp.gemini",
    "gcp.gen_ai",
    "gcp.vertex_ai",
    "groq",
    "ibm.watsonx.ai",
    "mistral_ai",
    "openai",
    "perplexity",
    "x_ai",
}


class NoopExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS


class FilteringExporter(SpanExporter):
    def __init__(self, inner: SpanExporter, blocklist: set[str] | None = None):
        self._inner = inner
        self._blocklist = blocklist or set()

    def export(self, spans: Sequence[ReadableSpan]):
        keep: list[ReadableSpan] = []
        for s in spans:
            attrs = s.attributes or {}
            provider = attrs.get("gen_ai.system", "") or attrs.get(
                "gen_ai.provider.name", ""
            )
            if provider not in self._blocklist:
                keep.append(s)

        return SpanExportResult.SUCCESS if not keep else self._inner.export(keep)

    def shutdown(self):
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


class LogScopeExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            scope = getattr(s, "instrumentation_scope", None)
            print(
                "SCOPE:",
                getattr(scope, "name", ""),
                "SPAN:",
                s.name,
                "ATTRS:",
                s.attributes,
            )
        return SpanExportResult.SUCCESS
