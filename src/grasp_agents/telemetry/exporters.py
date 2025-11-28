from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util.types import Attributes

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

CLOUD_PROVIDERS_NAME = {"metadata.google.internal"}


class NoopExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS


class FilteringExporter(SpanExporter):
    def __init__(
        self,
        inner: SpanExporter,
        llm_provider_blocklist: set[str] | None = None,
        attribute_filter: dict[str, set[str]] | None = None,
    ):
        self._inner = inner
        self._llm_provider_blocklist = llm_provider_blocklist or set()
        self._attribute_filter = attribute_filter or {}

    def _filter_based_on_llm_provider(self, attrs: Attributes) -> bool:
        attrs = attrs or {}
        provider = attrs.get("gen_ai.system", "") or attrs.get(
            "gen_ai.provider.name", ""
        )
        return provider not in self._llm_provider_blocklist

    def _filter_based_on_attrs(self, attrs: Attributes) -> bool:
        attrs = attrs or {}
        for name, values in self._attribute_filter.items():
            atr_value = attrs.get(name, "")
            if atr_value in values:
                return False
        return True

    def export(self, spans: Sequence[ReadableSpan]):
        keep: list[ReadableSpan] = []
        for s in spans:
            if self._filter_based_on_attrs(
                s.attributes
            ) and self._filter_based_on_llm_provider(s.attributes):
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
