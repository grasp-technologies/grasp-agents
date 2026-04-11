import os
from logging import getLogger

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from .exporters import CLOUD_PROVIDERS_NAMES, LLM_PROVIDER_NAMES, FilteringExporter
from .setup import init_tracing

logger = getLogger(__name__)


def init_phoenix(
    batch: bool = False,
    use_litellm_instr: bool = True,
    use_llm_provider_instr: bool = True,
    project_name: str = "grasp-agents",
) -> None:
    from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
    from phoenix.otel import BatchSpanProcessor, HTTPSpanExporter, SimpleSpanProcessor

    collector_endpoint = os.getenv("TELEMETRY_COLLECTOR_HTTP_ENDPOINT")

    if not collector_endpoint:
        logger.warning(
            "TELEMETRY_COLLECTOR_HTTP_ENDPOINT not set, cannot initialize Phoenix"
        )
        return

    # Ensure TracerProvider exists
    tracer_provider = trace_api.get_tracer_provider()
    if not isinstance(tracer_provider, TracerProvider):
        tracer_provider = init_tracing(project_name=project_name)

    # Convert spans to OpenInference format expected by Phoenix
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

    # Export to Phoenix backend
    # Use FilteringExporter to block LLM provider spans that are
    # already captured by OpenInference instrumentations
    blocklist: set[str] = (
        LLM_PROVIDER_NAMES if use_llm_provider_instr or use_litellm_instr else set()
    )
    exporter = FilteringExporter(
        inner=HTTPSpanExporter(endpoint=collector_endpoint, headers=None),
        llm_provider_blocklist=blocklist,
        attribute_filter={"http.url": CLOUD_PROVIDERS_NAMES},
    )
    if batch:
        span_processor = BatchSpanProcessor(span_exporter=exporter)
    else:
        span_processor = SimpleSpanProcessor(span_exporter=exporter)
    tracer_provider.add_span_processor(span_processor)

    # Auto-instrument LLM providers with OpenInference instrumentors
    if use_litellm_instr:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    if use_llm_provider_instr:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
