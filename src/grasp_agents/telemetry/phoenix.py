import os
from logging import getLogger

from openinference.instrumentation.litellm import (
    LiteLLMInstrumentor as OILiteLLMInstrumentor,
)
from openinference.instrumentation.openai import (
    OpenAIInstrumentor as OIOpenAIInstrumentor,
)
from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from phoenix.otel import BatchSpanProcessor, HTTPSpanExporter, SimpleSpanProcessor

from .exporters import LLM_PROVIDER_NAMES, FilteringExporter

logger = getLogger(__name__)


def init_phoenix(
    batch: bool = False,
    use_litellm_instr: bool = True,
    use_llm_provider_instr: bool = True,
):
    collector_endpoint = os.getenv("TELEMETRY_COLLECTOR_HTTP_ENDPOINT")

    if not collector_endpoint:
        logger.warning(
            "TELEMETRY_COLLECTOR_HTTP_ENDPOINT not set, cannot initialize Phoenix"
        )
        return

    # 1) Get or create tracer provider

    tracer_provider = trace_api.get_tracer_provider()

    if not isinstance(tracer_provider, TracerProvider):
        logger.info("No existing tracer provider, cannot initialize Phoenix")
        return

    # 2) Convert spans to OpenInference format expected by Phoenix
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

    # 3) Export to Phoenix backend
    # Use FilteringExporter to block LLM provider spans that are
    # already captured by OpenInference instrumentations
    blocklist: set[str] = (
        LLM_PROVIDER_NAMES if use_llm_provider_instr or use_litellm_instr else set()
    )
    exporter = FilteringExporter(
        inner=HTTPSpanExporter(endpoint=collector_endpoint, headers=None),
        blocklist=blocklist,
    )
    if batch:
        span_processor = BatchSpanProcessor(span_exporter=exporter)
    else:
        span_processor = SimpleSpanProcessor(span_exporter=exporter)
    tracer_provider.add_span_processor(span_processor)

    # 4) Replace OpenTelemetry instrumentations with OpenInference ones for better
    # compatibility with the Phoenix backend

    if use_litellm_instr:
        OILiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    if use_llm_provider_instr:
        OIOpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
