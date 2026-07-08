import importlib
import os
from logging import getLogger
from weakref import WeakSet

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from .exporters import CLOUD_PROVIDERS_NAMES, LLM_PROVIDER_NAMES, FilteringExporter
from .setup import init_tracing

logger = getLogger(__name__)

_phoenix_attached: WeakSet[TracerProvider] = WeakSet()


def _instrument(tracer_provider: TracerProvider, module: str, cls_name: str) -> None:
    """
    Apply an OpenInference instrumentor if it (and the SDK it patches) is present.

    Provider instrumentors and their SDKs are optional extras, so a provider the
    app doesn't use is skipped rather than failing telemetry setup.
    """
    try:
        instrumentor_cls = getattr(importlib.import_module(module), cls_name)
        instrumentor_cls().instrument(tracer_provider=tracer_provider)
        logger.debug("Phoenix: instrumented %s", cls_name)
    except Exception:
        logger.debug("Phoenix: skipped %s (not installed)", cls_name, exc_info=True)


def init_phoenix(
    batch: bool = False,
    use_litellm_instr: bool = True,
    use_llm_provider_instr: bool = True,
    project_name: str = "grasp-agents",
) -> None:
    """
    Attach the Phoenix exporter (and provider instrumentors) to the tracer
    provider. Re-entry against the same provider is a no-op — the span
    processors would otherwise duplicate and double every exported span.
    """
    # Deferred: these need the optional grasp-agents[phoenix] extra.
    from openinference.instrumentation.openllmetry import (  # noqa: PLC0415
        OpenInferenceSpanProcessor,
    )
    from phoenix.otel import (  # noqa: PLC0415
        BatchSpanProcessor,
        HTTPSpanExporter,
        SimpleSpanProcessor,
    )

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

    if tracer_provider in _phoenix_attached:
        logger.info("Phoenix already attached to this TracerProvider; skipping")
        return

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
    _phoenix_attached.add(tracer_provider)

    # Auto-instrument the provider SDKs with OpenInference instrumentors — one
    # per dedicated client (the openai one covers both the Responses and Chat
    # Completions APIs). Each is applied only if installed, so providers the app
    # doesn't use are skipped.
    if use_litellm_instr:
        _instrument(
            tracer_provider,
            "openinference.instrumentation.litellm",
            "LiteLLMInstrumentor",
        )
    if use_llm_provider_instr:
        for module, cls_name in (
            ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
            ("openinference.instrumentation.anthropic", "AnthropicInstrumentor"),
            ("openinference.instrumentation.google_genai", "GoogleGenAIInstrumentor"),
        ):
            _instrument(tracer_provider, module, cls_name)
