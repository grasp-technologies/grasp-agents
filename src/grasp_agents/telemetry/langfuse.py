import base64
import importlib
import os
from logging import getLogger

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from .exporters import CLOUD_PROVIDERS_NAMES, LLM_PROVIDER_NAMES, FilteringExporter
from .setup import init_tracing

logger = getLogger(__name__)

# Langfuse ingests OTLP/HTTP traces at this path under the instance host root.
_LANGFUSE_OTEL_PATH = "/api/public/otel/v1/traces"


def _instrument(tracer_provider: TracerProvider, module: str, cls_name: str) -> None:
    """
    Apply an OpenInference instrumentor if it (and the SDK it patches) is present.

    Provider instrumentors and their SDKs are optional extras, so a provider the
    app doesn't use is skipped rather than failing telemetry setup.
    """
    try:
        instrumentor_cls = getattr(importlib.import_module(module), cls_name)
        instrumentor_cls().instrument(tracer_provider=tracer_provider)
        logger.debug("Langfuse: instrumented %s", cls_name)
    except Exception:
        logger.debug("Langfuse: skipped %s (not installed)", cls_name, exc_info=True)


def _build_endpoint(host: str) -> str:
    host = host.rstrip("/")
    if host.endswith(_LANGFUSE_OTEL_PATH):
        return host
    return f"{host}{_LANGFUSE_OTEL_PATH}"


def init_langfuse(
    batch: bool = False,
    use_litellm_instr: bool = True,
    use_llm_provider_instr: bool = True,
    project_name: str = "grasp-agents",
) -> None:
    # Deferred: these need the optional grasp-agents[langfuse] extra.
    from openinference.instrumentation.openllmetry import (  # noqa: PLC0415
        OpenInferenceSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not host:
        logger.warning("LANGFUSE_HOST not set, cannot initialize Langfuse")
        return
    if not (public_key and secret_key):
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set, "
            "cannot initialize Langfuse"
        )
        return

    # Ensure TracerProvider exists
    tracer_provider = trace_api.get_tracer_provider()
    if not isinstance(tracer_provider, TracerProvider):
        tracer_provider = init_tracing(project_name=project_name)

    # Convert OpenLLMetry spans to the OpenInference format Langfuse understands
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

    # Langfuse authenticates OTLP ingestion with HTTP Basic auth built from the
    # project's public/secret API keys: base64("<public_key>:<secret_key>").
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    # Export to Langfuse backend over OTLP/HTTP.
    # Use FilteringExporter to block LLM provider spans that are
    # already captured by OpenInference instrumentations
    blocklist: set[str] = (
        LLM_PROVIDER_NAMES if use_llm_provider_instr or use_litellm_instr else set()
    )
    exporter = FilteringExporter(
        inner=OTLPSpanExporter(endpoint=_build_endpoint(host), headers=headers),
        llm_provider_blocklist=blocklist,
        attribute_filter={"http.url": CLOUD_PROVIDERS_NAMES},
    )
    if batch:
        span_processor = BatchSpanProcessor(span_exporter=exporter)
    else:
        span_processor = SimpleSpanProcessor(span_exporter=exporter)
    tracer_provider.add_span_processor(span_processor)

    # Auto-instrument the provider SDKs with OpenInference instrumentors — one
    # per dedicated client (the openai one covers both the Responses and Chat
    # Completions APIs). Each is applied only if installed, so providers the app
    # doesn't use are skipped. Langfuse ingests OpenInference spans natively.
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
