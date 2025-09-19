import os

from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from traceloop.sdk import Instruments, Traceloop

from phoenix.otel import register


def init_phoenix_tracing(
    project_name: str = "grasp-agents",
    batch: bool = False,
    block_instruments: set[Instruments] | None = None,
):
    # We choose to use instrumentation provided by Phoenix/OpenInference
    # for better compatibility with their tracing
    # Therefore, we do not use Traceloop's built-in instruments here.
    instruments_to_block = {
        Instruments.OPENAI,
        Instruments.ANTHROPIC,
        Instruments.VERTEXAI,
        Instruments.GOOGLE_GENERATIVEAI,
    }
    block_instruments = block_instruments or instruments_to_block
    collector_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    # Batching is recommended for production use to reduce overhead.
    # However, in some interactive environments (e.g., Jupyter notebooks),
    # batching may lead to lost spans if the process exits before the batch is sent.
    trace_provider = register(
        endpoint=collector_endpoint,
        project_name=project_name,
        set_global_tracer_provider=True,
        batch=batch,
    )

    # Processor to convert OpenTelemetry spans (emitted by OpenLLMetry)
    # to OpenInference format used by Phoenix.
    trace_provider.add_span_processor(OpenInferenceSpanProcessor())

    # OTLP exporter to send spans to Phoenix collector as opposed to Traceloop.
    exporter = OTLPSpanExporter(endpoint=collector_endpoint)
    Traceloop.init(  # type: ignore
        exporter=exporter,
        disable_batch=not batch,
        block_instruments={Instruments.OPENAI},
    )

    # Enable Phoenix instrumentation LiteLLM and OpenAI
    LiteLLMInstrumentor().instrument(tracer_provider=trace_provider)
    OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
    AnthropicInstrumentor().instrument(tracer_provider=trace_provider)
    VertexAIInstrumentor().instrument(tracer_provider=trace_provider)
    GoogleGenAIInstrumentor().instrument(tracer_provider=trace_provider)
