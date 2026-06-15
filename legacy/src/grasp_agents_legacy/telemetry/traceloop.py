import os
import threading
from logging import getLogger

from opentelemetry.sdk.resources import SERVICE_NAME
from traceloop.sdk import Traceloop

from .exporters import NoopExporter

logger = getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False


def init_traceloop(project_name: str = "grasp-agents"):
    global _initialized
    with _init_lock:
        if _initialized:
            return

        project_name_key = os.getenv("TELEMETRY_PROJECT_NAME_KEY", SERVICE_NAME)

        Traceloop.init(  # type: ignore
            app_name=project_name,
            exporter=NoopExporter(),
            resource_attributes={project_name_key: project_name},
            # disable_batch=False,
            # block_instruments={Instruments.OPENAI},
        )

        _initialized = True
