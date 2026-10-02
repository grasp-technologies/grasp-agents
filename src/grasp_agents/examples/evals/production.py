r"""
Demo: the grader in production, traced into Phoenix.

Grades the dataset's student answers as a service would — inputs only, no
teacher grades; eight students, each a session of three answers — with
tracing on, so that the online evaluation ``grader_evals.py:grader_online``
can score what it did::

    TELEMETRY_COLLECTOR_HTTP_ENDPOINT=$PHOENIX_BASE_URL/v1/traces \
        python src/grasp_agents/examples/evals/production.py
"""

import asyncio
import json

from opentelemetry import trace

from grasp_agents.examples.evals.grader_evals import (
    DATA,
    PRODUCTION_PROJECT,
    Submission,
    build_grader,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.telemetry import init_tracing
from grasp_agents.telemetry.phoenix import init_phoenix

STUDENTS = 8


async def serve() -> int:
    init_tracing(project_name=PRODUCTION_PROJECT)
    init_phoenix(project_name=PRODUCTION_PROJECT)
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line]
    for i, row in enumerate(rows):
        # A student's requests share a session.
        with SessionContext[None](session_key=f"student-{i % STUDENTS}"):
            grader = build_grader("v2")
            await grader.run(in_args=Submission.model_validate(row["input"]))
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is not None:
        force_flush()
    return len(rows)


if __name__ == "__main__":
    print(f"graded {asyncio.run(serve())} answers into {PRODUCTION_PROJECT!r}")
