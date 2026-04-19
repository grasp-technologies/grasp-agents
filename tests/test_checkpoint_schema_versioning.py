"""
Tests for persisted-checkpoint schema versioning.

Verifies that:
- ProcessorCheckpoint / TaskRecord serialize with a schema_version field
- Absence of schema_version in legacy bytes defaults to CURRENT_SCHEMA_VERSION
- Future schema versions raise CheckpointSchemaError (not silently corrupt)
- _deserialize_checkpoint re-raises CheckpointSchemaError instead of returning None
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from grasp_agents.durability import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_VERSION_SUMMARIES,
    CheckpointSchemaError,
    InMemoryCheckpointStore,
    TaskRecord,
    TaskStatus,
)
from grasp_agents.durability.checkpoints import ProcessorCheckpoint, WorkflowCheckpoint
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.types.events import Event, ProcPayloadOutEvent
from grasp_agents.types.io import ProcName
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow


class _Appender(Processor[str, str, None]):
    """Minimal processor used only for exercising workflow checkpoint I/O."""

    def __init__(
        self, name: str, *, recipients: list[ProcName] | None = None
    ) -> None:
        super().__init__(name=name, recipients=recipients)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[str] | None = None,
        exec_id: str,
        ctx: RunContext[None],
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for inp in in_args or []:
            yield ProcPayloadOutEvent(
                data=f"{inp}->{self.name}", source=self.name, exec_id=exec_id
            )


class TestSchemaVersionConstants:
    def test_current_version_has_summary(self) -> None:
        """Every known version including current has a one-line summary."""
        assert CURRENT_SCHEMA_VERSION in SCHEMA_VERSION_SUMMARIES
        assert SCHEMA_VERSION_SUMMARIES[CURRENT_SCHEMA_VERSION].strip() != ""

    def test_summaries_cover_all_versions_up_to_current(self) -> None:
        """No gaps between 1 and CURRENT_SCHEMA_VERSION."""
        for v in range(1, CURRENT_SCHEMA_VERSION + 1):
            assert v in SCHEMA_VERSION_SUMMARIES


class TestProcessorCheckpointVersion:
    def test_default_schema_version_is_current(self) -> None:
        cp = ProcessorCheckpoint(session_id="s1", processor_name="p")
        assert cp.schema_version == CURRENT_SCHEMA_VERSION

    def test_round_trip_preserves_schema_version(self) -> None:
        cp = ProcessorCheckpoint(session_id="s1", processor_name="p")
        raw = cp.model_dump_json()
        loaded = ProcessorCheckpoint.model_validate_json(raw)
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION

    def test_legacy_bytes_without_field_default_to_current(self) -> None:
        """Checkpoints written before versioning should still load."""
        legacy = json.dumps(
            {
                "session_id": "s1",
                "processor_name": "p",
                "checkpoint_number": 0,
                "saved_at": "2026-04-17T00:00:00+00:00",
                "session_metadata": {},
            }
        )
        loaded = ProcessorCheckpoint.model_validate_json(legacy)
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION

    def test_future_version_raises_schema_error(self) -> None:
        future = json.dumps(
            {
                "schema_version": CURRENT_SCHEMA_VERSION + 1,
                "session_id": "s1",
                "processor_name": "p",
            }
        )
        with pytest.raises(CheckpointSchemaError) as exc_info:
            ProcessorCheckpoint.model_validate_json(future)
        msg = str(exc_info.value)
        assert str(CURRENT_SCHEMA_VERSION + 1) in msg
        assert "newer than" in msg

    def test_schema_error_is_not_validation_error(self) -> None:
        """CheckpointSchemaError must propagate past Pydantic ValidationError wrapping."""
        future = json.dumps(
            {
                "schema_version": CURRENT_SCHEMA_VERSION + 5,
                "session_id": "s",
                "processor_name": "p",
            }
        )
        with pytest.raises(CheckpointSchemaError):
            ProcessorCheckpoint.model_validate_json(future)
        assert not issubclass(CheckpointSchemaError, PydanticValidationError)
        assert not issubclass(CheckpointSchemaError, ValueError)


class TestTaskRecordVersion:
    def _record(self, **overrides: object) -> TaskRecord:
        defaults: dict[str, object] = {
            "task_id": "t1",
            "parent_session_id": "s1",
            "tool_call_id": "c1",
            "tool_name": "do_thing",
        }
        defaults.update(overrides)
        return TaskRecord(**defaults)  # type: ignore[arg-type]

    def test_default_schema_version_is_current(self) -> None:
        rec = self._record()
        assert rec.schema_version == CURRENT_SCHEMA_VERSION

    def test_round_trip_preserves_schema_version(self) -> None:
        rec = self._record(status=TaskStatus.PENDING)
        loaded = TaskRecord.model_validate_json(rec.model_dump_json())
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION
        assert loaded.task_id == "t1"

    def test_legacy_bytes_without_field_default_to_current(self) -> None:
        legacy = json.dumps(
            {
                "task_id": "t1",
                "parent_session_id": "s1",
                "tool_call_id": "c1",
                "tool_name": "do_thing",
                "status": "pending",
                "created_at": "2026-04-17T00:00:00+00:00",
                "updated_at": "2026-04-17T00:00:00+00:00",
            }
        )
        loaded = TaskRecord.model_validate_json(legacy)
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION

    def test_future_version_raises_schema_error(self) -> None:
        future = json.dumps(
            {
                "schema_version": CURRENT_SCHEMA_VERSION + 1,
                "task_id": "t1",
                "parent_session_id": "s1",
                "tool_call_id": "c1",
                "tool_name": "do_thing",
            }
        )
        with pytest.raises(CheckpointSchemaError):
            TaskRecord.model_validate_json(future)


class TestDeserializePropagation:
    """_deserialize_checkpoint must re-raise CheckpointSchemaError, not swallow it."""

    @pytest.mark.asyncio
    async def test_workflow_deserialize_reraises_schema_error(self) -> None:
        """Plant a future-versioned blob under the workflow's key; loading must raise."""
        store = InMemoryCheckpointStore()
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[_Appender("A"), _Appender("B")]
        )
        wf.setup_session("s-future")
        ctx: RunContext[None] = RunContext(state=None, checkpoint_store=store)

        future_blob = json.dumps(
            {
                "schema_version": CURRENT_SCHEMA_VERSION + 1,
                "session_id": "s-future",
                "processor_name": "wf",
                "completed_step": 0,
                "packet": {
                    "sender": "A",
                    "payloads": ["x"],
                    "routing": [["B"]],
                },
            }
        ).encode("utf-8")
        await store.save("workflow/s-future", future_blob)

        with pytest.raises(CheckpointSchemaError):
            await wf._deserialize_checkpoint(ctx, WorkflowCheckpoint)

    @pytest.mark.asyncio
    async def test_generic_corruption_still_starts_fresh(self) -> None:
        """Non-schema corruption still falls into the silent-reset path."""
        store = InMemoryCheckpointStore()
        wf = SequentialWorkflow[str, str, None](
            name="wf", subprocs=[_Appender("A"), _Appender("B")]
        )
        wf.setup_session("s-corrupt")
        ctx: RunContext[None] = RunContext(state=None, checkpoint_store=store)

        await store.save("workflow/s-corrupt", b"{not valid json")

        # Generic corruption should NOT propagate — silent reset is intact.
        result = await wf._deserialize_checkpoint(ctx, WorkflowCheckpoint)
        assert result is None
