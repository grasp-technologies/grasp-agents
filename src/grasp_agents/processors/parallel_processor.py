import logging
from collections.abc import AsyncIterator, Sequence
from itertools import chain
from typing import Any

from grasp_agents.utils.streaming import stream_concurrent

from ..durability.checkpoints import ParallelCheckpoint
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..types.errors import ProcInputValidationError
from ..types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from ..types.io import InT, OutT, ProcName
from .processor import Processor

logger = logging.getLogger(__name__)


class ParallelProcessor(Processor[InT, OutT, CtxT]):
    def __init__(
        self, subproc: Processor[InT, OutT, CtxT], drop_failed: bool = False
    ) -> None:
        super().__init__(
            name=subproc.name + "_par",
            recipients=subproc.recipients,
            max_retries=0,
            tracing_enabled=subproc.tracing_enabled,
            tracing_exclude_input_fields=subproc.tracing_exclude_input_fields,
        )

        self._in_type = subproc.in_type
        self._out_type = subproc.out_type
        self._subproc = subproc

        self._drop_failed = drop_failed

        # This disables recipient selection in the subprocessor,
        # but preserves subproc.select_recipients_impl
        subproc.recipients = None

        # Session persistence (set via reset_session)
        self._session_id: str | None = None

    # --- Session persistence ---

    @property
    def resumable(self) -> bool:
        return True

    def reset_session(self, session_id: str) -> None:
        self._session_id = session_id

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self._session_id is not None:
            return None
        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

    @property
    def _checkpoint_store_key(self) -> str | None:
        if self._session_id is None:
            return None
        return f"parallel/{self._session_id}"

    async def _load_checkpoint(
        self, ctx: RunContext[CtxT]
    ) -> ParallelCheckpoint | None:
        store = ctx.store
        if store is None or self._checkpoint_store_key is None:
            return None
        data = await store.load(self._checkpoint_store_key)
        if data is None:
            return None
        checkpoint = ParallelCheckpoint.model_validate_json(data)
        logger.info(
            "Loaded parallel checkpoint %s (%d/%d completed)",
            self._session_id,
            len(checkpoint.completed),
            len(Packet[Any].model_validate(checkpoint.input_packet).payloads),
        )
        return checkpoint

    async def _save_checkpoint(
        self,
        ctx: RunContext[CtxT],
        *,
        input_packet: Packet[Any],
        completed: dict[int, dict[str, Any]],
    ) -> None:
        store = ctx.store
        if store is None or self._session_id is None:
            return
        assert self._checkpoint_store_key is not None
        checkpoint = ParallelCheckpoint(
            session_id=self._session_id,
            processor_name=self.name,
            input_packet=input_packet.model_dump(),
            completed=completed,
        )
        await store.save(
            self._checkpoint_store_key,
            checkpoint.model_dump_json().encode("utf-8"),
        )

    # --- Core ---

    @property
    def drop_failed(self) -> bool:
        return self._drop_failed

    def select_recipients_impl(
        self, output: OutT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        # Move recipient selection to the outer ParallelProcessor
        return self._subproc.select_recipients_impl(
            output=output, ctx=ctx, exec_id=exec_id
        )

    @property
    def subproc(self) -> Processor[InT, OutT, CtxT]:
        return self._subproc

    def _validate_in_args(
        self,
        chat_inputs: Any | None = None,
        in_args: list[InT] | None = None,
        *,
        exec_id: str,
    ) -> list[InT]:
        err_kwargs = {"proc_name": self.name, "exec_id": exec_id}
        if chat_inputs is not None:
            raise ProcInputValidationError(
                message=f"ParallelProcessor {self.name} does not support chat_inputs",
                **err_kwargs,
            )
        if in_args is None:
            raise ProcInputValidationError(
                message=f"ParallelProcessor {self.name} requires in_args to be provided",
                **err_kwargs,
            )

        return in_args

    def _join_payloads_from_packets(
        self, packets: Sequence[Packet[OutT] | None]
    ) -> list[OutT | None]:
        return list(
            chain.from_iterable(
                p.payloads if p is not None else [None] for p in packets
            )
        )

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> AsyncIterator[Event[Any]]:
        # --- Resume from checkpoint ---
        checkpoint = await self._load_checkpoint(ctx)
        completed_map: dict[int, dict[str, Any]] = {}

        if checkpoint is not None:
            input_packet = Packet[InT].model_validate(checkpoint.input_packet)
            all_in_args = list(input_packet.payloads)
            completed_map = dict(checkpoint.completed)
        else:
            all_in_args = self._validate_in_args(
                chat_inputs=chat_inputs, in_args=in_args, exec_id=exec_id
            )
            input_packet = Packet[InT](sender=self.name, payloads=all_in_args)

        # Pre-populate results from completed indices
        out_packets_map: dict[int, Packet[OutT] | None] = dict.fromkeys(
            range(len(all_in_args)), None
        )
        for idx_str, pkt_data in completed_map.items():
            idx = int(idx_str)  # JSON keys are strings
            out_packets_map[idx] = Packet[OutT].model_validate(pkt_data)

        # Create replicas only for uncompleted indices
        pending_indices = [i for i in range(len(all_in_args)) if i not in completed_map]
        if pending_indices:
            replicas = {i: self._subproc.copy() for i in pending_indices}
            streams = [
                replicas[i].run_stream(
                    in_args=all_in_args[i],
                    exec_id=f"{exec_id}/{i}",
                    ctx=ctx,
                )
                for i in pending_indices
            ]

            merged = stream_concurrent(streams)
            async for stream_idx, event in merged:
                real_idx = pending_indices[stream_idx]
                if (
                    isinstance(event, ProcPacketOutEvent)
                    and event.source == self._subproc.name
                ):
                    out_packets_map[real_idx] = event.data
                    completed_map[real_idx] = event.data.model_dump()
                    await self._save_checkpoint(
                        ctx,
                        input_packet=input_packet,
                        completed=completed_map,
                    )
                else:
                    yield event

            if merged.errors:
                failed = [pending_indices[e.index] for e in merged.errors]
                logger.warning(
                    "ParallelProcessor %s: %d/%d copies failed (indices %s)",
                    self.name,
                    len(merged.errors),
                    len(all_in_args),
                    failed,
                )

        # Emit results in input order
        out_packets = [out_packets_map[i] for i in sorted(out_packets_map)]
        if self.drop_failed:
            out_packets = [p for p in out_packets if p is not None]

        for p in self._join_payloads_from_packets(out_packets):
            yield ProcPayloadOutEvent(data=p, source=self.name, exec_id=exec_id)
