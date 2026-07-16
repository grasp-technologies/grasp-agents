import logging
from collections.abc import AsyncIterator, Sequence
from itertools import chain
from typing import Any, Literal, cast

from grasp_agents.durability.checkpoints import CheckpointKind, ParallelCheckpoint
from grasp_agents.session_context import SessionContext
from grasp_agents.types.errors import ProcInputValidationError, ProcRunError
from grasp_agents.types.events import Event, ProcPacketOutEvent, ProcPayloadOutEvent
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import BranchError, Packet
from grasp_agents.utils.callbacks import is_method_overridden
from grasp_agents.utils.errors import format_error_chain, root_cause
from grasp_agents.utils.streaming import stream_concurrent

from .processor import Processor

logger = logging.getLogger(__name__)

type OnError = Literal["raise", "drop", "keep"]
"""How a fan-out reacts to a failed branch:

- ``"raise"`` — fail the whole run on the first failed branch (default).
- ``"drop"`` — drop failed branches; successful payloads are emitted compactly.
- ``"keep"`` — keep a ``None`` at each failed slot so payloads stay aligned to
  the inputs; the output packet's ``failed`` maps the failed indices to their
  :class:`BranchError`.
"""


class ParallelProcessor[InT, OutT, CtxT](Processor[InT, OutT, CtxT]):
    _checkpoint_kind = CheckpointKind.PARALLEL

    def __init__(
        self,
        subproc: Processor[InT, OutT, CtxT],
        *,
        ctx: SessionContext[CtxT] | None = None,
        on_error: OnError | None = None,
        drop_failed: bool | None = None,
        path: list[str] | None = None,
    ) -> None:
        # Need to set _subproc before __init__ because it
        # executes _propagate_to_children which calls subproc.on_adopted
        self._subproc = subproc

        super().__init__(
            name=subproc.name + "_par",
            ctx=ctx,
            recipients=subproc.recipients,
            max_retries=0,
            path=path,
            tracing_enabled=subproc.tracing_enabled,
            tracing_exclude_input_fields=subproc.tracing_exclude_input_fields,
            durability_enabled=subproc.durability_enabled,
        )

        self._in_type = subproc.in_type
        self._out_type = subproc.out_type

        self._on_error: OnError = self._resolve_on_error(on_error, drop_failed)
        self._keep_failed: dict[int, BranchError] = {}

    @staticmethod
    def _resolve_on_error(
        on_error: OnError | None, drop_failed: bool | None
    ) -> OnError:
        if on_error is not None and drop_failed is not None:
            raise ValueError(
                "Pass on_error or drop_failed, not both (drop_failed is deprecated: "
                "False -> on_error='raise', True -> on_error='drop')."
            )
        if on_error is not None:
            return on_error
        if drop_failed:
            return "drop"
        return "raise"

    # --- Checkpointing ---

    def _propagate_to_children(self) -> None:
        self._subproc.on_adopted(self)

    async def load_checkpoint(self) -> ParallelCheckpoint | None:
        checkpoint = await self._deserialize_checkpoint(self._ctx, ParallelCheckpoint)
        if checkpoint is not None:
            logger.info(
                "Loaded parallel checkpoint %s (%d/%d completed)",
                self._checkpoint_store_key(self._ctx),
                len(checkpoint.completed),
                len(checkpoint.input_packet.payloads),
            )
        return checkpoint

    async def save_checkpoint(
        self,
        *,
        input_packet: Packet[Any],
        completed: dict[int, Packet[Any]],
    ) -> None:
        checkpoint = ParallelCheckpoint(
            session_key=self._ctx.session_key,
            processor_name=self.name,
            input_packet=input_packet,
            completed=completed,
        )
        await self._serialize_checkpoint(self._ctx, checkpoint)

    # --- Core ---

    async def aclose(self) -> None:
        """
        Cascade session teardown to the subprocessor template.

        Per-run replicas are clones closed at the end of each run; the
        template itself may also have been run directly.
        """
        await self._subproc.aclose()

    @property
    def subproc(self) -> Processor[InT, OutT, CtxT]:
        return self._subproc

    @property
    def on_error(self) -> OnError:
        return self._on_error

    @property
    def drop_failed(self) -> bool:
        return self._on_error == "drop"

    def select_recipients_impl(
        self, output: OutT, *, exec_id: str
    ) -> Sequence[ProcName]:
        if is_method_overridden("select_recipients_impl", self._subproc, Processor):
            return self._subproc.select_recipients_impl(output=output, exec_id=exec_id)
        return cast("list[ProcName]", self.recipients or [])

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self.is_resumable:
            return None
        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

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
                message=f"ParallelProcessor {self.name} requires in_args",
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

    def _build_packet(self, outputs: list[OutT], exec_id: str) -> Packet[OutT]:
        if self._on_error != "keep":
            return super()._build_packet(outputs, exec_id)

        for output in outputs:
            self.validate_output(output, exec_id=exec_id)

        routing: list[Sequence[ProcName]] | None = None
        if self.recipients is not None:
            routing = []
            for output in outputs:
                # A failed (None) slot routes nowhere; successes route as usual.
                recipients: Sequence[ProcName] = (
                    self.select_recipients(output=output, exec_id=exec_id)
                    if output is not None
                    else []
                )
                routing.append(recipients)

        return Packet(
            sender=self.name,
            payloads=outputs,
            routing=routing,
            failed=self._keep_failed,
        )

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        step: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[Event[Any]]:
        self._keep_failed = {}

        # --- Resume from checkpoint ---
        checkpoint = await self.load_checkpoint()
        completed_map: dict[int, Packet[Any]] = {}

        if checkpoint is not None:
            input_packet = checkpoint.input_packet
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
        for idx, pkt in completed_map.items():
            out_packets_map[idx] = cast("Packet[OutT]", pkt)

        # Replicas get unique names ``"<subproc_name>_<i>"`` so checkpoint
        # keys, event sources, and printer output all distinguish them.
        pending_indices = [i for i in range(len(all_in_args)) if i not in completed_map]
        if pending_indices:
            replicas: dict[int, Processor[InT, OutT, CtxT]] = {}
            for i in pending_indices:
                rep = self._subproc.copy()
                rep.name = f"{self._subproc.name}_{i}"
                # ``on_adopted`` re-derives path from ``self.path`` + new
                # ``rep.name`` and refreshes ctx (already shared via
                # ``SessionContext.__deepcopy__``, but kept for symmetry).
                rep.on_adopted(self)
                replicas[i] = rep

            streams = [
                replicas[i].run_stream(
                    in_args=all_in_args[i],
                    exec_id=f"{exec_id}/{i}",
                    step=0,
                )
                for i in pending_indices
            ]

            merged = stream_concurrent(streams)

            try:
                async for stream_idx, event in merged:
                    real_idx = pending_indices[stream_idx]
                    if (
                        isinstance(event, ProcPacketOutEvent)
                        # match this stream's own replica, not just any replica
                        # name, so a nested same-named packet can't be
                        # miscaptured
                        and event.source == replicas[real_idx].name
                    ):
                        out_packets_map[real_idx] = event.data
                        completed_map[real_idx] = event.data
                        await self.save_checkpoint(
                            input_packet=input_packet,
                            completed=completed_map,
                        )
                    else:
                        yield event
            finally:
                # Replicas are per-run clones — unreachable after this run, so
                # their sessions (shells/kernels/bg tasks) end here.
                for rep in replicas.values():
                    try:
                        await rep.aclose()
                    except Exception:
                        logger.warning(
                            "Failed to close parallel replica %r",
                            rep.name,
                            exc_info=True,
                        )

            if merged.errors:
                failed = [pending_indices[e.index] for e in merged.errors]
                if self._on_error == "raise":
                    detail = "; ".join(
                        f"index {pending_indices[e.index]}: "
                        f"{format_error_chain(e.exception)}"
                        for e in merged.errors
                    )
                    raise ProcRunError(
                        proc_name=self.name,
                        exec_id=exec_id,
                        message=(
                            f"{len(merged.errors)}/{len(all_in_args)} parallel "
                            f"copies of '{self._subproc.name}' failed ({detail}). "
                            "Set on_error='drop' to drop failed copies, or "
                            "on_error='keep' to keep them as aligned None payloads."
                        ),
                    ) from merged.errors[0].exception

                logger.warning(
                    "ParallelProcessor %s: %d/%d copies failed (indices %s)",
                    self.name,
                    len(merged.errors),
                    len(all_in_args),
                    failed,
                )
                for e in merged.errors:
                    logger.warning(
                        "ParallelProcessor %s: %s failed copy %d",
                        self.name,
                        "keeping" if self._on_error == "keep" else "dropped",
                        pending_indices[e.index],
                        exc_info=e.exception,
                    )

                if self._on_error == "keep":
                    self._keep_failed = {
                        pending_indices[e.index]: BranchError(
                            error=format_error_chain(e.exception),
                            error_type=type(root_cause(e.exception)).__qualname__,
                        )
                        for e in merged.errors
                    }

        # Emit results in input order. In ``keep`` mode a failed copy stays a
        # ``None`` slot so payloads line up 1:1 with the inputs; ``raise`` never
        # reaches here and ``drop`` compacts the failed slots out.
        ordered = [out_packets_map[i] for i in sorted(out_packets_map)]
        if self._on_error != "keep":
            ordered = [p for p in ordered if p is not None]

        for p in self._join_payloads_from_packets(ordered):
            yield ProcPayloadOutEvent(data=p, source=self.name, exec_id=exec_id)
