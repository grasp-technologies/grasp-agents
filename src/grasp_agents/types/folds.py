"""Compaction fold: a summarized span of the transcript log."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class FoldSpec(BaseModel):
    """
    A summarized span of the transcript log: messages ``[start, end)`` are
    replaced by ``summary`` in the model-facing view.

    Folds index the immutable log, so a fold stays valid until the log is
    truncated below ``end`` (rollback drops any fold past the rewind point).
    Persisted with the checkpoint so a lossy summary need not be recomputed on
    resume.
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    summary: str

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("FoldSpec span must be non-empty: end must exceed start")
        return self
