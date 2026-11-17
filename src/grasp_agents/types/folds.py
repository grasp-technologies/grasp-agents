"""Compaction fold: a summarized span of the transcript log."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class FoldSpec(BaseModel):
    """
    A summarized span of the transcript log: messages ``[start, end)`` are
    replaced by ``summary`` in the model-facing view, preceded by the messages
    at the ``pinned`` log indices, kept verbatim: user messages, and the launch
    notes of background tasks still running when the fold was made.

    Folds index the immutable log, so a fold stays valid until the log is
    truncated below ``end`` (rollback drops any fold past the rewind point).
    Persisted with the checkpoint so a lossy summary need not be recomputed on
    resume.
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    summary: str
    pinned: list[int] = Field(default_factory=list[int])

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("FoldSpec span must be non-empty: end must exceed start")
        if any(not self.start <= i < self.end for i in self.pinned):
            raise ValueError("FoldSpec pinned indices must lie within the span")
        return self


class CompactionPlan(BaseModel, frozen=True):
    """
    A fold decided on, before its summary is written: the log from its start up
    to ``end``, keeping the messages at the ``pinned`` log indices verbatim.
    """

    end: int = Field(gt=0)
    pinned: list[int] = Field(default_factory=list[int])

    @model_validator(mode="after")
    def _check_pinned(self) -> Self:
        if any(not 0 <= i < self.end for i in self.pinned):
            raise ValueError("CompactionPlan pinned indices must lie within the span")
        return self

    def to_fold(self, summary: str) -> FoldSpec:
        """The fold this plan records once ``summary`` is written."""
        return FoldSpec(start=0, end=self.end, summary=summary, pinned=self.pinned)
