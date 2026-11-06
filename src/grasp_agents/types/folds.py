"""Compaction fold: a summarized span of the transcript log."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class FoldSpec(BaseModel):
    """
    A summarized span of the transcript log: messages ``[start, end)`` are
    replaced by ``summary`` in the model-facing view, preceded by the user
    messages at the ``pinned`` log indices, kept verbatim.

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


class TokenAnchor(BaseModel):
    """
    The provider-reported input tokens of the last request, over the first
    ``message_count`` log messages: the base the next context-size estimate
    adds the newer messages to.
    """

    input_tokens: int = Field(gt=0)
    message_count: int = Field(ge=0)
