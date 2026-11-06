"""
Prompt / context construction.

Everything that shapes what the model sees: the :class:`PromptBuilder` and its
section/attachment vocabulary (:class:`SystemPromptSection`,
:class:`InputAttachment`), the built-in section providers
(:func:`make_env_info_section`, :func:`make_untrusted_content_section`), and
context compaction (:class:`SummarizingCompactor`, :class:`ContextBudget`,
:class:`CollapseToolOutputsProjector`).
"""

from .compaction import (
    COMPACTION_GUIDANCE,
    COMPACTION_INSTRUCTIONS,
    COMPACTION_REQUEST_TAG,
    COMPACTION_SECTION_NAME,
    CollapseToolOutputsProjector,
    ContextBudget,
    SummarizingCompactor,
    collapse_tool_outputs,
    count_rounds,
    make_compaction_section,
)
from .env_section import (
    CURRENT_TIME_ATTACHMENT_NAME,
    ENV_INFO_SECTION_NAME,
    make_current_time_attachment,
    make_env_info_section,
)
from .projection import SUMMARY_TAG, apply_folds, fold_summary_text
from .prompt_builder import (
    InputAttachment,
    InputAttachmentCompute,
    PromptBuilder,
    SectionCompute,
    SystemPromptSection,
)
from .system_reminder import SYSTEM_REMINDER_TAG, wrap_in_system_reminder
from .untrusted_content import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    UNTRUSTED_CONTENT_SECTION_NAME,
    UNTRUSTED_CONTENT_TAG,
    make_untrusted_content_section,
    unwrap_untrusted,
    wrap_untrusted,
)

__all__ = [
    "COMPACTION_GUIDANCE",
    "COMPACTION_INSTRUCTIONS",
    "COMPACTION_REQUEST_TAG",
    "COMPACTION_SECTION_NAME",
    "CURRENT_TIME_ATTACHMENT_NAME",
    "ENV_INFO_SECTION_NAME",
    "SUMMARY_TAG",
    "SYSTEM_REMINDER_TAG",
    "UNTRUSTED_CONTENT_INSTRUCTION",
    "UNTRUSTED_CONTENT_SECTION_NAME",
    "UNTRUSTED_CONTENT_TAG",
    "CollapseToolOutputsProjector",
    "ContextBudget",
    "InputAttachment",
    "InputAttachmentCompute",
    "PromptBuilder",
    "SectionCompute",
    "SummarizingCompactor",
    "SystemPromptSection",
    "apply_folds",
    "collapse_tool_outputs",
    "count_rounds",
    "fold_summary_text",
    "make_compaction_section",
    "make_current_time_attachment",
    "make_env_info_section",
    "make_untrusted_content_section",
    "unwrap_untrusted",
    "wrap_in_system_reminder",
    "wrap_untrusted",
]
