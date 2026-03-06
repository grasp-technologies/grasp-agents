"""
Convert grasp-agents InputItem[] → Gemini Content[].

The main entry point is ``items_to_gemini_contents``, which returns
``(system_instruction, contents)`` — system is extracted separately since
Gemini takes it as a config parameter, not inside the contents array.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

from grasp_agents.types.content import (
    BASE64_DATA_PREFIX,
    InputImage,
    InputTextContentPart,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)

from . import (
    GeminiBlob,
    GeminiContent,
    GeminiFileData,
    GeminiFunctionCall,
    GeminiFunctionResponse,
    GeminiPart,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def items_to_provider_inputs(
    items: Sequence[InputMessageItem | OutputItem | FunctionToolOutputItem],
) -> tuple[str | GeminiContent | None, list[GeminiContent]]:
    """
    Convert response items to Gemini content format.

    Returns ``(system_instruction, contents)`` where *system_instruction*
    is extracted from system/developer role items.
    """
    system_parts: list[str] = []
    contents: list[GeminiContent] = []
    i = 0
    n = len(items)

    while i < n:
        item = items[i]

        if isinstance(item, InputMessageItem):
            if item.role in {"system", "developer"}:
                system_parts.append(item.text)
                i += 1
            else:
                contents.append(_input_to_user_content(item))
                i += 1

        elif isinstance(item, FunctionToolOutputItem):
            contents.append(_tool_output_to_content(item))
            i += 1

        else:
            # Collect consecutive assistant-side items into one Content
            group: list[OutputItem] = [item]
            i += 1
            while i < n and isinstance(
                items[i],
                (
                    ReasoningItem,
                    OutputMessageItem,
                    FunctionToolCallItem,
                    WebSearchCallItem,
                ),
            ):
                group.append(items[i])  # type: ignore[arg-type]
                i += 1
            contents.append(_output_group_to_content(group))

    system: str | GeminiContent | None = None
    if system_parts:
        system = "\n\n".join(system_parts)

    return system, contents


def _input_to_user_content(item: InputMessageItem) -> GeminiContent:
    parts: list[GeminiPart] = []

    for part in item.content_parts:
        if isinstance(part, InputTextContentPart):
            parts.append(GeminiPart(text=part.text))

        elif isinstance(part, InputImage):
            parts.append(_image_to_part(part))

    if not parts:
        parts.append(GeminiPart(text=item.text))

    return GeminiContent(role="user", parts=parts)


def _image_to_part(img: InputImage) -> GeminiPart:
    if img.is_base64 and img.image_url:
        raw = img.image_url.removeprefix(BASE64_DATA_PREFIX)
        return GeminiPart(
            inline_data=GeminiBlob(
                data=base64.b64decode(raw),
                mime_type="image/jpeg",
            )
        )
    if img.is_url and img.image_url:
        return GeminiPart(
            file_data=GeminiFileData(file_uri=img.image_url),
        )
    raise ValueError("InputImage must have either a URL or base64 data")


def _tool_output_to_content(item: FunctionToolOutputItem) -> GeminiContent:
    if isinstance(item.output_parts, list):
        output_str = "\n".join(
            part.text
            for part in item.output_parts
            if isinstance(part, InputTextContentPart)
        )
    else:
        output_str = item.output_parts

    return GeminiContent(
        role="user",
        parts=[
            GeminiPart(
                function_response=GeminiFunctionResponse(
                    id=item.call_id,
                    name=item.call_id,  # Gemini requires name; use call_id
                    response={"result": output_str},
                )
            )
        ],
    )


def _output_group_to_content(
    group: Sequence[OutputItem],
) -> GeminiContent:
    parts: list[GeminiPart] = []

    for item in group:
        if isinstance(item, ReasoningItem):
            parts.append(_reasoning_to_part(item))

        elif isinstance(item, OutputMessageItem):
            if item.text:
                text_part = _message_to_part(item)
                parts.append(text_part)

        elif isinstance(item, FunctionToolCallItem):
            fc_part = _tool_call_to_part(item)
            parts.append(fc_part)

        # WebSearchCallItem: no Gemini equivalent, skip

    return GeminiContent(role="model", parts=parts)


def _reasoning_to_part(item: ReasoningItem) -> GeminiPart:
    # Gemini puts thought_signature on text parts and function call parts,
    # not on thinking parts (weirdly).

    text = item.content_text or item.summary_text or ""
    return GeminiPart(text=text, thought=True)


def _message_to_part(item: OutputMessageItem) -> GeminiPart:
    part = GeminiPart(text=item.text)
    psf = item.provider_specific_fields
    if psf and "thought_signature" in psf:
        part.thought_signature = base64.b64decode(psf["thought_signature"])

    return part


def _tool_call_to_part(item: FunctionToolCallItem) -> GeminiPart:
    part = GeminiPart(
        function_call=GeminiFunctionCall(
            id=item.call_id,
            name=item.name,
            args=json.loads(item.arguments),
        )
    )
    psf = item.provider_specific_fields
    if psf and "thought_signature" in psf:
        part.thought_signature = base64.b64decode(psf["thought_signature"])

    return part
