"""
Convert grasp-agents InputItem[] → Gemini Content[].

The main entry point is ``items_to_provider_inputs``, which returns
``(system_instruction, contents)`` — system is extracted separately since
Gemini takes it as a config parameter, not inside the contents array.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

from grasp_agents.llm_providers._file_helpers import file_part_data
from grasp_agents.types.content import (
    BASE64_DATA_PREFIX,
    InputFile,
    InputImage,
    InputText,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    UnknownItem,
    WebSearchCallItem,
)

from . import (
    GeminiBlob,
    GeminiContent,
    GeminiFileData,
    GeminiFunctionCall,
    GeminiFunctionResponse,
    GeminiMediaResolution,
    GeminiMediaResolutionLevel,
    GeminiPart,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def items_to_provider_inputs(
    items: Sequence[InputItem],
) -> tuple[str | GeminiContent | None, list[GeminiContent]]:
    """
    Convert response items to Gemini content format.

    Returns ``(system_instruction, contents)`` where *system_instruction*
    is extracted from system/developer role items.
    """
    system_parts: list[str] = []
    contents: list[GeminiContent] = []
    call_id_to_name: dict[str, str] = {
        item.call_id: item.name
        for item in items
        if isinstance(item, FunctionToolCallItem)
    }
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
            contents.append(
                _tool_output_to_content(item, call_id_to_name=call_id_to_name)
            )
            i += 1

        elif isinstance(item, UnknownItem):
            # Only the OpenAI Responses provider can round-trip these.
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

    for part in item.content:
        if isinstance(part, InputText):
            parts.append(GeminiPart(text=part.text))

        elif isinstance(part, InputImage):
            parts.append(_image_to_part(part))

        else:  # InputFile — the remaining InputPart member
            parts.append(_file_to_part(part))

    if not parts:
        parts.append(GeminiPart(text=item.text))

    return GeminiContent(role="user", parts=parts)


_DETAIL_TO_MEDIA_RESOLUTION = {
    "low": GeminiMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "medium": GeminiMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
    "high": GeminiMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    "ultra_high": GeminiMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
}


def _image_to_part(img: InputImage) -> GeminiPart:
    level = _DETAIL_TO_MEDIA_RESOLUTION.get(img.detail)
    resolution = GeminiMediaResolution(level=level) if level else None

    if img.is_base64 and img.image_url:
        raw = img.image_url.removeprefix(
            BASE64_DATA_PREFIX.format(mime_type=img.mime_type)
        )
        return GeminiPart(
            inline_data=GeminiBlob(data=base64.b64decode(raw), mime_type=img.mime_type),
            media_resolution=resolution,
        )

    if img.is_url and img.image_url:
        return GeminiPart(
            file_data=GeminiFileData(file_uri=img.image_url, mime_type=img.mime_type),
            media_resolution=resolution,
        )

    if img.is_file_id and img.file_id:
        return GeminiPart(
            file_data=GeminiFileData(file_uri=img.file_id, mime_type=img.mime_type),
            media_resolution=resolution,
        )

    raise ValueError("InputImage must have a URL, base64 data, or file_id")


def _file_to_part(file: InputFile) -> GeminiPart:
    if file.file_data:
        data, mime_type = file_part_data(file.file_data, file.filename)
        return GeminiPart(
            inline_data=GeminiBlob(data=base64.b64decode(data), mime_type=mime_type)
        )
    if file.file_url:
        _, mime_type = file_part_data("", file.filename)
        return GeminiPart(
            file_data=GeminiFileData(file_uri=file.file_url, mime_type=mime_type)
        )

    raise ValueError("InputFile must have base64 file_data or a file_url")


def _tool_output_to_content(
    item: FunctionToolOutputItem,
    call_id_to_name: dict[str, str] | None = None,
) -> GeminiContent:
    if isinstance(item.output, list):
        output_str = "\n".join(
            part.text for part in item.output if isinstance(part, InputText)
        )
    else:
        output_str = item.output

    name = (call_id_to_name or {}).get(item.call_id, item.call_id)

    return GeminiContent(
        role="user",
        parts=[
            GeminiPart(
                function_response=GeminiFunctionResponse(
                    id=item.call_id,
                    name=name,
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
            if item.text or item.refusal:
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
    # Gemini has no refusal part — round-trip refusals as text so the model
    # sees them on the next turn and can correct course.
    text = "\n".join(t for t in (item.text, item.refusal) if t)
    part = GeminiPart(text=text)
    psf = item.provider_specific_fields
    if psf and "thought_signature" in psf:
        part.thought_signature = base64.b64decode(psf["thought_signature"])

    return part


def _tool_call_to_part(item: FunctionToolCallItem) -> GeminiPart:
    # Tolerate empty/None-ish arguments; they mean a no-arg call and must
    # not crash the request build.
    raw = item.arguments.strip()
    part = GeminiPart(
        function_call=GeminiFunctionCall(
            id=item.call_id,
            name=item.name,
            args=json.loads(raw) if raw and raw != "null" else {},
        )
    )
    psf = item.provider_specific_fields
    if psf and "thought_signature" in psf:
        part.thought_signature = base64.b64decode(psf["thought_signature"])

    return part
