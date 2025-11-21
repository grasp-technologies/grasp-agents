from __future__ import annotations

from typing import Iterable

from ...typing.content import Content, ContentPartImage, ContentPartText
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import (
    ResponseInputImageParam,
)


def _image_to_data_url(image_part: ContentPartImage) -> str:
    """Return a data URL or remote URL for a ContentPartImage.

    - If the image is a URL, pass it through directly.
    - If the image is base64 data, wrap it in a PNG data URL.
    """
    if image_part.data.type == "url":
        return image_part.data.to_str()
    # Default to PNG mime; upstream can refine if needed
    return f"data:image/png;base64,{image_part.data.to_str()}"


def to_responses_content(content: Content | str) -> ResponseInputMessageContentListParam:
    """Convert our Content | str to a Responses input content list.

    Maps text parts to ResponseInputTextParam (type="input_text") and images to
    ResponseInputImageParam (type="input_image").
    """
    parts: ResponseInputMessageContentListParam = []
    if isinstance(content, Content):
        for p in content.parts:
            if isinstance(p, ContentPartText):
                parts.append(ResponseInputTextParam(type="input_text", text=p.data))
            elif isinstance(p, ContentPartImage):
                parts.append(
                    ResponseInputImageParam(
                        type="input_image",
                        detail=p.data.detail,
                        image_url=_image_to_data_url(p),
                    )
                )
        return parts

    # Plain string
    return [ResponseInputTextParam(type="input_text", text=str(content))]

