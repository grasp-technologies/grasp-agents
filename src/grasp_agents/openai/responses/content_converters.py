from collections.abc import Iterable

from openai.types.responses import (
    ResponseInputContent,
    ResponseInputImage,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputText,
    ResponseInputTextParam,
)

from ...typing.content import (
    Content,
    ContentPart,
    ContentPartImage,
    ContentPartText,
    ImageData,
)
from ..completions.content_converters import BASE64_PREFIX, image_data_to_str


def from_api_content(
    api_content: str | Iterable[ResponseInputContent],
) -> "Content":
    if isinstance(api_content, str):
        return Content(parts=[ContentPartText(data=api_content)])

    content_parts: list[ContentPart] = []
    for api_content_part in api_content:
        content_part: ContentPart

        if isinstance(api_content_part, ResponseInputText):
            text_data = api_content_part.text
            content_part = ContentPartText(data=text_data)

        elif isinstance(api_content_part, ResponseInputImage):
            url = api_content_part.image_url or ""
            detail = api_content_part.detail
            if url.startswith(BASE64_PREFIX):
                image_data = ImageData.from_base64(
                    base64_encoding=url.removeprefix(BASE64_PREFIX),
                    detail=detail,
                )
            else:
                image_data = ImageData.from_url(img_url=url, detail=detail)
            content_part = ContentPartImage(data=image_data)

        content_parts.append(content_part)  # type: ignore

    return Content(parts=content_parts)


def to_api_content(content: Content) -> ResponseInputMessageContentListParam:
    api_content: ResponseInputMessageContentListParam = []
    for content_part in content.parts:
        if isinstance(content_part, ContentPartText):
            api_content_part = ResponseInputTextParam(
                type="input_text", text=content_part.data
            )
        else:
            api_content_part = ResponseInputImageParam(
                type="input_image",
                detail=content_part.data.detail,
                image_url=image_data_to_str(content_part.data),
            )
        api_content.append(api_content_part)

    return api_content
