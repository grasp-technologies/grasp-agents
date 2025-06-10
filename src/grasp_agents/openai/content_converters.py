from collections.abc import Iterable

from ..typing.content import (
    Content,
    ContentPart,
    ContentPartImage,
    ContentPartText,
    ImageData,
)
from . import (
    OpenAIContentPartImageParam,
    OpenAIContentPartParam,
    OpenAIContentPartTextParam,
    OpenAIImageURL,
)

BASE64_PREFIX = "data:image/jpeg;base64,"


def image_data_to_str(image_data: ImageData) -> str:
    if image_data.type == "url":
        return str(image_data.url)
    if image_data.type == "base64":
        return f"{BASE64_PREFIX}{image_data.base64}"
    raise ValueError(f"Unsupported image data type: {image_data.type}")


def from_api_content(
    api_content: str | Iterable[OpenAIContentPartParam],
) -> "Content":
    if isinstance(api_content, str):
        return Content(parts=[ContentPartText(data=api_content)])

    content_parts: list[ContentPart] = []
    for api_content_part in api_content:
        content_part: ContentPart

        if api_content_part["type"] == "text":
            text_data = api_content_part["text"]
            content_part = ContentPartText(data=text_data)

        elif api_content_part["type"] == "image_url":
            url = api_content_part["image_url"]["url"]
            detail = api_content_part["image_url"].get("detail")
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


def to_api_content(
    content: Content,
) -> Iterable[OpenAIContentPartParam]:
    api_content: list[OpenAIContentPartParam] = []
    for content_part in content.parts:
        api_content_part: OpenAIContentPartParam
        if isinstance(content_part, ContentPartText):
            api_content_part = OpenAIContentPartTextParam(
                type="text", text=content_part.data
            )
        else:
            api_content_part = OpenAIContentPartImageParam(
                type="image_url",
                image_url=OpenAIImageURL(
                    url=image_data_to_str(content_part.data),
                    detail=content_part.data.detail,
                ),
            )
        api_content.append(api_content_part)

    return api_content
