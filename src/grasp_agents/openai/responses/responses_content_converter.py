from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any, List

from ... import OpenAIParsedResponse, OpenAIResponse


def extract_output_text(response: OpenAIResponse | OpenAIParsedResponse[Any]) -> str:
    parts: list[str] = []
    try:
        for out in getattr(response, "output", []) or []:
            if getattr(out, "type", "") == "message":
                for c in getattr(out, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        txt = getattr(c, "text", None)
                        if txt:
                            parts.append(txt)
    except Exception:
        pass
    return "\n".join(parts) if parts else ""


def extract_generated_images_base64(
    response: OpenAIResponse | OpenAIParsedResponse[Any],
) -> list[str]:
    results: list[str] = []
    try:
        for out in getattr(response, "output", []) or []:
            if getattr(out, "type", "") == "image_generation_call":
                result = getattr(out, "result", None)
                if isinstance(result, str) and result:
                    results.append(result)
    except Exception:
        pass
    return results


def extract_generated_images_bytes(
    response: OpenAIResponse | OpenAIParsedResponse[Any],
) -> list[bytes]:
    images_b64 = extract_generated_images_base64(response)
    images_bytes: list[bytes] = []
    for b64 in images_b64:
        try:
            images_bytes.append(base64.b64decode(b64))
        except Exception:
            # Skip invalid base64 payloads gracefully
            continue
    return images_bytes
