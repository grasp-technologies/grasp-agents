import fnmatch
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from openai import AsyncOpenAI, AsyncStream
from openai._types import NOT_GIVEN  # type: ignore[import]
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager as OpenAIAsyncChatCompletionStreamManager,
)
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.lib.streaming.chat import ChunkEvent as OpenAIChunkEvent
from pydantic import BaseModel

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings
from ..typing.tool import BaseTool
from . import (
    OpenAICompletion,
    OpenAICompletionChunk,
    OpenAIMessageParam,
    OpenAIParsedCompletion,
    OpenAIPredictionContentParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatText,
    OpenAIStreamOptionsParam,
    OpenAIToolChoiceOptionParam,
    OpenAIToolParam,
    OpenAIWebSearchOptions,
)
from .converters import OpenAIConverters

logger = logging.getLogger(__name__)

class OpenAIResponsesLLM(CloudLLMSettings, total=False):
    

@dataclass(frozen=True)
class OpenAIResponsesLLM(CloudLLM):
    
