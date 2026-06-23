# pyright: reportUnusedImport=false
"""Native Google Gemini / Vertex AI provider for grasp-agents."""

import base64

from google.genai.types import (
    Blob as GeminiBlob,
)
from google.genai.types import (
    Candidate as GeminiCandidate,
)
from google.genai.types import (
    Content as GeminiContent,
)
from google.genai.types import (
    FileData as GeminiFileData,
)
from google.genai.types import (
    FinishReason as GeminiFinishReason,
)
from google.genai.types import (
    FunctionCall as GeminiFunctionCall,
)
from google.genai.types import (
    FunctionCallingConfig as GeminiFunctionCallingConfig,
)
from google.genai.types import (
    FunctionDeclaration as GeminiFunctionDeclaration,
)
from google.genai.types import (
    FunctionResponse as GeminiFunctionResponse,
)
from google.genai.types import (
    GenerateContentConfig as GeminiConfig,
)
from google.genai.types import (
    GenerateContentResponse as GeminiResponse,
)
from google.genai.types import (
    GenerateContentResponseUsageMetadata as GeminiUsageMetadata,
)
from google.genai.types import (
    GoogleSearch as GeminiGoogleSearch,
)
from google.genai.types import (
    GoogleSearchDict as GeminiGoogleSearchDict,
)
from google.genai.types import (
    HttpOptionsDict as GeminiHttpOptionsDict,
)
from google.genai.types import (
    Part as GeminiPart,
)
from google.genai.types import (
    PartMediaResolution as GeminiMediaResolution,
)
from google.genai.types import (
    PartMediaResolutionLevel as GeminiMediaResolutionLevel,
)
from google.genai.types import (
    SafetySettingDict as GeminiSafetySettingDict,
)
from google.genai.types import (
    Schema as GeminiSchema,
)
from google.genai.types import (
    ThinkingConfigDict as GeminiThinkingConfigDict,
)
from google.genai.types import (
    Tool as GeminiTool,
)
from google.genai.types import (
    ToolConfig as GeminiToolConfig,
)

from .gemini_llm import (
    GeminiLLM,
    GeminiLLMSettings,
    GeminiPlatform,
    GeminiVertexClientConfig,
)
