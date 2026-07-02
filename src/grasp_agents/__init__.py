# pyright: reportUnusedImport=false
"""
grasp-agents public API.

This root namespace exposes the headline surface — what you need to build, run,
and stream an agent. Lower-tier helpers (prompt-section builders, loaders,
parsers, the loop ADT, error/event taxonomies, approval primitives, …) live in
their own subpackages and are imported from there, e.g.::

    from grasp_agents.agent import NextStep, ToolCallDecision
    from grasp_agents.context import make_env_info_section, SystemPromptSection
    from grasp_agents.types.llm_errors import LlmRateLimitError
    from grasp_agents.types.events import TurnStartEvent
    from grasp_agents.memory import scan_memdir
    from grasp_agents.skills import parse_slash_command
"""

# --- Agents / processors / workflows ---
from .agent import LLMAgent

# --- Sessions / durability / memory / skills ---
from .durability import AgentCheckpoint, CheckpointStore, InMemoryCheckpointStore

# --- Logging ---
from .grasp_logging import enable_verbose_stdout_logging, setup_logging
from .llm import LLM, FallbackLLM, LLMSettings, RetryPolicy
from .memory import MemoryEntry, MemoryProvider
from .printer import Printer, print_events
from .processors import ParallelProcessor, Processor
from .runner import Runner
from .session_context import RunContext, SessionContext
from .skills import SkillRegistry

# --- Tools ---
from .tools.agent_tool import AgentTool
from .tools.base import BaseTool
from .tools.function_tool import FunctionTool, function_tool
from .tools.processor_tool import ProcessorTool

# --- Messages / content / responses ---
from .types.content import (
    CacheControl,
    Content,
    InputImage,
    InputRenderable,
    InputRenderableModel,
)
from .types.events import Event, ProcPacketOutEvent, RunPacketOutEvent, StopReason
from .types.items import (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
)
from .types.packet import Packet
from .types.response import Response

# --- UI ---
from .ui.console import EventConsole, render_events
from .workflow import LoopedWorkflow, SequentialWorkflow, WorkflowProcessor

try:
    from .mcp import MCPClient, MCPServerSSE, MCPServerStdio
except ImportError:
    pass

__all__ = [
    "LLM",
    "AgentCheckpoint",
    "AgentTool",
    "AssistantMessage",
    "BaseTool",
    "CacheControl",
    "CheckpointStore",
    "Content",
    "DeveloperMessage",
    "Event",
    "EventConsole",
    "FallbackLLM",
    "FunctionTool",
    "InMemoryCheckpointStore",
    "InputImage",
    "InputRenderable",
    "InputRenderableModel",
    "LLMAgent",
    "LLMSettings",
    "LoopedWorkflow",
    "MCPClient",
    "MCPServerSSE",
    "MCPServerStdio",
    "MemoryEntry",
    "MemoryProvider",
    "Packet",
    "ParallelProcessor",
    "Printer",
    "ProcPacketOutEvent",
    "Processor",
    "ProcessorTool",
    "Response",
    "RetryPolicy",
    "RunContext",
    "RunPacketOutEvent",
    "Runner",
    "SequentialWorkflow",
    "SessionContext",
    "SkillRegistry",
    "StopReason",
    "SystemMessage",
    "UserMessage",
    "WorkflowProcessor",
    "enable_verbose_stdout_logging",
    "function_tool",
    "print_events",
    "render_events",
    "setup_logging",
]
