"""
The ``mounts`` system-prompt section.

When :attr:`SessionContext.file_backend` is a
:class:`~grasp_agents.file_backend.CompositeFileBackend`, this section tells
the agent which mounts make up its file tree, what each holds, which are
read-only, and which one the sandbox's ``Bash`` and code tools can see. It
renders nothing for any other backend.

:class:`~grasp_agents.LLMAgent` registers it automatically. To customize it,
pass a section built with :func:`make_mounts_section` to
``agent.add_system_prompt_section(...)``; the same name replaces the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.content import CacheControl

MOUNTS_SECTION_NAME = "mounts"

_COPY_TOOL_NAME = "Copy"


def make_mounts_section(
    *,
    section_name: str = MOUNTS_SECTION_NAME,
    cache_control: CacheControl | None = None,
) -> SystemPromptSection:
    """
    Build the ``mounts`` section.

    Each mount renders as ``- `<path>`: <description> (<access>)``, with
    "visible to Bash and code execution" added for the mount whose backend
    includes ``ctx.environment``'s file backend. A closing line names the
    mounts that exist inside the sandbox, and another points at ``Copy``
    when the agent has that tool.
    """

    def compute(
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        agent_ctx: AgentContext | None = None,
        **_: Any,
    ) -> str | None:
        del exec_id
        from grasp_agents.file_backend.composite import (  # ruff: ignore[import-outside-top-level]
            CompositeFileBackend,
        )

        backend = ctx.file_backend if ctx is not None else None
        if not isinstance(backend, CompositeFileBackend):
            return None

        environment = ctx.environment if ctx is not None else None
        env_backend = environment.file_backend if environment is not None else None

        rows: list[str] = []
        visible: list[str] = []
        for mount in backend.mounts:
            flags = ["read-only" if mount.read_only else "read-write"]
            if env_backend is not None and mount.backend.includes(env_backend):
                flags.append("visible to Bash and code execution")
                visible.append(f"`{mount.path}`")
            description = f": {mount.description}" if mount.description else ""
            rows.append(f"- `{mount.path}`{description} ({', '.join(flags)})")

        lines = ["Your file tools work on one tree made of these mounts:", *rows]
        if env_backend is not None:
            if visible:
                verb = "exists" if len(visible) == 1 else "exist"
                lines.append(
                    f"Inside the sandbox where Bash and code run, only "
                    f"{', '.join(visible)} {verb}; the other mounts are not there."
                )
            else:
                lines.append(
                    "None of these mounts exists inside the sandbox where Bash "
                    "and code run."
                )
        if agent_ctx is not None and _COPY_TOOL_NAME in agent_ctx.tools:
            lines.append(
                "Use Copy to move files between mounts; their content stays "
                "out of the conversation."
            )
        return "<mounts>\n" + "\n".join(lines) + "\n</mounts>"

    return SystemPromptSection(
        name=section_name, compute=compute, cache_control=cache_control
    )


__all__ = ["MOUNTS_SECTION_NAME", "make_mounts_section"]
