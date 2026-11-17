"""
E2B's template API as :meth:`E2BTemplateConfig.build` uses it: the state of a
template's latest build, and a build.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal, Protocol

from e2b import AsyncTemplate, TemplateException
from e2b.api import AsyncApiClient, encode_path_param, handle_api_exception
from e2b.api.client.api.templates import (
    get_templates_aliases_alias,
    get_templates_template_id,
)
from e2b.api.client.models import TemplateAliasResponse, TemplateWithBuilds
from e2b.connection_config import ConnectionConfig
from e2b.template.types import BuildInfo

if TYPE_CHECKING:
    from e2b.template.main import TemplateBuilder, TemplateFinal


@dataclass(frozen=True)
class TemplateBuildState:
    """The latest build of an E2B template; ``reason`` is E2B's, for a failed one."""

    status: Literal["building", "ready", "failed"]
    reason: str | None = None


class E2BTemplateAPI(Protocol):
    """What :meth:`E2BTemplateConfig.build` needs from E2B's template API."""

    async def latest_build(self, name: str) -> TemplateBuildState | None:
        """The latest build of the template ``name``; ``None`` if no template has it."""
        ...

    async def build(
        self,
        template: TemplateBuilder | TemplateFinal,
        name: str,
        *,
        tags: list[str] | None,
        cpu_count: int,
        memory_mb: int,
        skip_cache: bool,
    ) -> None:
        """Build ``template`` as ``name`` and wait for it; raise if it fails."""
        ...


class E2BTemplateClient:
    """
    :class:`E2BTemplateAPI` over the ``e2b`` SDK. ``api`` are its connection
    params (``api_key``, ``domain``, …); E2B reads unset ones from the
    environment.
    """

    def __init__(self, **api: Any) -> None:
        self._api = api

    async def latest_build(self, name: str) -> TemplateBuildState | None:
        # E2B's build-status lookup does not always find a build by the
        # template's name, so the name is resolved to the template's id.
        async with AsyncApiClient(ConnectionConfig(**self._api)) as client:
            alias = await get_templates_aliases_alias.asyncio_detailed(
                encode_path_param(name), client=client
            )
            if alias.status_code == HTTPStatus.NOT_FOUND:
                return None
            if not isinstance(alias.parsed, TemplateAliasResponse):
                raise handle_api_exception(alias, TemplateException)
            template_id = alias.parsed.template_id
            found = await get_templates_template_id.asyncio_detailed(
                encode_path_param(template_id), client=client
            )
        if found.status_code == HTTPStatus.NOT_FOUND:
            return None
        if not isinstance(found.parsed, TemplateWithBuilds):
            raise handle_api_exception(found, TemplateException)
        if not found.parsed.builds:
            return None

        latest = max(found.parsed.builds, key=lambda build: build.created_at)
        if latest.status.value == "ready":
            return TemplateBuildState("ready")
        if latest.status.value != "error":
            return TemplateBuildState("building")
        status = await AsyncTemplate.get_build_status(
            BuildInfo(
                template_id=template_id,
                build_id=str(latest.build_id),
                name=name,
                alias=name,
            ),
            **self._api,
        )
        return TemplateBuildState(
            "failed", status.reason.message if status.reason else None
        )

    async def build(
        self,
        template: TemplateBuilder | TemplateFinal,
        name: str,
        *,
        tags: list[str] | None,
        cpu_count: int,
        memory_mb: int,
        skip_cache: bool,
    ) -> None:
        await AsyncTemplate.build(
            template,
            name,
            tags=tags,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            skip_cache=skip_cache,
            **self._api,
        )


__all__ = ["E2BTemplateAPI", "E2BTemplateClient", "TemplateBuildState"]
