"""
JSON-friendly E2B config: declare a sandbox **template** (base image + ordered
build steps) and the **sandbox-create** options in one file, instead of passing
a bare ``template="name"`` string around.

Two pieces:

* :class:`E2BTemplateConfig` maps to E2B's programmatic template builder
  (``e2b.Template``): a base (python / image / dockerfile / …) plus an ordered
  list of build :data:`Step` s (pip / apt / run / copy / …). :meth:`build`
  builds it via the SDK and returns the template ref, **idempotently** — if its
  ``alias`` already exists the build is skipped. :meth:`to_dockerfile` renders
  the definition without building (handy for review + tests).
* :class:`E2BEnvironmentConfig` is the whole environment: either a prebuilt
  ``template`` ref **or** an inline ``template_build`` (built on demand), plus
  the create-time options (roots, network, timeout, metadata, …). :meth:`build`
  realizes an :class:`E2BEnvironment`.

Example::

    cfg = load_e2b_config("e2b.json")
    env = await cfg.build()            # builds the template if inline
    async with env as live:            # creates the sandbox
        ...

Template *building* is slow + cached, so it is a deliberate step here (run once;
reuse the alias), never silently on every sandbox create.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..policy import NetworkPolicy
from ._handle import DEFAULT_EXEC_TIMEOUT, DEFAULT_WORKSPACE

if TYPE_CHECKING:
    from pathlib import Path

    from e2b.template.main import TemplateBuilder, TemplateFinal

    from .environment import E2BEnvironment


# ---------------------------------------------------------------------------
# Build steps — pure-data models; each maps 1:1 to an ``e2b.TemplateBuilder``
# method, applied in order by :func:`_apply_step`.
# ---------------------------------------------------------------------------


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCmd(_Step):
    op: Literal["run_cmd"]
    command: str | list[str]
    user: str | None = None


class PipInstall(_Step):
    op: Literal["pip_install"]
    packages: str | list[str]


class AptInstall(_Step):
    op: Literal["apt_install"]
    packages: str | list[str]
    no_install_recommends: bool = False
    fix_missing: bool = False


class NpmInstall(_Step):
    op: Literal["npm_install"]
    packages: str | list[str] | None = None
    g: bool = False
    dev: bool = False


class Copy(_Step):
    op: Literal["copy"]
    src: str | list[str]
    dest: str
    user: str | None = None
    mode: int | None = None


class GitClone(_Step):
    op: Literal["git_clone"]
    url: str
    path: str | None = None
    branch: str | None = None
    depth: int | None = None
    user: str | None = None


class MakeDir(_Step):
    op: Literal["make_dir"]
    path: str | list[str]
    mode: int | None = None
    user: str | None = None


class SetEnvs(_Step):
    op: Literal["set_envs"]
    envs: dict[str, str]


class SetWorkdir(_Step):
    op: Literal["set_workdir"]
    workdir: str


class SetUser(_Step):
    op: Literal["set_user"]
    user: str


Step: TypeAlias = Annotated[
    RunCmd
    | PipInstall
    | AptInstall
    | NpmInstall
    | Copy
    | GitClone
    | MakeDir
    | SetEnvs
    | SetWorkdir
    | SetUser,
    Field(discriminator="op"),
]


def _no_steps() -> list[Step]:
    return []


def _apply_step(builder: TemplateBuilder, step: Step) -> TemplateBuilder:
    """Apply one config step to an e2b ``TemplateBuilder``; return the next one."""
    if isinstance(step, RunCmd):
        return builder.run_cmd(step.command, user=step.user)
    if isinstance(step, PipInstall):
        return builder.pip_install(step.packages)
    if isinstance(step, AptInstall):
        return builder.apt_install(
            step.packages,
            no_install_recommends=step.no_install_recommends,
            fix_missing=step.fix_missing,
        )
    if isinstance(step, NpmInstall):
        return builder.npm_install(step.packages, g=step.g, dev=step.dev)
    if isinstance(step, Copy):
        # e2b widens src to ``str | Path | list[str | Path]``; our config is
        # string-only (list is invariant, hence the cast).
        return builder.copy(
            cast("str | list[str | Path]", step.src),
            step.dest,
            user=step.user,
            mode=step.mode,
        )
    if isinstance(step, GitClone):
        return builder.git_clone(
            step.url,
            path=step.path,
            branch=step.branch,
            depth=step.depth,
            user=step.user,
        )
    if isinstance(step, MakeDir):
        return builder.make_dir(
            cast("str | list[str | Path]", step.path), mode=step.mode, user=step.user
        )
    if isinstance(step, SetEnvs):
        return builder.set_envs(step.envs)
    if isinstance(step, SetWorkdir):
        return builder.set_workdir(step.workdir)
    return builder.set_user(step.user)


# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------


class TemplateBase(BaseModel):
    """
    The template's starting image — set exactly one of these (default: E2B's
    ``from_base_image``). ``dockerfile`` is the full escape hatch: any Docker
    capability not modeled as a step can be expressed there.
    """

    model_config = ConfigDict(extra="forbid")

    python: str | None = None
    image: str | None = None
    image_username: str | None = None
    image_password: str | None = None
    debian: str | None = None
    ubuntu: str | None = None
    node: str | None = None
    dockerfile: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> TemplateBase:
        chosen = [
            n
            for n in ("python", "image", "debian", "ubuntu", "node", "dockerfile")
            if getattr(self, n) is not None
        ]
        if len(chosen) > 1:
            raise ValueError(f"TemplateBase: set at most one base image, got {chosen}.")
        return self

    def start(self) -> TemplateBuilder:
        from e2b import Template  # noqa: PLC0415

        t = Template()
        if self.python is not None:
            return t.from_python_image(self.python)
        if self.image is not None:
            return t.from_image(
                self.image, username=self.image_username, password=self.image_password
            )
        if self.debian is not None:
            return t.from_debian_image(self.debian)
        if self.ubuntu is not None:
            return t.from_ubuntu_image(self.ubuntu)
        if self.node is not None:
            return t.from_node_image(self.node)
        if self.dockerfile is not None:
            return t.from_dockerfile(self.dockerfile)
        return t.from_base_image()


# ---------------------------------------------------------------------------
# Template config
# ---------------------------------------------------------------------------


class E2BTemplateConfig(BaseModel):
    """A declarative E2B template: base + ordered build steps + build params."""

    model_config = ConfigDict(extra="forbid")

    base: TemplateBase = Field(default_factory=TemplateBase)
    steps: list[Step] = Field(default_factory=_no_steps)
    # Terminal readiness — ``start_cmd`` requires ``ready_cmd``.
    ready_cmd: str | None = None
    start_cmd: str | None = None
    # Build params (E2B ``Template.build``).
    alias: str | None = None
    tags: list[str] = Field(default_factory=list)
    cpu_count: int = 2
    memory_mb: int = 1024
    skip_cache: bool = False

    @model_validator(mode="after")
    def _start_needs_ready(self) -> E2BTemplateConfig:
        if self.start_cmd is not None and self.ready_cmd is None:
            raise ValueError("start_cmd requires ready_cmd (E2B set_start_cmd).")
        return self

    def _builder(self) -> TemplateBuilder | TemplateFinal:
        builder = self.base.start()
        for step in self.steps:
            builder = _apply_step(builder, step)
        if self.start_cmd is not None and self.ready_cmd is not None:
            return builder.set_start_cmd(self.start_cmd, self.ready_cmd)
        if self.ready_cmd is not None:
            return builder.set_ready_cmd(self.ready_cmd)
        return builder

    def to_dockerfile(self) -> str:
        """Render the template as a Dockerfile (no build, no network)."""
        from e2b import Template  # noqa: PLC0415

        return Template.to_dockerfile(self._builder())

    async def build(self, **api: Any) -> str:
        """
        Build the template and return its ref (alias or template id) for
        ``create(template=...)``. Idempotent: skips the build when ``alias``
        already exists (unless ``skip_cache``).
        """
        from e2b import AsyncTemplate  # noqa: PLC0415

        if (
            self.alias
            and not self.skip_cache
            and await AsyncTemplate.alias_exists(self.alias, **api)
        ):
            return self.alias
        info = await AsyncTemplate.build(
            self._builder(),
            alias=self.alias,
            tags=self.tags or None,
            cpu_count=self.cpu_count,
            memory_mb=self.memory_mb,
            skip_cache=self.skip_cache,
            **api,
        )
        return info.alias or info.template_id


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


class E2BEnvironmentConfig(BaseModel):
    """
    A full E2B environment spec: a template (prebuilt ref or inline build) plus
    the sandbox-create options. :meth:`build` realizes an :class:`E2BEnvironment`
    (building the template first when ``template_build`` is given).
    """

    model_config = ConfigDict(extra="forbid")

    # Provide exactly one: a prebuilt ref, or an inline definition to build.
    template: str | None = None
    template_build: E2BTemplateConfig | None = None

    allowed_roots: list[str] = Field(default_factory=lambda: [DEFAULT_WORKSPACE])
    deny_read: list[str] = Field(default_factory=list)
    allow_read: list[str] = Field(default_factory=list)
    deny_write: list[str] = Field(default_factory=list)
    network: NetworkPolicy = NetworkPolicy.NONE
    sandbox_timeout: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    secure: bool = True
    domain: str | None = None
    pause_on_exit: bool = False
    code_interpreter: bool = False
    default_timeout: float = DEFAULT_EXEC_TIMEOUT

    @model_validator(mode="after")
    def _one_template_source(self) -> E2BEnvironmentConfig:
        if self.template is not None and self.template_build is not None:
            raise ValueError(
                "Set either 'template' (a prebuilt ref) or 'template_build' (an "
                "inline definition to build), not both."
            )
        return self

    async def build(self, *, api_key: str | None = None) -> E2BEnvironment:
        """Build the template (if inline) and construct the environment."""
        from .environment import e2b_environment  # noqa: PLC0415

        api: dict[str, Any] = {}
        if api_key is not None:
            api["api_key"] = api_key
        if self.domain is not None:
            api["domain"] = self.domain

        template = self.template
        if self.template_build is not None:
            template = await self.template_build.build(**api)

        return e2b_environment(
            allowed_roots=self.allowed_roots,
            deny_read=self.deny_read,
            allow_read=self.allow_read,
            deny_write=self.deny_write,
            template=template,
            sandbox_timeout=self.sandbox_timeout,
            network=self.network,
            env=self.env or None,
            metadata=self.metadata or None,
            api_key=api_key,
            domain=self.domain,
            secure=self.secure,
            pause_on_exit=self.pause_on_exit,
            code_interpreter=self.code_interpreter,
            default_timeout=self.default_timeout,
        )


def load_e2b_config(path: str | Path) -> E2BEnvironmentConfig:
    """Load + validate an :class:`E2BEnvironmentConfig` from a JSON file."""
    from pathlib import Path as _Path  # noqa: PLC0415

    return E2BEnvironmentConfig.model_validate(
        json.loads(_Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "AptInstall",
    "Copy",
    "E2BEnvironmentConfig",
    "E2BTemplateConfig",
    "GitClone",
    "MakeDir",
    "NpmInstall",
    "PipInstall",
    "RunCmd",
    "SetEnvs",
    "SetUser",
    "SetWorkdir",
    "Step",
    "TemplateBase",
    "load_e2b_config",
]
