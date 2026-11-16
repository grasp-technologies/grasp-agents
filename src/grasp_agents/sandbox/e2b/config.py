"""
JSON-friendly E2B config: declare a sandbox **template** (base image + ordered
build steps) and the **sandbox-create** options in one file, instead of passing
a bare ``template="name"`` string around.

Two pieces:

* :class:`E2BTemplateConfig` maps to E2B's programmatic template builder
  (``e2b.Template``): a base (an existing E2B template / python / image /
  dockerfile / …) plus an ordered list of build :data:`Step` s (pip / apt /
  run / copy / …). :meth:`build` returns its ``alias`` once a ready build of
  it exists: it reuses a ready build, waits for one under way, and builds when
  the latest build failed or there is none. :meth:`to_dockerfile` renders the
  definition without building (handy for review + tests).
* :class:`E2BEnvironmentConfig` is the whole environment: either a prebuilt
  ``template`` ref **or** an inline ``template_build`` (built on demand), plus
  the create-time options (roots, network, timeout, metadata, …). :meth:`build`
  realizes an :class:`E2BEnvironment`.

Example::

    cfg = load_e2b_config("e2b.json")
    env = await cfg.build()            # builds an inline template unless ready
    async with env as live:            # creates the sandbox
        ...

Template *building* is slow + cached, so it is a deliberate step here (run once;
reuse the alias), never silently on every sandbox create.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from e2b import BuildException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp_agents.sandbox.policy import NetworkPolicy

from ._handle import DEFAULT_EXEC_TIMEOUT, DEFAULT_WORKSPACE
from ._names import LABEL_PATTERN
from .templates import E2BTemplateAPI, E2BTemplateClient, TemplateBuildState

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


type Step = Annotated[
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
    The template's starting point — set at most one of these (default: E2B's
    ``from_base_image``). ``template`` starts from an existing E2B template, by
    name or id, and reuses its cached layers, so a few steps on top of a large
    template build fast. ``dockerfile`` is the full escape hatch: any Docker
    capability not modeled as a step can be expressed there.
    """

    model_config = ConfigDict(extra="forbid")

    template: str | None = None
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
            for n in (
                "template",
                "python",
                "image",
                "debian",
                "ubuntu",
                "node",
                "dockerfile",
            )
            if getattr(self, n) is not None
        ]
        if len(chosen) > 1:
            raise ValueError(f"TemplateBase: set at most one base, got {chosen}.")
        return self

    def start(self) -> TemplateBuilder:
        from e2b import Template  # ruff: ignore[import-outside-top-level]

        t = Template()
        if self.template is not None:
            return t.from_template(self.template)
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

# One build of a name at a time per event loop: E2B fails a build of a name
# when another build of it starts.
_build_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _build_lock(name: str) -> asyncio.Lock:
    locks = _build_locks.setdefault(asyncio.get_running_loop(), {})
    return locks.setdefault(name, asyncio.Lock())


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
        """
        Render the template as a Dockerfile (no build, no network). E2B cannot
        render a ``template`` base, so that raises ``ValueError``.
        """
        from e2b import Template  # ruff: ignore[import-outside-top-level]

        return Template.to_dockerfile(self._builder())

    async def build(
        self,
        *,
        templates: E2BTemplateAPI | None = None,
        wait_timeout: float = 1800.0,
        poll_interval: float = 5.0,
        **api: Any,
    ) -> str:
        """
        Make sure a ready build of the template ``alias`` exists and return
        ``alias``, the ref for ``create(template=...)``. The latest build of
        ``alias`` decides what happens:

        * ready: return at once, without building;
        * under way: poll it every ``poll_interval`` seconds until it ends,
          then return, or raise E2B's ``BuildException`` with E2B's reason if
          it failed; raise ``TimeoutError`` if it is still building after
          ``wait_timeout`` seconds;
        * failed, or no template has the name: build the template.

        With ``skip_cache`` it always builds, without E2B's layer cache. A
        build that fails raises ``BuildException``. A ready build is not
        compared with this definition: give a changed definition a new
        ``alias``, or set ``skip_cache``.

        Calls in one process build a name one at a time. Two processes that
        build the same new name at once can still make one of the builds fail,
        since E2B fails a build when another build of its name starts: that
        call raises ``BuildException``, and the next call uses the other build,
        or builds again if the latest build failed.

        ``templates`` is E2B's template API: by default an
        :class:`E2BTemplateClient` over ``api`` (``api_key``, ``domain``, …).
        """
        name = self.alias
        if name is None:
            raise ValueError(
                "E2BTemplateConfig.build() needs an alias to build the template as."
            )
        templates = templates or E2BTemplateClient(**api)
        async with _build_lock(name):
            state = None if self.skip_cache else await templates.latest_build(name)
            if state is not None and state.status == "building":
                state = await _build_finished(
                    templates, name, timeout=wait_timeout, poll_interval=poll_interval
                )
            if state is not None and state.status == "ready":
                return name
            await templates.build(
                self._builder(),
                name,
                tags=self.tags or None,
                cpu_count=self.cpu_count,
                memory_mb=self.memory_mb,
                skip_cache=self.skip_cache,
            )
        return name


async def _build_finished(
    templates: E2BTemplateAPI, name: str, *, timeout: float, poll_interval: float
) -> TemplateBuildState | None:
    """Poll the build of ``name`` under way until it ends; raise if it failed."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if loop.time() >= deadline:
            raise TimeoutError(
                f"E2B template {name!r} is still building after {timeout:g} s."
            )
        await asyncio.sleep(poll_interval)
        state = await templates.latest_build(name)
        if state is None or state.status != "building":
            break
    if state is not None and state.status == "failed":
        raise BuildException(
            f"The build of E2B template {name!r} failed: "
            f"{state.reason or 'E2B gave no reason'}"
        )
    return state


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
    allowed_domains: list[str] = Field(default_factory=list)
    denied_domains: list[str] = Field(default_factory=list)
    sandbox_timeout: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    secure: bool = True
    domain: str | None = None
    pause_on_exit: bool = False
    code_interpreter: bool = False
    default_timeout: float = DEFAULT_EXEC_TIMEOUT
    lazy: bool = False
    label: str | None = Field(default=None, pattern=LABEL_PATTERN)

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
        from .environment import (  # ruff: ignore[import-outside-top-level]
            e2b_environment,
        )

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
            allowed_domains=self.allowed_domains,
            denied_domains=self.denied_domains,
            env=self.env or None,
            metadata=self.metadata or None,
            api_key=api_key,
            domain=self.domain,
            secure=self.secure,
            pause_on_exit=self.pause_on_exit,
            code_interpreter=self.code_interpreter,
            default_timeout=self.default_timeout,
            lazy=self.lazy,
            label=self.label,
        )


def load_e2b_config(path: str | Path) -> E2BEnvironmentConfig:
    """Load + validate an :class:`E2BEnvironmentConfig` from a JSON file."""
    from pathlib import Path as _Path  # ruff: ignore[import-outside-top-level]

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
