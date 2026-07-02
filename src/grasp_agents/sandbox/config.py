"""
JSON-friendly environment config: a pydantic :class:`EnvironmentConfig` that
deserializes a settings file into a :class:`SandboxPolicy` + confinement + exec
limits and builds a ready :class:`LocalEnvironment`.

Example::

    {
      "confinement": "auto",
      "filesystem": {
        "allowed_roots": ["."], "deny_read": [], "allow_read": [], "deny_write": []
      },
      "network": {"policy": "none", "allowed_domains": [], "denied_domains": []},
      "exec": {"inherit_host_env": true, "env_scrub": ["*_API_KEY"],
               "cpu_timeout": null, "max_memory_mb": null}
    }

    env = load_environment_config("sandbox.json").build()
    ctx = SessionContext(environment=env)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from .local.environment import LocalEnvironment, local_environment
from .local.supervisor import ProcessSupervisor, SupervisorLimits
from .policy import DEFAULT_ENV_SCRUB, NetworkPolicy

if TYPE_CHECKING:
    from pathlib import Path

Confinement = Literal["none", "seatbelt", "bwrap", "srt", "auto"]


class FilesystemConfig(BaseModel):
    """Filesystem section of the environment config."""

    model_config = ConfigDict(extra="forbid")

    allowed_roots: list[str] = Field(default_factory=lambda: ["."])
    readonly_roots: list[str] = Field(default_factory=list)
    deny_read: list[str] = Field(default_factory=list)
    allow_read: list[str] = Field(default_factory=list)
    deny_write: list[str] = Field(default_factory=list)
    include_dotfile_denylist: bool = True


class NetworkConfig(BaseModel):
    """Network section of the environment config."""

    model_config = ConfigDict(extra="forbid")

    policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_domains: list[str] = Field(default_factory=list)
    denied_domains: list[str] = Field(default_factory=list)


class ExecConfig(BaseModel):
    """Exec section: subprocess environment + resource ceilings."""

    model_config = ConfigDict(extra="forbid")

    inherit_host_env: bool = True
    # Interpreter the code-interpreter kernel launches with (default
    # ``sys.executable``); e.g. ``"/path/to/venv/bin/python"``.
    python: str | None = None
    # Distribution names required in the env (e.g. ``["torch"]``); verified
    # present at setup, or installed when ``provision`` is set.
    packages: list[str] = Field(default_factory=list)
    # Opt in to the framework creating the ``python`` venv (if absent) and
    # installing missing ``packages`` into it. Off by default.
    provision: bool = False
    env: dict[str, str] = Field(default_factory=dict)
    # ``None`` (omitted in JSON) keeps the secret-scrub denylist
    # (``DEFAULT_ENV_SCRUB``); an explicit ``[]`` disables scrubbing.
    env_scrub: list[str] | None = None
    overall_timeout: float | None = 600.0
    idle_timeout: float | None = None
    cpu_timeout: float | None = None
    max_memory_mb: int | None = None
    max_file_size_mb: int | None = None


class EnvironmentConfig(BaseModel):
    """A full, JSON-serializable environment spec; :meth:`build` realizes it."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    confinement: Confinement = "none"
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    # JSON key is ``exec``; the attribute is ``exec_`` to avoid shadowing the
    # builtin.
    exec_: ExecConfig = Field(default_factory=ExecConfig, alias="exec")

    def build(self) -> LocalEnvironment:
        """Construct the configured :class:`LocalEnvironment`."""
        ex = self.exec_
        supervisor = ProcessSupervisor(
            SupervisorLimits(
                overall_timeout=ex.overall_timeout,
                idle_timeout=ex.idle_timeout,
                cpu_timeout=ex.cpu_timeout,
                max_memory_mb=ex.max_memory_mb,
                max_file_size_mb=ex.max_file_size_mb,
            )
        )
        fs = self.filesystem
        return local_environment(
            allowed_roots=fs.allowed_roots,
            readonly_roots=fs.readonly_roots,
            deny_read=fs.deny_read,
            allow_read=fs.allow_read,
            deny_write=fs.deny_write,
            confinement=self.confinement,
            network=self.network.policy,
            allowed_domains=self.network.allowed_domains,
            denied_domains=self.network.denied_domains,
            include_dotfile_denylist=fs.include_dotfile_denylist,
            env=ex.env,
            env_scrub=ex.env_scrub if ex.env_scrub is not None else DEFAULT_ENV_SCRUB,
            inherit_host_env=ex.inherit_host_env,
            python=ex.python,
            packages=ex.packages,
            provision=ex.provision,
            supervisor=supervisor,
        )


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load + validate an :class:`EnvironmentConfig` from a JSON file."""
    from pathlib import Path as _Path  # noqa: PLC0415

    return EnvironmentConfig.model_validate(
        json.loads(_Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "EnvironmentConfig",
    "ExecConfig",
    "FilesystemConfig",
    "NetworkConfig",
    "load_environment_config",
]
