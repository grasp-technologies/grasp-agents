"""
How a :class:`SandboxPolicy`'s egress maps onto E2B's sandbox network
settings, at create time and for a live sandbox's ``update_network``.

E2B allows an allowlist of hosts (exact, or ``*.``-wildcards) only next to a
deny of all other traffic, and denies only addresses, never hosts. So
``denied_domains`` (checked first, "denied wins") are enforced by leaving
them out of the allowlist, and a denied host an allowed wildcard still
covers cannot be enforced at all.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from e2b.sandbox.network import ALL_TRAFFIC

from grasp_agents.sandbox.policy import NetworkPolicy

if TYPE_CHECKING:
    from e2b import SandboxNetworkOpts

    from grasp_agents.sandbox.policy import SandboxPolicy


def e2b_egress(policy: SandboxPolicy) -> tuple[bool, SandboxNetworkOpts | None]:
    """
    ``(allow_internet_access, network)`` for ``policy``: ``network`` is
    ``None`` when the internet switch alone expresses it. Raises
    ``NotImplementedError`` for what E2B cannot enforce.
    """
    if policy.network is NetworkPolicy.LOOPBACK:
        raise NotImplementedError(
            "E2B does not implement network='loopback'. Use NetworkPolicy.NONE, "
            "ALLOWLIST or ALL."
        )
    if policy.network is NetworkPolicy.NONE:
        return False, None
    if policy.network is NetworkPolicy.ALL:
        return True, None

    denied = [domain.lower() for domain in policy.denied_domains]
    allowed = [
        domain.lower()
        for domain in policy.allowed_domains
        if not any(fnmatchcase(domain.lower(), pattern) for pattern in denied)
    ]
    for host in denied:
        covering = next((p for p in allowed if "*" in p and fnmatchcase(host, p)), None)
        if covering is not None:
            raise NotImplementedError(
                f"E2B cannot deny {host!r} while allowing {covering!r}: it denies "
                "only addresses, so a denied host must not fall under an allowed "
                "wildcard. List the allowed hosts one by one instead."
            )
    network: SandboxNetworkOpts = {"deny_out": [ALL_TRAFFIC]}
    if allowed:
        network["allow_out"] = allowed
    return True, network
