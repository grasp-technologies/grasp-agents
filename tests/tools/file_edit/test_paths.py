"""
Unit tests for path resolution, allowed-roots enforcement, and the
sensitive-path deny list.

Uses :func:`tmp_path` (pytest builtin) as a real filesystem sandbox so
``Path.resolve(strict=True)`` and symlink-traversal checks actually
exercise kernel behavior rather than mocked paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from grasp_agents.tools.file_edit.paths import (
    PathAccessError,
    check_sensitive_path,
    has_binary_extension,
    is_blocked_device,
    resolve_safe,
)

# ---------------------------------------------------------------------------
# resolve_safe
# ---------------------------------------------------------------------------


def test_resolve_safe_accepts_path_under_root(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    target = tmp_path / "a" / "file.txt"
    target.write_text("x")

    resolved = resolve_safe(target, [tmp_path], must_exist=True)
    assert resolved == target.resolve()


def test_resolve_safe_rejects_path_outside_root(tmp_path: Path) -> None:
    other = tmp_path.parent
    with pytest.raises(PathAccessError, match="outside allowed roots"):
        resolve_safe(other / "some_file", [tmp_path], must_exist=False)


def test_resolve_safe_rejects_nonexistent_when_must_exist(
    tmp_path: Path,
) -> None:
    with pytest.raises(PathAccessError, match="does not exist"):
        resolve_safe(tmp_path / "nope.txt", [tmp_path], must_exist=True)


def test_resolve_safe_allows_nonexistent_when_not_must_exist(
    tmp_path: Path,
) -> None:
    """
    Write-to-new-file needs a path that resolves under the root but
    doesn't yet exist — this is the Write branch.
    """
    resolved = resolve_safe(tmp_path / "new.txt", [tmp_path], must_exist=False)
    assert resolved == (tmp_path / "new.txt").resolve()


def test_resolve_safe_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.txt").write_text("x")
    resolved = resolve_safe("~/a.txt", [tmp_path], must_exist=True)
    assert resolved == (tmp_path / "a.txt").resolve()


def test_resolve_safe_raises_when_no_allowed_roots() -> None:
    with pytest.raises(PathAccessError, match="No allowed_roots"):
        resolve_safe("/tmp/anything", [], must_exist=False)


@pytest.mark.skipif(os.name == "nt", reason="symlink permissions differ on Windows")
def test_resolve_safe_rejects_symlink_escape(tmp_path: Path) -> None:
    """
    A symlink inside the allowed root that points outside must be
    rejected — resolve() follows symlinks, so the check lands on the
    real target.
    """
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    link = root / "escape"
    link.symlink_to(outside)

    with pytest.raises(PathAccessError, match="outside allowed roots"):
        resolve_safe(link, [root], must_exist=True)


def test_resolve_safe_multiple_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "file.txt").write_text("x")

    resolved = resolve_safe(root_b / "file.txt", [root_a, root_b], must_exist=True)
    assert resolved == (root_b / "file.txt").resolve()


# ---------------------------------------------------------------------------
# is_blocked_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/dev/zero",
        "/dev/urandom",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/fd/0",
        "/proc/self/fd/0",
        "/proc/12345/fd/1",
    ],
)
def test_is_blocked_device_true(path: str) -> None:
    assert is_blocked_device(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/home/user/a.txt",
        "/tmp/file",
        "/dev/something-else",
        "/proc/cpuinfo",
    ],
)
def test_is_blocked_device_false(path: str) -> None:
    assert is_blocked_device(path) is False


# ---------------------------------------------------------------------------
# has_binary_extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("image.png", True),
        ("image.PNG", True),  # case-insensitive
        ("archive.tar.gz", True),  # suffix is .gz
        ("doc.pdf", True),
        ("script.py", False),
        ("readme.md", False),
        ("config.yaml", False),
        ("no_extension", False),
    ],
)
def test_has_binary_extension(name: str, expected: bool) -> None:
    assert has_binary_extension(name) is expected


# ---------------------------------------------------------------------------
# check_sensitive_path — system-path baseline (not overridable per session)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/etc/hosts",
        "/boot/grub.cfg",
        "/usr/lib/systemd/system/ssh.service",
        "/private/etc/hosts",  # macOS
        "/private/var/db/dscache",
        "/private/var/log/system.log",
        "/private/var/root/.profile",
    ],
)
def test_check_sensitive_system_path_denied(path: str) -> None:
    err = check_sensitive_path(Path(path))
    assert err is not None
    assert "sensitive system path" in err


@pytest.mark.parametrize("path", ["/var/run/docker.sock", "/run/docker.sock"])
def test_check_sensitive_docker_socket_denied(path: str) -> None:
    err = check_sensitive_path(Path(path))
    assert err is not None
    assert "sensitive system path" in err


def test_session_overrides_do_not_bypass_system_deny() -> None:
    """
    System-path denies are hard — a session opt-in is not the right
    level of authorisation for ``/etc/sudoers``.
    """
    p = Path("/etc/sudoers")
    err = check_sensitive_path(p, session_overrides={p})
    assert err is not None


# ---------------------------------------------------------------------------
# check_sensitive_path — user-dotfile additions (session-overridable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/home/user/.env",
        "/home/user/.env.local",
        "/home/user/.env.production",
        "/home/user/project/.env",
        "/home/user/.netrc",
        "/home/user/.pypirc",
    ],
)
def test_dotfile_denied_by_default(path: str) -> None:
    err = check_sensitive_path(Path(path))
    assert err is not None
    assert "credential-like file" in err


@pytest.mark.parametrize(
    "path",
    [
        "/home/user/.ssh/id_rsa",
        "/home/user/.ssh/config",
        "/home/user/.aws/credentials",
        "/home/user/.gnupg/private.key",
        "/home/user/.kube/config",
        "/home/user/.docker/config.json",
    ],
)
def test_dotfile_dir_denied_by_default(path: str) -> None:
    err = check_sensitive_path(Path(path))
    assert err is not None
    assert "credential-sensitive directory" in err


def test_dotfile_allowed_when_include_dotfiles_false() -> None:
    err = check_sensitive_path(Path("/home/user/.ssh/id_rsa"), include_dotfiles=False)
    assert err is None


def test_dotfile_allowed_via_session_overrides() -> None:
    p = Path("/home/user/.env")
    err = check_sensitive_path(p, session_overrides={p})
    assert err is None


def test_dotfile_overrides_are_path_specific() -> None:
    """Overriding one dotfile doesn't open up all dotfiles."""
    err = check_sensitive_path(
        Path("/home/user/.ssh/id_rsa"),
        session_overrides={Path("/home/user/.env")},
    )
    assert err is not None


def test_config_dir_is_not_denied_by_default() -> None:
    """
    ``.config/`` hosts routine editable content (``.config/fish/``,
    ``.config/nvim/``). Our deny list stops short of ``.config``.
    """
    assert check_sensitive_path(Path("/home/user/.config/fish/config.fish")) is None


def test_ordinary_file_allowed() -> None:
    assert check_sensitive_path(Path("/home/user/project/src/main.py")) is None
