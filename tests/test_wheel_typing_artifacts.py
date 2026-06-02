"""Ensure published wheels ship PEP 561 typing artifacts."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("pyproject.toml").is_file(),
    reason="packaging metadata is only available in the source tree",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wheel_paths() -> list[Path]:
    dist_dir = _repo_root() / "dist"
    if not dist_dir.is_dir():
        return []
    return sorted(dist_dir.glob("*.whl"))


@pytest.fixture(scope="module")
def wheel_members() -> set[str]:
    """Return archive member paths from the newest built wheel, if any."""
    wheels = _wheel_paths()
    if not wheels:
        pytest.skip("no wheel found in dist/; run `uv build` before this test")
    with zipfile.ZipFile(wheels[-1]) as archive:
        return set(archive.namelist())


def test_wheel_includes_semanticcache_py_typed(wheel_members: set[str]) -> None:
    """The semanticcache package must advertise inline types via py.typed."""
    assert "semanticcache/py.typed" in wheel_members


def test_wheel_includes_fastapi_semcache_stub(wheel_members: set[str]) -> None:
    """The install-name compatibility package must ship its stub module."""
    assert "fastapi_semcache/__init__.pyi" in wheel_members


def test_wheel_includes_proxy_stub(wheel_members: set[str]) -> None:
    """Optional proxy helpers keep their dedicated stub file."""
    assert "semanticcache/proxy.pyi" in wheel_members
