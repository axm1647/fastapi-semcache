"""Regression tests for lazy loading of optional proxy helpers."""

from __future__ import annotations

import builtins
import sys
from typing import NamedTuple

import fastapi_semcache
import pytest
import semanticcache


class _ReloadedSemanticCache(NamedTuple):
    module: object
    modules_before: set[str]


def _block_fastapi_import(
    name: str,
    globals: dict[str, object] | None = None,
    locals: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    """Raise when the top-level ``fastapi`` package is imported."""
    if level == 0 and (name == "fastapi" or name.startswith("fastapi.")):
        msg = "fastapi blocked for test"
        raise ImportError(msg)
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = builtins.__import__


@pytest.fixture
def isolated_semanticcache(monkeypatch: pytest.MonkeyPatch) -> _ReloadedSemanticCache:
    """Reload ``semanticcache`` after clearing cached submodules."""
    modules_before = set(sys.modules)
    to_drop = [
        name
        for name in list(sys.modules)
        if name == "semanticcache"
        or name == "fastapi_semcache"
        or name.startswith("semanticcache.")
    ]
    for name in to_drop:
        sys.modules.pop(name, None)
    sys.modules.pop("fastapi", None)
    monkeypatch.setattr(builtins, "__import__", _block_fastapi_import)
    import semanticcache as reloaded_semanticcache

    return _ReloadedSemanticCache(reloaded_semanticcache, modules_before)


def _newly_loaded_modules(state: _ReloadedSemanticCache) -> set[str]:
    return set(sys.modules) - state.modules_before


def test_core_import_does_not_require_fastapi(
    isolated_semanticcache: _ReloadedSemanticCache,
) -> None:
    """Importing the core package must not import FastAPI at module load time."""
    assert isolated_semanticcache.module.SemanticCacheMiddleware is not None
    assert "fastapi" not in _newly_loaded_modules(isolated_semanticcache)


def test_proxy_helper_loads_without_fastapi(
    isolated_semanticcache: _ReloadedSemanticCache,
) -> None:
    """Resolving the proxy helper must not import FastAPI until it is called."""
    factory = isolated_semanticcache.module.create_semantic_cache_proxy_app
    assert callable(factory)
    assert "fastapi" not in _newly_loaded_modules(isolated_semanticcache)


def test_proxy_factory_requires_fastapi_extra(
    isolated_semanticcache: _ReloadedSemanticCache,
) -> None:
    """Building a proxy app raises a clear error when FastAPI is unavailable."""

    class _MinimalCache:
        async def get(self, *args: object, **kwargs: object) -> object:
            _ = args, kwargs
            return None

        async def put(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    with pytest.raises(ImportError, match="proxy"):
        isolated_semanticcache.module.create_semantic_cache_proxy_app(
            upstream="http://127.0.0.1:8001",
            cache=_MinimalCache(),
        )


def test_fastapi_semcache_stub_lazy_proxy_export() -> None:
    """The install-name stub lazily re-exports the proxy helper."""
    factory = fastapi_semcache.create_semantic_cache_proxy_app
    assert factory is semanticcache.create_semantic_cache_proxy_app
