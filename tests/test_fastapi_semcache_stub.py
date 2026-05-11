"""Regression tests for the ``fastapi_semcache`` install-name stub module."""

import importlib

import fastapi_semcache
import semanticcache


def test_fastapi_semcache_stub_imports_successfully() -> None:
    """Importing ``fastapi_semcache`` succeeds and re-exports the public API."""
    mod = importlib.import_module("fastapi_semcache")
    expected_names = [
        "CacheEntry",
        "CacheQuery",
        "CacheResult",
        "DEFAULT_MAX_BODY_BYTES",
        "ResponseShapeValidator",
        "ResponseValidationContext",
        "resolve_cache_scope",
        "SemanticCache",
        "SemanticCacheMiddleware",
        "create_semantic_cache_proxy_app",
        "get_cache_settings",
    ]
    for name in expected_names:
        assert hasattr(mod, name), f"fastapi_semcache is missing attribute: {name}"


def test_fastapi_semcache_stub_is_identical_to_semanticcache() -> None:
    """Every public name in the stub must be the identical object from semanticcache."""
    for name in fastapi_semcache.__all__:
        assert getattr(fastapi_semcache, name) is getattr(semanticcache, name), (
            f"fastapi_semcache.{name} is not the same object as semanticcache.{name}"
        )
